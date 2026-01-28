"""
Unified CoST (Chain-of-Social-Thought) 
"""

#  profile 
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_profile_dir = os.path.join(_current_dir, 'profile')

#  profile ， sys.path
# Python ('', 'profile') sys.path 
if os.path.exists(_profile_dir):
    # (remove existing paths)
    paths_to_remove = ['', _current_dir]
    for path in paths_to_remove:
        if path in sys.path:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
    
    # Add current directory to path if not already present
    if _current_dir not in sys.path:
        sys.path.append(_current_dir)

# Remove profile module if it exists
if 'profile' in sys.modules:
    try:
        profile_module = sys.modules['profile']
        if hasattr(profile_module, '__file__') and profile_module.__file__:
            if _profile_dir in profile_module.__file__ or _current_dir in profile_module.__file__:
                # Remove profile module if it exists
                del sys.modules['profile']
    except:
        pass

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
import wandb
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    output_dir: str = "./checkpoints"
    
    train_data_path: str = "./data/processed_final/train.jsonl"
    val_data_path: str = "./data/processed_final/val.jsonl"
    
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_steps: int = -1  # -1  num_epochs
    max_grad_norm: float = 1.0
    
    max_seq_length: int = 8192  #  8K， GPU （4K/16K）
    
    eval_strategy: str = "steps"  # "steps"  "epoch"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    fp16: bool = True
    dataloader_num_workers: int = 4
    logging_steps: int = 50
    report_to: str = "wandb"  # "wandb", "tensorboard", "none"
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    
    # Wandb 
    wandb_project: str = "amour-training"
    wandb_run_name: Optional[str] = None


class AmourDataset(Dataset):
    """Dataset for Amour training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        """
        
        Args:
            data_path: JSONL 
            tokenizer: Tokenizer
            max_length: 
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        logger.info(f"Loading data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=""):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    formatted_list = self._format_item(item)
                    if formatted_list:
                        # formatted_list ，
                        self.data.extend(formatted_list)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.data)} training samples")
    
    def _format_item(self, item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """
         Apeat 
        
        ： scenario, agent_a, agent_b, conversation 
        ：List[{"input": prompt, "output": target}] - 
        """
        try:
            scenario = item.get('scenario', {})
            conversation = item.get('conversation', [])
            
            if not conversation:
                return None
            
            samples = []
            
            for i, turn in enumerate(conversation):
                speaker = turn.get('speaker', '')
                message = turn.get('message', '')
                cost_thought = turn.get('CoST_thought', '').strip()
                emotion_before = turn.get('emotion_before', {})
                emotion_after = turn.get('emotion_after', {})
                relation_before = turn.get('relation_before', {})
                relation_after = turn.get('relation_after', {})
                observation = turn.get('observation', '')
                context = turn.get('context', '')
                
                if not message:
                    continue
                
                #  CoST_thought，
                if not cost_thought:
                    cost_thought = "Analyzing the situation and updating emotional state."
                
                #  agent（A  B）
                if speaker == 'A':
                    agent_info = item.get('agent_a', {})
                    other_agent_info = item.get('agent_b', {})
                    initial_relation = item.get('initial_relation', {}).get('a_to_b', {})
                elif speaker == 'B':
                    agent_info = item.get('agent_b', {})
                    other_agent_info = item.get('agent_a', {})
                    initial_relation = item.get('initial_relation', {}).get('b_to_a', {})
                else:
                    continue
                
                #  prompt
                prompt = self._build_prompt(
                    scenario=scenario,
                    agent_info=agent_info,
                    other_agent_info=other_agent_info,
                    conversation=conversation[:i],
                    current_observation=observation or context,
                    emotion_before=emotion_before,
                    relation_before=relation_before,
                    initial_relation=initial_relation
                )
                
                target = self._build_target(
                    cost_thought=cost_thought,
                    emotion_after=emotion_after,
                    relation_after=relation_after,
                    message=message
                )
                
                samples.append({
                    'input': prompt,
                    'output': target
                })
            
            # （）
            return samples if samples else None
            
        except Exception as e:
            logger.warning(f"Format error: {e}")
            return None
    
    def _build_prompt(self, scenario: Dict, agent_info: Dict, other_agent_info: Dict,
                     conversation: List[Dict], current_observation: str,
                     emotion_before: Dict, relation_before: Dict,
                     initial_relation: Dict) -> str:
        """Build prompt for training"""
        parts = []
        
        # 1. 
        if scenario:
            scenario_desc = scenario.get('description', '')
            scenario_topic = scenario.get('topic', '')
            if scenario_desc or scenario_topic:
                parts.append(f"## Scenario\n{scenario_desc}")
                if scenario_topic:
                    parts.append(f"Topic: {scenario_topic}")
        
        # 2. Agent Profile
        agent_profile = agent_info.get('profile', {}).get('static_traits', {})
        if agent_profile:
            profile_parts = ["## Your Character Profile"]
            if agent_profile.get('name'):
                profile_parts.append(f"Name: {agent_profile['name']}")
            if agent_profile.get('occupation'):
                profile_parts.append(f"Occupation: {agent_profile['occupation']}")
            if agent_profile.get('personality'):
                profile_parts.append(f"Personality: {agent_profile['personality']}")
            if agent_profile.get('talking_style'):
                profile_parts.append(f"Talking Style: {agent_profile['talking_style']}")
            if agent_profile.get('age'):
                profile_parts.append(f"Age: {agent_profile['age']}")
            if agent_profile.get('mbti'):
                profile_parts.append(f"MBTI: {agent_profile['mbti']}")
            parts.append("\n".join(profile_parts))
        
        # 3. 
        if emotion_before:
            emotion_parts = ["## Current Emotional State"]
            for key, value in emotion_before.items():
                emotion_parts.append(f"{key.capitalize()}: {value:.2f}")
            parts.append("\n".join(emotion_parts))
        
        # 4. 
        if relation_before:
            relation_parts = ["## Current Relationship"]
            for key, value in relation_before.items():
                if isinstance(value, (int, float)):
                    relation_parts.append(f"{key}: {value:.2f}")
                else:
                    relation_parts.append(f"{key}: {value}")
            parts.append("\n".join(relation_parts))
        
        # 5. 
        if conversation:
            conv_parts = ["## Conversation History"]
            #  agent  speaker ID（ agent_info ）
            current_speaker = None
            if agent_info.get('profile', {}).get('static_traits', {}).get('name'):
                #  agent_a/agent_b 
                # ： agent_info  speaker
                pass
            
            for turn in conversation:
                speaker = turn.get('speaker', '')
                msg = turn.get('message', '')
                if msg:
                    # ： A/B 
                    speaker_label = f"Agent {speaker}" if speaker in ['A', 'B'] else speaker
                    conv_parts.append(f"{speaker_label}: {msg}")
            parts.append("\n".join(conv_parts))
        
        # 6. /
        if current_observation:
            parts.append(f"## Current Context\n{current_observation}")
        
        # 7. 
        parts.append("## Task\nBased on the above information, you need to:")
        parts.append("1. Analyze the situation and update your emotional state (CoST Thought)")
        parts.append("2. Update your relationship state")
        parts.append("3. Generate an appropriate response")
        parts.append("\nPlease provide your response in the following format:")
        parts.append("```json\n{\n  \"emotion\": {...},\n  \"relation\": {...}\n}\n```\n\n[Internal Monologue]\nYour thought process here...\n\n[Response]\nYour response here...")
        
        return "\n\n".join(parts)
    
    def _build_target(self, cost_thought: str, emotion_after: Dict,
                     relation_after: Dict, message: str) -> str:
        """Build target output for training"""
        parts = []
        
        # 1. （JSON ）
        state_json = {
            "emotion": emotion_after,
            "relation": relation_after
        }
        parts.append(f"```json\n{json.dumps(state_json, indent=2, ensure_ascii=False)}\n```")
        
        # 2. 
        if cost_thought:
            parts.append(f"\n[Internal Monologue]\n{cost_thought}")
        
        # 3. 
        if message:
            parts.append(f"\n[Response]\n{message}")
        
        return "\n".join(parts)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        output_text = item['output']
        
        full_text = f"{input_text}\n\n{output_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # （ loss ）
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_length = input_encoding['input_ids'].shape[1]
        
        #  labels（-100 ， loss）
        labels = encoding['input_ids'].clone()
        labels[:, :input_length] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }


def compute_metrics(eval_pred):
    """"""
    predictions, labels = eval_pred
    
    #  perplexity
    # ，
    predictions = predictions.argmax(axis=-1)
    
    # （ -100）
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0.0
    
    return {
        'accuracy': accuracy
    }


def main():
    parser = argparse.ArgumentParser(description=' Amour ')
    parser.add_argument('--config', type=str, help='（JSON）')
    parser.add_argument('--model-name', type=str, default='meta-llama/Meta-Llama-3.1-8B',
                       help='')
    parser.add_argument('--train-data', type=str, default='./data/processed_final/train.jsonl',
                       help='')
    parser.add_argument('--val-data', type=str, default='./data/processed_final/val.jsonl',
                       help='')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                       help='')
    parser.add_argument('--num-epochs', type=int, default=3,
                       help='')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=None,
                       help='（）')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='')
    parser.add_argument('--wandb-project', type=str, default='amour-training',
                       help='Wandb ')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Wandb ')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                       help='')
    parser.add_argument('--no-wandb', action='store_true',
                       help=' wandb')
    #  --local-rank  --local_rank（DeepSpeed ）
    parser.add_argument('--local-rank', '--local_rank', type=int, default=-1,
                       dest='local_rank',
                       help='local rank（）')
    parser.add_argument('--deepspeed', type=str, default=None,
                       help='DeepSpeed（，DeepSpeed）')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_name=args.model_name,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else 4,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name or f"amour-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        resume_from_checkpoint=args.resume_from_checkpoint,
        report_to="none" if args.no_wandb else "wandb"
    )
    
    #  wandb
    if not args.no_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config)
        )
    
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    #  tokenizer 
        logger.info(f": {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    #  DeepSpeed
    use_deepspeed = args.deepspeed is not None if hasattr(args, 'deepspeed') else False
    
    #  DeepSpeed ZeRO stage（）
    zero_stage = None
    if use_deepspeed and args.deepspeed:
        try:
            import json
            with open(args.deepspeed, 'r') as f:
                deepspeed_config = json.load(f)
                if 'zero_optimization' in deepspeed_config:
                    zero_stage = deepspeed_config['zero_optimization'].get('stage', 0)
                logger.info(f"Using DeepSpeed ZeRO stage: {zero_stage}")
        except Exception as e:
            logger.warning(f"Failed to load DeepSpeed config: {e}")
    
    # GPU， device_map="auto"， Trainer/DeepSpeed 
    # GPU device_map="auto"
    num_gpus = torch.cuda.device_count()
    
    #  bf16（ fp16 ）
    use_bf16 = False
    if config.fp16 and torch.cuda.is_bf16_supported():
        use_bf16 = True
        logger.info(" bf16 ， bf16 （ fp16 ）")
        dtype = torch.bfloat16
    elif config.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    if use_deepspeed:
        logger.info(f" DeepSpeed ZeRO ， {num_gpus} GPU")
        # DeepSpeed ， device_map
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            use_cache=False,  # DeepSpeed 
            torch_dtype=dtype,
        )
    elif num_gpus > 1:
        logger.info(f" {num_gpus} GPU，GPU")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
        )
    else:
        logger.info("GPU")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            device_map="auto"
        )
    
    # ：ZeRO-3 
    #  ZeRO-3， use_reentrant=False
    #  ZeRO-2，
    if use_deepspeed:
        # DeepSpeed ，
        #  ZeRO-3 ，
        if zero_stage == 3:
            logger.info(" DeepSpeed ZeRO-3， TrainingArguments （）")
        else:
            logger.info(" DeepSpeed， TrainingArguments ")
    else:
        #  DeepSpeed 
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("")
    
    # # 2. （Embedding）（ DeepSpeed ）
    # if hasattr(model, "enable_input_require_grads"):
    #     model.enable_input_require_grads()
    # else:
    #     def make_inputs_require_grad(module, input, output):
    #         output.requires_grad_(True)
    #     model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
    # 3.  KV Cache（， checkpoint ）
    if hasattr(model, "config"):
        model.config.use_cache = False
        logger.info(" KV Cache ")
    
    # logger.info(" (use_reentrant=False) ")

    logger.info("...")
    train_dataset = AmourDataset(
        config.train_data_path,
        tokenizer,
        max_length=config.max_seq_length
    )
    
    logger.info("...")
    val_dataset = AmourDataset(
        config.val_data_path,
        tokenizer,
        max_length=config.max_seq_length
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    num_gpus = torch.cuda.device_count()
    #  DeepSpeed ZeRO stage 
    enable_gradient_checkpointing = (zero_stage is None or zero_stage != 3) if use_deepspeed else True
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        max_grad_norm=config.max_grad_norm,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        fp16=config.fp16 and not torch.cuda.is_bf16_supported(),  # bf16，bf16
        bf16=config.fp16 and torch.cuda.is_bf16_supported(),  # bf16
        dataloader_num_workers=config.dataloader_num_workers,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        seed=config.seed,
        logging_dir=f"{config.output_dir}/logs",

        # ---  ---
        # 1. ：ZeRO-3 ，
        #  ZeRO-2，
        #  ZeRO-3，， use_reentrant=False
        #  DeepSpeed ZeRO stage 
        gradient_checkpointing=enable_gradient_checkpointing,  # ZeRO-3 

        # GPU
        ddp_find_unused_parameters=False,  # GPU
        ddp_backend="nccl",  #  NCCL （NVIDIA GPU）
        local_rank=args.local_rank if hasattr(args, 'local_rank') else -1,
        # DeepSpeed 
        deepspeed=args.deepspeed if hasattr(args, 'deepspeed') and args.deepspeed else None,
    )
    
    if num_gpus > 1:
        logger.info(f"GPU:")
        logger.info(f" GPU : {num_gpus}")
        logger.info(f" GPU batch size: {config.per_device_train_batch_size}")
        logger.info(f" : {config.gradient_accumulation_steps}")
        logger.info(f" batch size: {num_gpus * config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    
    #  Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if config.load_best_model_at_end else None,
    )
    
    logger.info("...")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
    logger.info("...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info("Training complete!")
    logger.info(f"Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

