from transformers import AutoTokenizer, AutoModelForCausalLM
from templates import system_prompt
from schema import ConversationDataset, OneRoundResults, OneRoundResult, ConversationTurn, SystemSetting
from openai import OpenAI
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from tqdm import tqdm


@dataclass
class Config:
    api_key: str
    base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float
    human_role: str = "user"
    gpt_role: str = "assistant"
    is_debug: bool = False


def preview_chat_template(
    system: str,
    user_prompt: str,
    tokenizer_path: str,
    chat_template_path: str = None
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    if chat_template_path:
        with open(chat_template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        tokenizer.chat_template = template_content
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_prompt})
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


def call_llm(
    messages: List[ConversationTurn] | List[Dict[str, str]],
    config: Config
) -> str:
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url
    )
    messages_list = get_message_list(messages, config)
    if config.model_name == "amour":
        response = client.chat.completions.create(
            model=config.model_name,
            messages=messages_list,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=["</assistant_end>", "<|eot_id|>", "<|end_of_text|>", "</Response>"]
        )
    else:
        response = client.chat.completions.create(
            model=config.model_name,
            messages=messages_list,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p
        )
    
    if not response.choices or len(response.choices) == 0:
        print(f"API returned empty choices. Response: {response}")
        return "ERROR: API returned empty choices"
    
    if not hasattr(response.choices[0], 'message') or response.choices[0].message is None:
        print(f"API response missing message. Response: {response}")
        return "ERROR: API response missing message"
    
    content = response.choices[0].message.content
    
    if content:
        for stop_token in ["</assistant_end>", "<|eot_id|>", "<|end_of_text|>", "</Response>"]:
            if stop_token in content:
                content = content[:content.find(stop_token) + len(stop_token)]
                break
    
    return content

def get_message_list(messages: List[ConversationTurn] | List[Dict[str, str]], config: Config) -> List[Dict[str, str]]:
    ret = []
    for message in messages:
        if isinstance(message, ConversationTurn):
            current_role = message.role
            value = message.value
        elif isinstance(message, Dict):
            current_role = message['role']
            value = message['content']
        else:
            raise ValueError(f"Unknown message type: {type(message)}")
        
        if current_role in ["user", "human"]:
            if config.model_name == "amour":
                ret.append({"role": "user", "content": value})
            else:
                ret.append({"role": config.human_role, "content": value})
        elif current_role in ["assistant", "gpt"]:
            if config.model_name == "amour":
                ret.append({"role": "assistant", "content": value})
            else:
                ret.append({"role": config.gpt_role, "content": value})
        elif current_role == "system":
            ret.append({"role": "system", "content": value})
        else:
            raise ValueError(f"Unknown role: {current_role}")
    return ret


def generate_one_round_for_dataset(config: Config, dataset: ConversationDataset) -> OneRoundResults:
    one_round_results = OneRoundResults(dataset_path=str(dataset.file_path), results=[])
    for conversation_data in tqdm(dataset.conversations):
        result = OneRoundResult(conversation_id=conversation_data.conversation_id, 
                                predicted_response=None, 
                                true_response=conversation_data.conversations[-1].value, 
                                error_message=None)
        if conversation_data.conversations[0].role != "system":
            result.error_message = "no system message"
        else:
            messages = conversation_data.conversations[:-1]
            response = call_llm(messages, config)
            result.predicted_response = response
        
        one_round_results.results.append(result)
    return one_round_results

def generate_reverse_one_round_for_dataset(config: Config, dataset: ConversationDataset) -> OneRoundResults:
    one_round_results = OneRoundResults(dataset_path=str(dataset.file_path), results=[])
    for conversation_data in tqdm(dataset.conversations):
        result = OneRoundResult(conversation_id=conversation_data.conversation_id, 
                                predicted_response=None, 
                                true_response=conversation_data.conversations[-1].value, 
                                error_message=None)
        if conversation_data.conversations[0].role != "system":
            result.error_message = "no system message"
        else:
            messages = conversation_data.conversations[:-1]
            if len(messages) >= 2:
                messages 
            response = call_llm(messages, config)