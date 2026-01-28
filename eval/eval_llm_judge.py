"""
eval_llm_judge.py - Evaluate completed inference results

Use LLM Judge to evaluate completed dialogue results.

Supported input formats:
1. Standard format: JSON file containing predicted_response and true_response
   { "results": [...], "dataset_path": "..." }
2. original_conversation format: JSON file containing complete dialogue data
   [ { "conversation_id": ..., "original_conversation": {...} }, ... ]

Output: JSON file with added llm_judge_scores

Evaluation dimensions:
1. Character Fidelity - 0-100 points
2. Response Quality - 0-100 points  
3. Emotion Awareness - 0-100 points
4. CoST Quality - 0-100 points
"""

import json
import argparse
import os
import threading
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_judge import LLMJudge, LLMJudgeConfig


def parse_system_message(system_message: str) -> Dict[str, str]:
    """
    Parse scenario and character information from system message
    
    Args:
        system_message: System prompt message
        
    Returns:
        Dictionary containing parsed information
    """
    info = {
        "scenario": "",
        "topic": "",
        "character_name": "",
        "occupation": "",
        "personality": "",
        "talking_style": "",
        "archetype": ""
    }
    
    try:
        if "Scenario:" in system_message:
            info["scenario"] = system_message.split("Scenario:")[1].split("Topic:")[0].strip()
        
        if "Topic:" in system_message:
            info["topic"] = system_message.split("Topic:")[1].split("Character Information:")[0].strip()
        
        if "Your name:" in system_message:
            info["character_name"] = system_message.split("Your name:")[1].split("\n")[0].strip()
        
        if "Your occupation:" in system_message:
            info["occupation"] = system_message.split("Your occupation:")[1].split("\n")[0].strip()
        
        if "Your personality:" in system_message:
            info["personality"] = system_message.split("Your personality:")[1].split("\n")[0].strip()
        
        if "Your talking style:" in system_message:
            info["talking_style"] = system_message.split("Your talking style:")[1].split("\n")[0].strip()
        
        if "Your archetype:" in system_message:
            archetype_text = system_message.split("Your archetype:")[1].split("Big Five:")[0].strip()
            # Extract archetype name (remove description)
            info["archetype"] = archetype_text.split(".")[0].strip()
    
    except Exception as e:
        print(f"  Warning: Error parsing system message: {str(e)}")
    
    return info


def extract_response_content(full_response: str) -> str:
    """
    Extract content within <Response> tags from complete response
    
    Args:
        full_response: Complete AI response (including Internal Monologue, etc.)
        
    Returns:
        Content within Response tags
    """
    import re
    
    # Match <Response>...</Response> or <Response>... (to end of file)
    pattern = r"<Response>(.*?)(?:</Response>|</assistant_end>|$)"
    match = re.search(pattern, full_response, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If no tags found, return original content
    return full_response.strip()


def load_original_data(original_file: Path) -> Dict[str, Dict]:
    """
    Load original dataset and create mapping from conversation_id to conversation history
    
    Args:
        original_file: Original dataset file path
        
    Returns:
        Mapping from conversation_id to conversation data
    """
    print(f"Loading original dataset: {original_file}")
    
    with open(original_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Build mapping: conversation_id -> conversation data
    conversation_map = {}
    for conv in data:
        conv_id = conv.get("conversation_id") or conv.get("id")
        if conv_id:
            conversation_map[conv_id] = conv
    
    print(f"Loaded {len(conversation_map)} conversations")
    return conversation_map


def evaluate_single_result(
    result: Dict,
    conversation_map: Dict[str, Dict],
    judge: LLMJudge,
    index: int,
    total: int,
    use_original_conversation: bool = False
) -> Dict:
    """
    Evaluate a single result (for parallel processing)
    
    Args:
        result: Single inference result
        conversation_map: Mapping from conversation_id to conversation data (None if use_original_conversation=True)
        judge: LLM Judge instance
        index: Current index
        total: Total count
        use_original_conversation: Whether to use original_conversation from result
        
    Returns:
        Result with added llm_judge_scores
    """
    conv_id = result.get('conversation_id')
    
    # If using original_conversation format, extract predicted_response from it
    if use_original_conversation:
        original_conv = result.get('original_conversation')
        if not original_conv:
            result['llm_judge_scores'] = {
                'error_message': 'Missing original_conversation field'
            }
            return result
        
        conversations = original_conv.get('conversations', [])
        if not conversations:
            result['llm_judge_scores'] = {
                'error_message': 'No conversation content in original_conversation'
            }
            return result
        
        # Extract last gpt response as predicted_response
        last_gpt_turn = None
        for turn in reversed(conversations):
            role = turn.get('role') or turn.get('from')
            if role in ['gpt', 'assistant']:
                last_gpt_turn = turn
                break
        
        if not last_gpt_turn:
            result['llm_judge_scores'] = {
                'error_message': 'Could not find last gpt response'
            }
            return result
        
        predicted_response = last_gpt_turn.get('value')
        error_message = None
        conv_data = original_conv  # Use original_conversation directly
    else:
        # Original logic: get predicted_response from result
        predicted_response = result.get('predicted_response')
        error_message = result.get('error_message')
        
        # If generation phase has error or no predicted response, skip evaluation
        if error_message or not predicted_response:
            result['llm_judge_scores'] = {
                'error_message': 'Error in generation phase, skipping evaluation'
            }
            return result
        
        # Get conversation info from original data
        if conv_id not in conversation_map:
            result['llm_judge_scores'] = {
                'error_message': f'Could not find original data for conversation ID {conv_id}'
            }
            return result
        
        conv_data = conversation_map[conv_id]
        conversations = conv_data.get('conversations', [])
        
        if not conversations:
            result['llm_judge_scores'] = {
                'error_message': 'No conversation content in original data'
            }
            return result
    
    try:
        # Parse system message
        system_message = ""
        for turn in conversations:
            role = turn.get('role') or turn.get('from')
            if role == 'system':
                system_message = turn.get('value')
                break
        
        if not system_message:
            result['llm_judge_scores'] = {
                'error_message': 'Could not find system message'
            }
            return result
        
        # Parse scenario and character information
        info = parse_system_message(system_message)
        
        # Build conversation history (excluding system and last gpt response)
        conversation_history = []
        for turn in conversations[1:]:  # Skip system
            role = turn.get('role') or turn.get('from')
            value = turn.get('value')
            
            if role in ['human', 'user']:
                conversation_history.append({
                    "role": "user",
                    "content": value
                })
            elif role in ['gpt', 'assistant']:
                # Only add up to second-to-last turn
                if turn != conversations[-1]:
                    # Extract Response content
                    response_content = extract_response_content(value)
                    conversation_history.append({
                        "role": "assistant",
                        "content": response_content
                    })
        
        # Extract Response content from predicted response
        predicted_response_content = extract_response_content(predicted_response)
        
        # Evaluate all dimensions
        scores = judge.evaluate_all(
            scenario=info["scenario"],
            topic=info["topic"],
            character_name=info["character_name"],
            occupation=info["occupation"],
            personality=info["personality"],
            talking_style=info["talking_style"],
            archetype=info["archetype"],
            conversation_history=conversation_history,
            ai_response=predicted_response_content
        )
        
        # Add scores to result
        result['llm_judge_scores'] = {
            'character_fidelity': scores.character_fidelity,
            'response_quality': scores.response_quality,
            'emotion_awareness': scores.emotion_awareness,
            'cost_quality': scores.cost_quality,
            'raw_response': scores.raw_response,
            'error_message': scores.error_message
        }
    
    except Exception as e:
        result['llm_judge_scores'] = {
            'error_message': f"Evaluation failed: {str(e)}"
        }
    
    return result


def evaluate_results(input_file: Path, output_file: Path, original_file: Path,
                     judge_config: LLMJudgeConfig, max_workers: int = 4):
    """
    Evaluate completed inference results (supports multi-worker parallelism)
    
    Args:
        input_file: Input file (result file containing predicted_response)
        output_file: Output file (result with added llm_judge_scores)
        original_file: Original dataset file (for getting conversation history)
        judge_config: LLM Judge configuration
        max_workers: Maximum number of parallel workers (default 4)
    """
    # Load input data
    print(f"Loading inference results: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get result list and determine format type
    use_original_conversation = False
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
        dataset_path = data.get('dataset_path', '')
    elif isinstance(data, list):
        results = data
        dataset_path = ''
        # Check if it's original_conversation format
        if results and 'original_conversation' in results[0]:
            use_original_conversation = True
            print("Detected original_conversation format, will extract conversation data from it")
    else:
        raise ValueError("Unsupported input file format")
    
    # If original_conversation format, no need to load additional original data
    if use_original_conversation:
        conversation_map = None
        print("Using original_conversation data from input file")
    else:
        # Load original dataset
        conversation_map = load_original_data(original_file)
    
    # Initialize LLM Judge (each worker creates its own instance)
    # Note: OpenAI client is thread-safe
    
    # Evaluate each result
    total = len(results)
    print(f"\nStarting LLM Judge evaluation of {total} conversations...")
    print(f"Parallelism: {max_workers} workers")
    
    evaluated_results = [None] * total  # Pre-allocate list to maintain order
    stats_lock = threading.Lock()
    stats = {
        'success': 0,
        'failed': 0,
        'cf_scores': [],
        'rq_scores': [],
        'ea_scores': [],
        'cq_scores': []
    }
    
    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create independent judge instance for each worker
        # Submit all tasks
        future_to_index = {}
        for i, result in enumerate(results):
            # Each task creates its own judge instance (thread-safe)
            judge = LLMJudge(judge_config)
            future = executor.submit(
                evaluate_single_result,
                result.copy(),  # Pass copy to avoid race conditions
                conversation_map,
                judge,
                i,
                total,
                use_original_conversation  # Pass format flag
            )
            future_to_index[future] = i
        
        # Use tqdm to display progress
        with tqdm(total=total, desc="Evaluation progress") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    evaluated_results[index] = result
                    
                    # Update statistics (thread-safe)
                    with stats_lock:
                        scores = result.get('llm_judge_scores', {})
                        # Check if all scores are not None
                        if (scores.get('character_fidelity') is not None and
                            scores.get('response_quality') is not None and
                            scores.get('emotion_awareness') is not None and
                            scores.get('cost_quality') is not None):
                            stats['success'] += 1
                            stats['cf_scores'].append(scores['character_fidelity'])
                            stats['rq_scores'].append(scores['response_quality'])
                            stats['ea_scores'].append(scores['emotion_awareness'])
                            stats['cq_scores'].append(scores['cost_quality'])
                        else:
                            stats['failed'] += 1
                    
                    pbar.update(1)
                    
                    # Display intermediate results every 10
                    if (stats['success'] + stats['failed']) % 10 == 0:
                        with stats_lock:
                            if stats['cf_scores']:
                                avg_cf = sum(stats['cf_scores']) / len(stats['cf_scores'])
                                avg_rq = sum(stats['rq_scores']) / len(stats['rq_scores'])
                                avg_ea = sum(stats['ea_scores']) / len(stats['ea_scores'])
                                avg_cq = sum(stats['cq_scores']) / len(stats['cq_scores'])
                                tqdm.write(f"  Current average scores: CF={avg_cf:.1f} RQ={avg_rq:.1f} "
                                          f"EA={avg_ea:.1f} CQ={avg_cq:.1f}")
                
                except Exception as e:
                    print(f"\nError processing index {index}: {str(e)}")
                    # Keep original result, add error message
                    if index < len(results):
                        result = results[index].copy()
                        result['llm_judge_scores'] = {
                            'error_message': f"Worker error: {str(e)}"
                        }
                        evaluated_results[index] = result
                        with stats_lock:
                            stats['failed'] += 1
                    pbar.update(1)
    
    print("\nLLM Judge evaluation completed!")
    
    # Save results
    print(f"Saving results to: {output_file}")
    output_data = {
        'dataset_path': dataset_path,
        'original_data_path': str(original_file),
        'results': evaluated_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Statistics of evaluation results
    print("\nEvaluation statistics:")
    print(f"  Total: {total}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    
    if stats['cf_scores']:
        # Calculate average scores
        print(f"\nAverage scores (out of 100):")
        print(f"  Character Fidelity: {sum(stats['cf_scores'])/len(stats['cf_scores']):.2f}")
        print(f"  Response Quality: {sum(stats['rq_scores'])/len(stats['rq_scores']):.2f}")
        print(f"  Emotion Awareness: {sum(stats['ea_scores'])/len(stats['ea_scores']):.2f}")
        print(f"  CoST Quality: {sum(stats['cq_scores'])/len(stats['cq_scores']):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate completed dialogue results using LLM Judge"
    )
    
    # Required parameters
    parser.add_argument("--input_file", type=str, required=True,
                       help="Input file path (inference results containing predicted_response)")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output file path (result with added llm_judge_scores)")
    parser.add_argument("--original_file", type=str, default=None,
                       help="Original dataset file path (for getting conversation history). Not needed if input file contains original_conversation field")
    
    # Parallel parameters
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Number of workers for parallel processing (default 4)")
    
    # LLM Judge parameters
    parser.add_argument("--judge_api_key", type=str, default=None,
                       help="API Key for LLM Judge (default reads from environment variable ARK_API_KEY)")
    parser.add_argument("--judge_base_url", type=str,
                       default="https://ark.cn-beijing.volces.com/api/v3",
                       help="API Base URL for LLM Judge")
    parser.add_argument("--judge_model_name", type=str,
                       default="ep-20260119001210-xsndk",
                       help="Model name used by LLM Judge (default DeepSeek-V3)")
    parser.add_argument("--judge_temperature", type=float, default=0.7,
                       help="Temperature parameter for LLM Judge")
    parser.add_argument("--judge_max_tokens", type=int, default=2048,
                       help="Maximum tokens for LLM Judge")
    
    args = parser.parse_args()
    
    # Validate parameters: if not original_conversation format, must provide original_file
    # Need to read input file first to determine format
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine if original_file is needed
    needs_original_file = True
    if isinstance(data, list) and data and 'original_conversation' in data[0]:
        needs_original_file = False
    
    if needs_original_file and not args.original_file:
        parser.error("--original_file parameter is required (unless input file contains original_conversation field)")
    
    # If no original_file provided, use input file itself as placeholder (won't be used)
    original_file_path = Path(args.original_file) if args.original_file else Path(args.input_file)
    
    # Create LLM Judge configuration
    judge_config = LLMJudgeConfig(
        api_key=args.judge_api_key or os.getenv('ARK_API_KEY', '0fec45ed-aa81-4355-bb03-ccc4d3a0dbc8'),
        base_url=args.judge_base_url,
        model_name=args.judge_model_name,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens
    )
    
    print("=" * 60)
    print("LLM Judge Batch Evaluation Tool (Multi-Worker Parallel Support)")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    if args.original_file:
        print(f"Original data: {args.original_file}")
    else:
        print(f"Original data: Using original_conversation from input file")
    print(f"Output file: {args.output_file}")
    print(f"Judge model: {judge_config.model_name}")
    print(f"Judge API: {judge_config.base_url}")
    print(f"Parallel Workers: {args.max_workers}")
    print("=" * 60)
    
    # Run evaluation
    evaluate_results(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        original_file=original_file_path,
        judge_config=judge_config,
        max_workers=args.max_workers
    )
    
    print("\n✅ Evaluation completed!")
