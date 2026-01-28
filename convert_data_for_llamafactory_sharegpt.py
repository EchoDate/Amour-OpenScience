#!/usr/bin/env python3
"""
Convert Amour training data to Llama-Factory ShareGPT format (improved version B)
ShareGPT format is more suitable for multi-turn conversations, Llama-Factory will automatically handle conversation history
"""
import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

def extract_agent_info(item: Dict) -> Tuple[Dict, Dict]:
    """
    Extract agent information (name, profile, etc.) from raw data
    
    Returns: (agent_a_info, agent_b_info)
    """
    agent_a = item.get('agent_a', {})
    agent_b = item.get('agent_b', {})
    
    def extract_name_and_profile(agent_data, speaker_label):
        profile = agent_data.get('profile', {})
        static_traits = profile.get('static_traits', {})
        
        # Extract name
        name = static_traits.get('name', '').strip()
        if not name:
            # Extract name from personality (usually first word of first sentence)
            personality = static_traits.get('personality', '')
            match = re.search(r'^([A-Z][a-z]+)\s+(?:is|was|has|s|are|were)', personality)
            if match:
                name = match.group(1)
        
        if not name:
            name = f"Agent_{speaker_label}"
        
        return {
            'name': name,
            'occupation': static_traits.get('occupation', ''),
            'personality': static_traits.get('personality', ''),
            'talking_style': static_traits.get('talking_style', ''),
        }
    
    agent_a_info = extract_name_and_profile(agent_a, 'A')
    agent_b_info = extract_name_and_profile(agent_b, 'B')
    
    return agent_a_info, agent_b_info


def convert_apeat_to_sharegpt(input_file: str, output_file: str):
    """
    Convert Apeat format to ShareGPT format
    
    ShareGPT format (includes human, observation, gpt):
    {
        "conversations": [
            {"from": "system", "value": "scenario info"},
            {"from": "human", "value": "user message"},
            {"from": "observation", "value": "observed info"},
            {"from": "gpt", "value": "state update + reasoning + response"},
            {"from": "human", "value": "user message"},
            {"from": "observation", "value": "observed info"},
            {"from": "gpt", "value": "state update + reasoning + response"}
        ]
    }
    
    For CoST format, gpt value contains: state update (JSON) + reasoning + response
    """
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                
                # Extract scenario information
                scenario = item.get('scenario', {})
                scenario_text = f"Scenario: {scenario.get('description', '')}\nTopic: {scenario.get('topic', '')}"
                
                # Extract conversation
                conversation = item.get('conversation', [])
                if not conversation:
                    continue
                
                # Extract agent information
                agent_a_info, agent_b_info = extract_agent_info(item)
                
                # Build system prompt (includes scenario info, role info)
                # Note: We're training the assistant (Agent B), so system should include:
                # - Scenario info
                # - Other person's (Agent A) name (for addressing only, not full profile)
                # - Own (Agent B) complete profile
                system_prompt = f"""{scenario_text}

Role Information:
- Other person's name: {agent_a_info['name']} (use this name when addressing them in responses)

- Your name: {agent_b_info['name']}
- Your occupation: {agent_b_info['occupation']}
- Your personality: {agent_b_info['personality']}
- Your talking style: {agent_b_info['talking_style']}

When responding to messages, you need to:
1. Update emotional state (JSON): emotion and relation
2. Think about your response ([Internal Monologue]): analyze the situation and determine how to respond
3. Respond to the other person ([Response]): natural and appropriate. When referring to the other person, use "{agent_a_info['name']}"."""
                
                conversations = []
                
                conversations.append({
                    "from": "system",
                    "value": system_prompt
                })
                
                for i, turn in enumerate(conversation):
                    speaker = turn.get('speaker', '')
                    message = turn.get('message', '')
                    observation = turn.get('observation', '').strip()
                    
                    if not message:
                        continue
                    
                    if speaker == 'A':
                        # Other person's message (human)
                        conversations.append({
                            "from": "human",
                            "value": message
                        })
                    else:
                        # Agent B's turn (gpt, with CoST format)
                        # Handle observation field
                        # If observation exists and differs from previous message (avoid duplication)
                        # Only add observation if it's meaningful
                        prev_message = conversation[i-1].get('message', '') if i > 0 else ''
                        
                        # Check if observation is meaningful and not duplicate
                        # If observation is significantly longer than previous or contains scenario info
                        if observation and observation != prev_message:
                            # Add observation (scene description or other contextual info)
                            if i == 0 or len(observation) > len(prev_message) * 1.5 or scenario_text[:50] in observation:
                                conversations.append({
                                    "from": "observation",
                                    "value": observation
                                })
                        
                        # Format agent response (CoST format, which agent to use as current)
                        assistant_value = format_cost_response(turn, current_speaker='B', other_person_name=agent_a_info['name'])
                        conversations.append({
                            "from": "gpt",
                            "value": assistant_value
                        })
                
                # Only add if conversations exist
                if conversations:
                    data.append({
                        "conversations": conversations
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Error: JSON decode error at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error: Failed to process line {line_num}: {e}")
                continue
    
    # Save as JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete: {len(data)} conversations saved to {output_file}")
    print(f"Format: ShareGPT (multi-turn)")
    
    return len(data)


def clean_cost_thought(cost_thought: str, current_speaker: str, other_person_name: str = 'the other person') -> str:
    """
    Clean CoST thought to replace agent references for training
    
    Args:
        cost_thought: original thought
        current_speaker: ('A'=first agent, 'B'=second agent)
    
    Transformations:
        - If training B: Agent_A -> other_person_name, Agent_B -> I
        - If training A: Agent_B -> other_person_name, Agent_A -> I
    """
    if not cost_thought:
        return cost_thought
    
    import re
    
    # Use other_person_name instead of Agent_A/Agent_B
    if current_speaker == 'B':
        # Training assistant, so Agent_A becomes other_person_name, Agent_B becomes I
        # Replace: "The speaker, Agent_B" -> "I"
        cleaned = re.sub(r'the speaker, Agent_B', 'I', cost_thought, flags=re.IGNORECASE)
        cleaned = re.sub(r'the speaker, Agent_A', other_person_name, cleaned, flags=re.IGNORECASE)
        # "The dialogue from Agent_X"
        cleaned = re.sub(r'the dialogue from Agent_A', f'{other_person_name}\'s dialogue', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'the dialogue from Agent_B', 'my dialogue', cleaned, flags=re.IGNORECASE)
        # General replacements: Agent_A -> other_person_name, Agent_B -> I
        cleaned = cleaned.replace('Agent_A', other_person_name).replace('Agent_B', 'I')
        cleaned = re.sub(r'Agent_A\'s', f'{other_person_name}\'s', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'Agent_B\'s', 'my', cleaned, flags=re.IGNORECASE)
    else:
        # Training user, so Agent_B becomes other_person_name, Agent_A becomes I
        cleaned = re.sub(r'the speaker, Agent_A', 'I', cost_thought, flags=re.IGNORECASE)
        cleaned = re.sub(r'the speaker, Agent_B', other_person_name, cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'the dialogue from Agent_B', f'{other_person_name}\'s dialogue', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'the dialogue from Agent_A', 'my dialogue', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace('Agent_B', other_person_name).replace('Agent_A', 'I')
        cleaned = re.sub(r'Agent_B\'s', f'{other_person_name}\'s', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'Agent_A\'s', 'my', cleaned, flags=re.IGNORECASE)
    
    # Remove remaining agent references (as fallback)
    cleaned = re.sub(r'Agent_[AB]', '', cleaned)
    
    return cleaned


def format_cost_response(turn: Dict, current_speaker: str = 'B', other_person_name: str = 'the other person') -> str:
    """
    Format CoST response: reasoning + state + response
    CoST: Context + user query --> Output : Reasoning + States Update + Respond
    
    Args:
        turn: turn data
        current_speaker: ('A'=first agent, 'B'=second agent)
    """
    cost_thought = turn.get('CoST_thought', '').strip()
    emotion_after = turn.get('emotion_after', {})
    relation_after = turn.get('relation_after', {})
    message = turn.get('message', '')
    
    # Clean agent references
    if cost_thought:
        cost_thought = clean_cost_thought(cost_thought, current_speaker, other_person_name)
    
    output_parts = []
    
    # 1. Reasoning - CoST thought
    if cost_thought:
        output_parts.append(f"[Internal Monologue]\n{cost_thought}")
    else:
        output_parts.append("[Internal Monologue]\nNo additional thoughts.")
    
    # 2. States Update - JSON format
    if emotion_after or relation_after:
        state_json = {
            "emotion": emotion_after,
            "relation": relation_after
        }
        output_parts.append(f"```json\n{json.dumps(state_json, indent=2, ensure_ascii=False)}\n```")
    
    # 3. Respond
    if message:
        output_parts.append(f"[Response]\n{message}")
    
    return "\n\n".join(output_parts)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Amour data to Llama-Factory ShareGPT format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_data_for_llamafactory_sharegpt.py --input train.jsonl --output amour_dataset_sharegpt.json
  
Format:
  - human: other person's message
  - observation: environmental or contextual information (optional)
  - gpt: agent response (reasoning, state update, and reply)
  
Notes:
  - Outputs ShareGPT format
  - Compatible with Llama-Factory
  - Add to dataset_info.json using sharegpt type
        """
    )
    parser.add_argument('--input', type=str, required=True, help='Input file path (JSONL format)')
    parser.add_argument('--output', type=str, required=True, help='Output file path (JSON format)')
    
    args = parser.parse_args()
    
    convert_apeat_to_sharegpt(args.input, args.output)


if __name__ == "__main__":
    main()
