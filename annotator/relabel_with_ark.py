"""
CoST Relabeling Script (Ark Version): Relabel CoST format data based on DeepSeek-V3.2 (ByteDance Ark)
Implements all functional requirements in the task documentation

Main Features:
1. Conversation length standardization (3-12 turns, preferring 6-8 turns)
2. Regenerate first-person monologues using DeepSeek-V3.2 (CoST format)
3. Archetype initialization and Big-5 personality trait mapping
4. Dual-track state representation (tags + float values)
5. State update logic (plan → candidate → EMA → coupling)
6. Multi-worker parallel processing architecture
7. Support for three inference backends: online FaaS, batch FaaS, local API (implementing local API first)
"""
import json
import sys
import time
import random
import re
import math
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# Ark API imports
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("Warning: openai package not found. Install with: pip install openai")

try:
    from volcenginesdkarkruntime import Ark
    ArkClient = Ark
except ImportError:
    ArkClient = None
    print("Warning: volcenginesdkarkruntime package not found. Install with: pip install volcenginesdkarkruntime")

# Add paths for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
workspace_path = parent_dir.parent

for dir_path in [str(current_dir), str(parent_dir), str(workspace_path)]:
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)


# ==================== Ark API Wrapper ====================

class ArkLLMWrapper:
    """
    Ark API wrapper class, implements the same interface as AnnotatorLLMWrapper
    Uses ByteDance Ark's OpenAI-compatible interface
    Supports Session Cache feature to reduce API call costs
    """
    
    def __init__(self, api_key: str = None, model_name: str = None, 
                 base_url: str = None):
        """
        Initialize Ark API client
        
        Args:
            api_key: Ark API Key (if None, read from environment variable ARK_API_KEY)
            model_name: Model name (defaults to online inference model)
            base_url: API base URL
        """
        if OpenAI is None:
            raise ImportError("Need to install openai package: pip install openai")
        
        self.api_key = api_key or os.getenv('ARK_API_KEY', 'your-api-key-here')
        self.model_name = model_name
        self.base_url = base_url
        
        # Initialize OpenAI client (compatible with Ark API, for regular calls)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        # Initialize Ark client (for Session Cache)
        if ArkClient is not None:
            self.ark_client = ArkClient(
                base_url=self.base_url,
                api_key=self.api_key
            )
        else:
            self.ark_client = None
    
    def generate(self, system_message: str, prompt: str) -> Tuple[bool, str]:
        """
        Generate response
        
        Args:
            system_message: System message
            prompt: User prompt
            
        Returns:
            (success, response): Whether successful, generated response
        """
        try:
            # Build message list
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Call API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            
            # Extract response content
            if response and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return True, content.strip()
                else:
                    return False, "Generated response is empty"
            else:
                return False, "API response format error"
                
        except Exception as e:
            return False, f"API call error: {str(e)}"
    
    def generate_with_cache(self, system_message: str, prompt: str, 
                           previous_response_id: Optional[str] = None,
                           use_cache: bool = True,
                           verbose: bool = False) -> Tuple[bool, str, Optional[str], Optional[Dict[str, Any]]]:
        """
        Generate response using Session Cache
        
        Args:
            system_message: System message
            prompt: User prompt
            previous_response_id: Previous response ID (for cache reuse)
            use_cache: Whether to enable cache
            verbose: Whether to output detailed information
            
        Returns:
            (success, response, response_id, cache_info): Whether successful, generated response, response ID, cache info
        """
        cache_info = {}
        
        if self.ark_client is None:
            # If Ark client is unavailable, fall back to regular call
            if verbose:
                print(f"  ⚠️  Ark client unavailable, falling back to regular call")
            success, response = self.generate(system_message, prompt)
            return success, response, None, None
        
        try:
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "thinking": {"type": "disabled"}
            }
            
            # If cache is enabled, add cache parameters
            if use_cache:
                request_params["caching"] = {"type": "enabled"}
            
            # If there's a previous_response_id, use it to reuse cache
            # Note: When using previous_response_id, only pass new user message, system message is already in cache
            if previous_response_id:
                # When reusing cache, only pass new user message
                request_params["previous_response_id"] = previous_response_id
                request_params["input"] = [{"role": "user", "content": prompt}]
                cache_info['cache_mode'] = 'reused'
                cache_info['previous_response_id'] = previous_response_id
                cache_info['input_tokens_estimated'] = len(prompt) // 3  # Simple estimate: only count new user message
                if verbose:
                    print(f"  Cache mode: reused (previous_response_id: {previous_response_id})")
                    print(f"  Estimated input tokens: {cache_info['input_tokens_estimated']} (user message only)")
            else:
                # First call, pass complete messages (including system and user)
                input_messages = []
                if system_message:
                    input_messages.append({"role": "system", "content": system_message})
                input_messages.append({"role": "user", "content": prompt})
                request_params["input"] = input_messages
                cache_info['cache_mode'] = 'created'
                cache_info['input_tokens_estimated'] = (len(system_message) + len(prompt)) // 3
                if verbose:
                    print(f"  Cache mode: creating new cache")
                    print(f"  Estimated input tokens: {cache_info['input_tokens_estimated']} (system + user)")
            
            # Call API
            if verbose:
                print(f"  Calling API: model={self.model_name}, cache={'enabled' if use_cache else 'disabled'}")
            response = self.ark_client.responses.create(**request_params)
            
            # Extract response content and usage information
            usage_info = None
            if hasattr(response, 'usage'):
                usage_info = response.usage
                if isinstance(usage_info, dict):
                    cache_info['usage'] = usage_info
                elif hasattr(usage_info, 'model_dump'):
                    cache_info['usage'] = usage_info.model_dump()
                elif hasattr(usage_info, '__dict__'):
                    cache_info['usage'] = usage_info.__dict__
            
            # Extract response content
            # Ark API response format may differ, need to try multiple extraction methods
            content = None
            response_id = None
            
            # Method 1: Try to access text attribute (Ark API's main attribute)
            if response and hasattr(response, 'text'):
                try:
                    text_value = response.text
                    if text_value is not None:
                        # text may be a string, object, or list
                        if isinstance(text_value, str):
                            content = text_value
                        elif isinstance(text_value, list) and len(text_value) > 0:
                            # text may be a list, take first element
                            first_item = text_value[0]
                            if isinstance(first_item, str):
                                content = first_item
                            elif hasattr(first_item, 'content'):
                                content = first_item.content
                            elif hasattr(first_item, 'text'):
                                content = first_item.text
                            else:
                                content = str(first_item)
                        elif hasattr(text_value, 'content'):
                            content = text_value.content
                        elif hasattr(text_value, 'text'):
                            content = text_value.text
                        else:
                            # Try to convert to string
                            content = str(text_value)
                        if verbose and not content:
                            print(f"  ⚠️  text attribute exists but cannot extract content: {type(text_value)}")
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Error accessing text attribute: {str(e)}")
            
            # Method 2: Try to access output attribute (Ark API's another possible attribute)
            if not content and response and hasattr(response, 'output'):
                output_value = response.output
                if output_value:
                    if isinstance(output_value, str):
                        content = output_value
                    elif hasattr(output_value, 'content'):
                        content = output_value.content
                    elif hasattr(output_value, 'text'):
                        content = output_value.text
                    elif isinstance(output_value, list) and len(output_value) > 0:
                        # output may be a list
                        first_item = output_value[0]
                        if isinstance(first_item, str):
                            content = first_item
                        elif hasattr(first_item, 'content'):
                            content = first_item.content
                        elif hasattr(first_item, 'text'):
                            content = first_item.text
            
            # Method 3: Try standard choices format (OpenAI compatible format)
            if not content and response and hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    content = response.choices[0].message.content
                elif hasattr(response.choices[0], 'content'):
                    content = response.choices[0].content
                elif hasattr(response.choices[0], 'text'):
                    content = response.choices[0].text
            
            # Method 4: Try to directly access content attribute
            if not content and response and hasattr(response, 'content'):
                content_value = response.content
                if isinstance(content_value, str):
                    content = content_value
                elif hasattr(content_value, 'content'):
                    content = content_value.content
            
            # Extract response_id
            if hasattr(response, 'id'):
                response_id = response.id
            
            # Method 5: Try using model_dump or to_dict method (Pydantic model)
            if not content and response:
                try:
                    if hasattr(response, 'model_dump'):
                        dumped = response.model_dump()
                        if verbose:
                            print(f"  🔍 Using model_dump to extract response, keys: {list(dumped.keys()) if isinstance(dumped, dict) else 'not a dict'}")
                        # Try to extract content from dump result
                        if isinstance(dumped, dict):
                            # Try multiple possible keys
                            for key in ['text', 'output', 'content', 'message', 'choices']:
                                if key in dumped and dumped[key]:
                                    value = dumped[key]
                                    if isinstance(value, str):
                                        content = value
                                        break
                                    elif isinstance(value, list) and len(value) > 0:
                                        first_item = value[0]
                                        if isinstance(first_item, str):
                                            content = first_item
                                            break
                                        elif isinstance(first_item, dict) and 'content' in first_item:
                                            content = first_item['content']
                                            break
                                    elif isinstance(value, dict) and 'content' in value:
                                        content = value['content']
                                        break
                    elif hasattr(response, 'to_dict'):
                        dumped = response.to_dict()
                        if isinstance(dumped, dict) and 'text' in dumped:
                            content = dumped['text'] if isinstance(dumped['text'], str) else str(dumped['text'])
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Error extracting using model_dump: {str(e)}")
            
            if content:
                cache_info['response_length'] = len(str(content))
                if usage_info and verbose:
                    if isinstance(usage_info, dict):
                        print(f"  Token usage: {usage_info}")
                    else:
                        print(f"  Token usage: {usage_info}")
                return True, str(content).strip(), response_id, cache_info
            else:
                # If cannot extract content, try more detailed debugging
                if verbose:
                    print(f"  🔍 Response object details:")
                    print(f"    Type: {type(response)}")
                    if hasattr(response, 'text'):
                        print(f"    text attribute: {type(response.text)}, value: {str(response.text)[:200] if response.text else 'None'}")
                    if hasattr(response, 'output'):
                        print(f"    output attribute: {type(response.output)}, value: {str(response.output)[:200] if response.output else 'None'}")
                    if hasattr(response, 'model_dump'):
                        try:
                            dumped = response.model_dump()
                            print(f"    model_dump keys: {list(dumped.keys()) if isinstance(dumped, dict) else 'not a dict'}")
                            if isinstance(dumped, dict):
                                for key in ['text', 'output', 'content']:
                                    if key in dumped:
                                        print(f"    {key}: {type(dumped[key])}, value: {str(dumped[key])[:200]}")
                        except:
                            pass
                
                # If cannot extract content, return error message with response object info for debugging
                error_msg = f"Cannot extract response content. Response type: {type(response)}"
                cache_info['error'] = error_msg
                return False, error_msg, response_id, cache_info
                
        except Exception as e:
            error_str = str(e)
            cache_info['error'] = error_str
            
            # Check if it's a cache service related error (such as cache service not activated)
            is_cache_error = (
                'AccessDenied.CacheService' in error_str or
                'cache service' in error_str.lower() or
                'cache' in error_str.lower() and '403' in error_str
            )
            
            if is_cache_error:
                # Cache service unavailable, falling back to regular call
                if verbose:
                    print(f"  ⚠️  Cache service unavailable ({error_str[:100]}...), falling back to regular call")
                cache_info['fallback_to_normal'] = True
                try:
                    success, response = self.generate(system_message, prompt)
                    return success, response, None, cache_info
                except Exception as e2:
                    if verbose:
                        print(f"  ✗ Regular call also failed: {str(e2)}")
                    return False, f"API call error (both cache and regular calls failed): {str(e2)}", None, cache_info
            else:
                # Other error, return directly
                if verbose:
                    print(f"  ✗ API call exception: {error_str[:200]}")
                return False, f"API call error: {error_str}", None, cache_info
    
    def delete_cache(self, response_id: str) -> bool:
        """
        Delete Session Cache
        
        Args:
            response_id: Response ID to delete
            
        Returns:
            Whether successfully deleted
        """
        if self.ark_client is None:
            return False
        
        try:
            self.ark_client.responses.delete(response_id)
            return True
        except Exception as e:
            print(f"Warning: Failed to delete cache {response_id}: {str(e)}")
            return False
    
    def num_tokens_from_messages(self, messages: list) -> int:
        """
        Estimate token count (simple estimation)
        
        Args:
            messages: Message list, format is [{"role": "system/user", "content": "..."}]
            
        Returns:
            Estimated token count
        """
        total_text = " ".join([msg.get("content", "") for msg in messages])
        # Simple estimation: average 3 characters = 1 token (conservative estimate)
        return len(total_text) // 3


# ==================== ID Generation and Management ====================

def ensure_conversation_id(data: Dict[str, Any], line_index: Optional[int] = None) -> str:
    """
    Ensure conversation has a unique ID
    
    Priority:
    1. If id field already exists, use it directly
    2. If conversation_id field exists, use it
    3. Otherwise, generate hash ID based on conversation content
    
    Args:
        data: Conversation data
        line_index: Line number (optional, for generating more stable IDs)
    
    Returns:
        Conversation ID string
    """
    # First priority: use existing id field
    if 'id' in data and data['id']:
        return str(data['id'])
    
    # Second priority: use conversation_id field
    if 'conversation_id' in data and data['conversation_id']:
        return str(data['conversation_id'])
    
    # If no ID, generate hash ID based on conversation content
    # Generate stable hash using conversations field
    conversations = data.get('conversations', [])
    if conversations:
        # Convert conversations to string and calculate hash
        conv_str = json.dumps(conversations, sort_keys=True, ensure_ascii=False)
        conv_id = hashlib.md5(conv_str.encode('utf-8')).hexdigest()
        return f"conv_{conv_id[:16]}"  # Use first 16 characters with prefix
    
    # If no conversations field, use line_index or random ID
    if line_index is not None:
        return f"conv_{line_index:06d}"
    
    # Last fallback: use timestamp and random number
    return f"conv_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


def add_id_to_data(data: Dict[str, Any], line_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Add id field to data (if not exists)
    
    Args:
        data: Conversation data
        line_index: Line number (optional)
    
    Returns:
        Data with id field added
    """
    data = data.copy()
    if 'id' not in data or not data.get('id'):
        data['id'] = ensure_conversation_id(data, line_index)
    return data


# ==================== R1: Conversation Length Normalization ====================

def fix_human_message_names(human_value: str, user_name: str, other_person_name: str) -> str:
    """
    Fix names in human messages (Issue 4 fix)
    
    If a name appears in human message that's not defined in system, replace with correct name
    
    Args:
        human_value: Human message content
        user_name: Correct user name
        other_person_name: Correct other person name
    
    Returns:
        Fixed human message
    """
    if not human_value:
        return human_value
    
    result = human_value
    
    # Extract all possible names (words starting with capital letter)
    potential_names = re.findall(r'\b([A-Z][a-z]+)\b', human_value)
    excluded_words = {'i', 'the', 'this', 'that', 'hey', 'oh', 'my', 'god', 
                   'january', 'february', 'march', 'april', 'may', 'june', 
                   'july', 'august', 'september', 'october', 'november', 'december',
                   'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                   'yes', 'no', 'ok', 'okay', 'thanks', 'thank', 'please', 'sorry'}
    
    for name in potential_names:
        if name.lower() in excluded_words:
            continue
        
        # If this name is not a defined name and appears in a direct address position
        if name != user_name and name != other_person_name:
            # Check if it's in a direct address position (like ", Chloe" or "Chloe,")
            direct_address_patterns = [
                rf',\s+{re.escape(name)}(?:\s|$|\.|!)',  # ", Chloe" or ", Chloe."
                rf'{re.escape(name)},\s+',  # "Chloe, "
                rf'You\'re\s+not\s+alone[^,]*,\s+{re.escape(name)}',  # "You're not alone, Chloe"
                rf'Hey\s+{re.escape(name)}(?:\s|$|,|!)',  # "Hey Chloe"
                rf'Hi\s+{re.escape(name)}(?:\s|$|,|!)',  # "Hi Chloe"
                rf'{re.escape(name)}\s+[!?]',  # "Chloe!" or "Chloe?"
            ]
            
            for pattern in direct_address_patterns:
                if re.search(pattern, result, re.IGNORECASE):
                    # Replace with correct user name (assuming user is being addressed)
                    result = re.sub(pattern, lambda m: m.group(0).replace(name, user_name), result, flags=re.IGNORECASE)
                    break
    
    return result


def fix_conversation_order(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fix conversation order, ensure human→assistant alternation pattern for each turn
    
    Rules:
    1. First non-system turn must be human
    2. Then must be human→assistant alternation pattern
    3. If two consecutive human or assistant turns appear, delete extra turn
    4. Ensure conversation ends with assistant turn
    
    Args:
        conversations: Conversation list (ShareGPT format)
    
    Returns:
        Fixed conversation list (ensuring human→assistant alternation)
    """
    if not conversations:
        return conversations
    
    # Separate system messages and actual conversation turns
    system_turns = [c for c in conversations if c.get('from') == 'system']
    actual_turns = [c for c in conversations if c.get('from') != 'system']
    
    if not actual_turns:
        return conversations
    
    # Fix order: ensure human→assistant alternation
    fixed_turns = []
    expected_next = 'human'  # First non-system turn should be human
    
    for turn in actual_turns:
        turn_type = turn.get('from', '')
        
        # Skip unknown types
        if turn_type not in ['human', 'gpt', 'assistant', 'user']:
            continue
        
        # Unified processing: convert assistant and user to ShareGPT format
        if turn_type == 'assistant':
            turn_type = 'gpt'
            # Update turn's from field
            turn = turn.copy()
            turn['from'] = 'gpt'
        elif turn_type == 'user':
            turn_type = 'human'
            # Update turn's from field
            turn = turn.copy()
            turn['from'] = 'human'
        
        # If current turn type matches expectation, add it
        if turn_type == expected_next:
            fixed_turns.append(turn)
            # Switch expected next type
            expected_next = 'gpt' if expected_next == 'human' else 'human'
        # If doesn't match expectation, skip this turn (delete extra turn)
        # Example: two consecutive human turns, second will be skipped
        # Example: two consecutive assistant turns, second will be skipped
    
    # Ensure conversation ends with gpt turn
    # If last turn is human turn, remove it
    if fixed_turns and fixed_turns[-1].get('from') == 'human':
        fixed_turns = fixed_turns[:-1]
    
    # If first turn is not human, might need to add placeholder human turn
    # But this situation shouldn't normally happen, so return directly
    if fixed_turns and fixed_turns[0].get('from') != 'human':
        # If first is assistant, might need to delete it (no corresponding human)
        # Or keep it (if data is naturally like this)
        # Here choose to keep, but log warning
        pass
    
    return system_turns + fixed_turns


def normalize_conversation_length(conversations: List[Dict[str, Any]], 
                                  min_turns: int = 3, 
                                  max_turns: int = 12,
                                  preferred_range: Tuple[int, int] = (6, 8)) -> List[Dict[str, Any]]:
    """
    Standardize conversation length to 3-12 turns
    Ensure conversation ends with assistant turn
    Ensure each turn is human→assistant alternation pattern
    
    Args:
        conversations: Conversation list (ShareGPT format)
        min_turns: Minimum turns (default 3)
        max_turns: Maximum turns (default 12)
        preferred_range: Preferred range (default 6-8)
    
    Returns:
        Normalized conversation list (ensuring ends with assistant turn, human→assistant alternation)
    """
    # First fix conversation order (ensure human→assistant alternation)
    conversations = fix_conversation_order(conversations)
    
    # Calculate actual conversation turns (excluding system messages)
    actual_turns = [c for c in conversations if c.get('from') != 'system']
    num_turns = len(actual_turns)
    
    # If more than 12 turns, randomly sample target length
    if num_turns > max_turns:
        if preferred_range:
            target_length = random.randint(preferred_range[0], preferred_range[1])
        else:
            target_length = random.randint(min_turns, max_turns)
        
        # Ensure target_length is even (human+assistant is one round)
        # If target_length is odd, subtract 1 to make it even
        if target_length % 2 == 1:
            target_length -= 1
            if target_length < min_turns:
                target_length = min_turns if min_turns % 2 == 0 else min_turns + 1
        
        # Keep system messages and first target_length turns of conversation
        system_msg = [c for c in conversations if c.get('from') == 'system']
        trimmed_turns = actual_turns[:target_length]
        conversations = system_msg + trimmed_turns
        actual_turns = trimmed_turns
    
    # Ensure conversation ends with gpt turn（Critical issue 4 fix）
    # If last turn is human turn, remove it
    if actual_turns and actual_turns[-1].get('from') == 'human':
        # Find position of last gpt/assistant turn (unified conversion to gpt)
        last_gpt_index = -1
        for i in range(len(actual_turns) - 1, -1, -1):
            turn_from = actual_turns[i].get('from')
            if turn_from in ['gpt', 'assistant']:
                last_gpt_index = i
                # Unified format: convert assistant to gpt
                if turn_from == 'assistant':
                    actual_turns[i] = actual_turns[i].copy()
                    actual_turns[i]['from'] = 'gpt'
                break
        
        if last_gpt_index >= 0:
            # Keep up to last gpt turn
            system_msg = [c for c in conversations if c.get('from') == 'system']
            trimmed_turns = actual_turns[:last_gpt_index + 1]
            return system_msg + trimmed_turns
        else:
            # If no gpt turn, return empty (this situation shouldn't happen)
            return [c for c in conversations if c.get('from') == 'system']
    
    # Unified format: convert all assistant to gpt, user to human
    for turn in actual_turns:
        if turn.get('from') == 'assistant':
            turn['from'] = 'gpt'
        elif turn.get('from') == 'user':
            turn['from'] = 'human'
    
    return conversations
    
    return conversations


# ==================== System Prompt Processing ====================

def fix_pronouns_in_text(text: str, user_name: str) -> str:
    """
    Fix pronoun issues in text
    
    Rules:
    - First person (I, me, my, myself) remains unchanged
    - Third person (user name, he/she, his/her, him/her) converts to second person (you, your, yourself)
    - Note case and verb form: He is -> You are, She works -> You work
    
    Args:
        text: Text to fix
        user_name: User name (for identifying third person references)
    
    Returns:
        Fixed text
    """
    if not text or not user_name:
        return text
    
    result = text
    
    # Step 1: First replace possessive (must be before replacing nominative to avoid conflicts)
    # Her/His + noun -> Your (possessive)
    result = re.sub(r'\bHer\s+(\w+)', r'Your \1', result)  # Her students -> Your students
    result = re.sub(r'\bher\s+(\w+)', r'your \1', result)  # her students -> your students
    result = re.sub(r'\bHis\s+(\w+)', r'Your \1', result)  # His work -> Your work
    result = re.sub(r'\bhis\s+(\w+)', r'your \1', result)  # his work -> your work
    
    # Step 2: Replace user name and adjust subsequent verbs
    # Handle "Ben is" -> "You are"
    result = re.sub(rf'\b{re.escape(user_name)}\s+is\b', 'You are', result, flags=re.IGNORECASE)
    result = re.sub(rf'\b{re.escape(user_name)}\s+was\b', 'You were', result, flags=re.IGNORECASE)
    result = re.sub(rf'\b{re.escape(user_name)}\s+has\b', 'You have', result, flags=re.IGNORECASE)
    result = re.sub(rf'\b{re.escape(user_name)}\s+does\b', 'You do', result, flags=re.IGNORECASE)
    
    # Handle third person singular verbs (remove s)
    # "Ben works" -> "You work"
    result = re.sub(rf'\b{re.escape(user_name)}\s+(\w+)s\b', r'You \1', result, flags=re.IGNORECASE)
    
    # Handle other common verbs
    result = re.sub(rf'\b{re.escape(user_name)}\s+(works?|goes?|comes?|says?|thinks?|feels?|knows?|wants?|needs?|likes?|loves?|hates?|makes?|takes?|gives?|gets?|sees?|hears?|tells?)\b', 
                   r'You \1', result, flags=re.IGNORECASE)
    
    # Replace remaining user names (cases without verbs)
    # If name is at sentence start, replace with "You"
    result = re.sub(rf'(^|\.\s+){re.escape(user_name)}\b', r'\1You', result, flags=re.MULTILINE | re.IGNORECASE)
    # User names in other positions
    result = re.sub(rf'\b{re.escape(user_name)}\b', 'you', result, flags=re.IGNORECASE)
    
    # Step 3: Replace third person nominative pronouns and adjust verbs
    # He/She is -> You are
    result = re.sub(r'\b(He|She)\s+is\b', 'You are', result, flags=re.IGNORECASE)
    # He/She was -> You were
    result = re.sub(r'\b(He|She)\s+was\b', 'You were', result, flags=re.IGNORECASE)
    # He/She has -> You have
    result = re.sub(r'\b(He|She)\s+has\b', 'You have', result, flags=re.IGNORECASE)
    # He/She does -> You do
    result = re.sub(r'\b(He|She)\s+does\b', 'You do', result, flags=re.IGNORECASE)
    
    # Handle third person singular verbs (remove s)
    result = re.sub(r'\b(He|She)\s+(\w+)s\b', r'You \2', result, flags=re.IGNORECASE)
    
    # Handle other common verbs
    result = re.sub(r'\b(He|She)\s+(works?|goes?|comes?|says?|thinks?|feels?|knows?|wants?|needs?|likes?|loves?|hates?|makes?|takes?|gives?|gets?|sees?|hears?|tells?)\b', 
                   r'You \2', result, flags=re.IGNORECASE)
    
    # Replace remaining third person nominative (cases without verbs)
    result = re.sub(r'\bHe\b(?!\s+\w)', 'You', result)  # Avoid matching Her
    result = re.sub(r'\bShe\b', 'You', result)
    result = re.sub(r'\bhe\b(?!\s+\w)', 'you', result)
    result = re.sub(r'\bshe\b', 'you', result)
    
    # Step 4: Replace accusative
    result = re.sub(r'\bHim\b', 'You', result)
    # Her as accusative (no noun following)
    result = re.sub(r'\bHer\b(?!\s+\w)', 'You', result)
    result = re.sub(r'\bhim\b', 'you', result)
    result = re.sub(r'\bher\b(?!\s+\w)', 'you', result)
    
    # Step 5: Replace reflexive pronouns
    result = re.sub(r'\bHimself\b', 'Yourself', result)
    result = re.sub(r'\bHerself\b', 'Yourself', result)
    result = re.sub(r'\bhimself\b', 'yourself', result)
    result = re.sub(r'\bherself\b', 'yourself', result)
    
    # Step 6: Replace possessive pronouns (Hers -> Yours)
    result = re.sub(r'\bHers\b', 'Yours', result)
    result = re.sub(r'\bhers\b', 'yours', result)
    
    return result


def rebuild_system_prompt_with_archetype(
    original_system_prompt: str,
    user_name: str,
    other_person_name: str,
    occupation: str,
    personality: str,
    talking_style: str,
    archetype_name: str,
    big5_text: Dict[str, str]
) -> str:
    """
    Rebuild system prompt, add archetype and Big Five info
    
    Args:
        original_system_prompt: Original system prompt
        user_name: User name
        other_person_name: Other person name
        occupation: Occupation
        personality: Personality description
        talking_style: Talking style
        archetype_name: Archetype name (e.g., "A2_RiskAversePlanner")
        big5_text: Big FiveText label dictionary
    
    Returns:
        Rebuilt system prompt
    """
    # Extract Scenario and Topic (more flexible matching, remove extra blank lines and newlines)
    scenario_match = re.search(r'Scenario[：:]\s*(.+?)(?=\n\s*Topic[：:]|\n\n|$)', original_system_prompt, re.DOTALL | re.IGNORECASE)
    if not scenario_match:
        # Try another format
        scenario_match = re.search(r'Scenario[：:]\s*(.+?)(?=\n|$)', original_system_prompt, re.MULTILINE | re.IGNORECASE)
    scenario = scenario_match.group(1).strip() if scenario_match else "[abstract scenario]"
    # Clean scenario, remove extra blank lines
    scenario = re.sub(r'\n+', ' ', scenario).strip()
    
    topic_match = re.search(r'Topic[：:]\s*(.+?)(?=\n\s*Character|\n\n|$)', original_system_prompt, re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not topic_match:
        # Try another format
        topic_match = re.search(r'Topic[：:]\s*(.+?)(?=\n|$)', original_system_prompt, re.MULTILINE | re.IGNORECASE)
    topic = topic_match.group(1).strip() if topic_match else "[topic]"
    # Clean topic, remove extra blank lines
    topic = re.sub(r'\n+', ' ', topic).strip()
    
    # Format Big Five info (lowercase, separated by semicolons)
    big5_str = f"openness = {big5_text.get('Openness', 'medium').lower()}; conscientiousness = {big5_text.get('Conscientiousness', 'medium').lower()}; extraversion = {big5_text.get('Extraversion', 'medium').lower()}; agreeableness = {big5_text.get('Agreeableness', 'medium').lower()}; neuroticism = {big5_text.get('Neuroticism', 'medium').lower()}"
    
    # Extract archetype display name (remove prefix, e.g. A2_RiskAversePlanner -> RiskAversePlanner)
    archetype_display = archetype_name.split('_', 1)[-1] if '_' in archetype_name else archetype_name
    
    # Ensure user_name is not "you"
    if not user_name or user_name.lower() in ['you', 'yourself', 'your', 'your name']:
        user_name = "Character"  # Use default value
    
    # Clean personality and talking_style, remove extra blank lines and newlines
    personality_clean = re.sub(r'\n+', ' ', personality).strip() if personality else ""
    talking_style_clean = re.sub(r'\n+', ' ', talking_style).strip() if talking_style else ""
    occupation_clean = re.sub(r'\n+', ' ', occupation).strip() if occupation else ""
    
    # Build new system prompt (exactly according to user requirements, pay attention to format details)
    new_system_prompt = f"""Scenario: {scenario}
Topic: {topic}
Character Information:
- Other person's name: {other_person_name} (use this name when addressing the other person in your responses)
- Your name: {user_name}
- Your occupation: {occupation_clean}
- Your personality: {personality_clean}
- Your archetype: {archetype_display}. Big Five: {big5_str}.
- Your talking style: {talking_style_clean}

Output requirements (order fixed):
1) <Internal Monologue> with <Observation Analysis>, <Self-Reflection>, <State Update Plan>
2) <Current State> as discrete tags only (Stress/Trust/Dominance/Focus)
3) <Response>
End with </assistant_end>"""
    
    return new_system_prompt


def extract_personality_from_system(system_prompt: str) -> str:
    """
    Extract personality text from system prompt
    
    Args:
        system_prompt: System prompt
    
    Returns:
        Personality text
    """
    # Extract Your personality section
    pers_match = re.search(r'Your personality:\s*(.+?)(?=\n-?\s*Your talking style:|\n\n[A-Z][^:]*:|$)', 
                          system_prompt, re.DOTALL | re.IGNORECASE)
    if pers_match:
        return pers_match.group(1).strip()
    return ""


def extract_occupation_from_system(system_prompt: str) -> str:
    """
    Extract occupation text from system prompt
    
    Args:
        system_prompt: System prompt
    
    Returns:
        Occupation text
    """
    occ_match = re.search(r'Your occupation:\s*(.+?)(?=\n-?\s*Your personality:|\n\n[A-Z][^:]*:|$)', 
                         system_prompt, re.DOTALL | re.IGNORECASE)
    if occ_match:
        return occ_match.group(1).strip()
    return ""


def extract_talking_style_from_system(system_prompt: str) -> str:
    """
    Extract talking style text from system prompt
    
    Args:
        system_prompt: System prompt
    
    Returns:
        Talking style text
    """
    style_match = re.search(r'Your talking style:\s*(.+?)(?=\n\n[A-Z][^:]*:|$)', 
                           system_prompt, re.DOTALL | re.IGNORECASE)
    if style_match:
        return style_match.group(1).strip()
    return ""


def infer_archetype_from_personality(personality_text: str, llm, verbose: bool = False, conversations: Optional[List[Dict[str, Any]]] = None) -> Union[Optional[str], Dict[str, Any]]:
    """
    Infer archetype based on personality text, optionally infer character names simultaneously
    
    Use LLM to analyze personality text, match to one of closest 24 archetypes
    If conversation data provided, simultaneously infer "Other person's name"  "Your name"
    
    Args:
        personality_text: Personality text
        llm: LLM instance
        verbose: Whether to output detailed information
        conversations: Optional conversation data list (original conversation data after normalize_conversation_length processing)
    
    Returns:
        If conversations is None, return archetype name (string), return None if failed
        If conversations is not None, return dictionary:{
            'archetype': Archetype name or None,
            'other_person_name': Other person name or None,
            'your_name': Your name or None
        }
    """
    if not personality_text:
        if conversations is None:
            return None
        else:
            return {'archetype': None, 'other_person_name': None, 'your_name': None}
    
    # Build archetype description list (using new ARCHETYPE_DEFINITIONS)
    archetype_descriptions = []
    for archetype_name, archetype_data in ALL_ARCHETYPES.items():
        # Extract archetype name (remove prefix)
        display_name = archetype_name.split('_', 1)[-1] if '_' in archetype_name else archetype_name
        label = archetype_data.get('label', display_name)
        big5_text = f"Openness={archetype_data.get('Openness', 'medium')}, Conscientiousness={archetype_data.get('Conscientiousness', 'medium')}, Extraversion={archetype_data.get('Extraversion', 'medium')}, Agreeableness={archetype_data.get('Agreeableness', 'medium')}, Neuroticism={archetype_data.get('Neuroticism', 'medium')}"
        desc = f"- {archetype_name} ({label}): {big5_text}"
        archetype_descriptions.append(desc)
    
    archetype_list = "\n".join(archetype_descriptions)
    
    # Build conversation history text (if conversation data provided)
    conversation_text = ""
    if conversations:
        conversation_lines = []
        for conv in conversations:
            role = conv.get('from', '')
            value = conv.get('value', '')
            if role == 'system':
                continue  # Skip system messages
            elif role == 'human':
                conversation_lines.append(f"Human: {value}")
            elif role in ['gpt', 'assistant']:
                # Unified format: convert assistant to gpt
                if role == 'assistant':
                    role = 'gpt'
                # Only extract Response section, if no Response tag extract content before </assistant_end>
                response_match = re.search(r'<Response>(.*?)</Response>', value, re.DOTALL)
                if response_match:
                    response_text = response_match.group(1).strip()
                else:
                    # Try to extract content before </assistant_end>, remove known tags
                    end_match = re.search(r'</assistant_end>', value)
                    if end_match:
                        before_end = value[:end_match.start()].strip()
                        before_end = re.sub(r'<Internal Monologue>.*?</Internal Monologue>', '', before_end, flags=re.DOTALL | re.IGNORECASE)
                        before_end = re.sub(r'<Current State>.*?</Current State>', '', before_end, flags=re.DOTALL | re.IGNORECASE)
                        before_end = re.sub(r'<Response>.*?</Response>', '', before_end, flags=re.DOTALL | re.IGNORECASE)
                        response_text = before_end.strip()
                    else:
                        response_text = value.strip()
                conversation_lines.append(f"Assistant: {response_text}")
        conversation_text = "\n\n".join(conversation_lines)
    
    # Build prompt
    if conversations:
        prompt = f"""You are an expert at analyzing personality descriptions and matching them to Big-5 personality archetypes, and also identifying character names from conversations.

# Personality Description:
{personality_text}

# Conversation History:
{conversation_text}

# Available Archetypes (24 total):
{archetype_list}

# Your Tasks:
1. Based on the personality description above, identify which archetype best matches this personality.
2. Based on the conversation history, identify the character names:
   - "Other person's name": The name of the person the assistant is talking to (the human in the conversation)
   - "Your name": The name of the assistant character (the one speaking in assistant turns)

Consider for archetype:
- Openness: curiosity, creativity, willingness to try new things
- Conscientiousness: organization, dependability, self-discipline
- Extraversion: sociability, assertiveness, emotional expressiveness
- Agreeableness: trust, altruism, kindness, affection
- Neuroticism: emotional instability, anxiety, moodiness

Consider for names:
- Look for names mentioned in the conversation (capitalized words that appear multiple times)
- Look for how characters address each other
- If a name cannot be determined from the conversation, use "Unknown" for that field

# Output Format:
Output in the following format (one per line):
Archetype: [archetype name, e.g., "A1_Analyst" or "C1_Empath"]
Other person's name: [name or "Unknown"]
Your name: [name or "Unknown"]

If unsure about archetype, choose the closest match."""
    else:
        prompt = f"""You are an expert at analyzing personality descriptions and matching them to Big-5 personality archetypes.

# Personality Description:
{personality_text}

# Available Archetypes (24 total):
{archetype_list}

# Your Task:
Based on the personality description above, identify which archetype best matches this personality.

Consider:
- Openness: curiosity, creativity, willingness to try new things
- Conscientiousness: organization, dependability, self-discipline
- Extraversion: sociability, assertiveness, emotional expressiveness
- Agreeableness: trust, altruism, kindness, affection
- Neuroticism: emotional instability, anxiety, moodiness

# Output Format:
Output ONLY the archetype name (e.g., "A1_Analyst" or "C1_Empath"), nothing else. If unsure, choose the closest match.

Archetype:"""
    
    system_message = "You are an expert personality analyst."
    
    try:
        success, response = llm.generate(system_message, prompt)
        if success and response:
            response = response.strip()
            
            if conversations:
                # Parse response containing names
                result = {'archetype': None, 'other_person_name': None, 'your_name': None}
                
                # Extract archetype
                archetype_match = re.search(r'Archetype:\s*([^\n]+)', response, re.IGNORECASE)
                if archetype_match:
                    archetype_str = archetype_match.group(1).strip().strip('"\'')
                    # Verify if it's a valid archetype name
                    if archetype_str in ALL_ARCHETYPES:
                        result['archetype'] = archetype_str
                    else:
                        # Try to match (ignore case and prefix)
                        for archetype_name in ALL_ARCHETYPES.keys():
                            if archetype_str.lower() == archetype_name.lower() or archetype_str.lower() in archetype_name.lower():
                                result['archetype'] = archetype_name
                                break
                            display_name = archetype_name.split('_', 1)[-1] if '_' in archetype_name else archetype_name
                            if archetype_str.lower() == display_name.lower():
                                result['archetype'] = archetype_name
                                break
                
                # Other person's name
                other_name_match = re.search(r"Other person'?s name:\s*([^\n]+)", response, re.IGNORECASE)
                if other_name_match:
                    other_name = other_name_match.group(1).strip().strip('"\'')
                    if other_name.lower() != 'unknown':
                        result['other_person_name'] = other_name
                
                # Your name
                your_name_match = re.search(r'Your name:\s*([^\n]+)', response, re.IGNORECASE)
                if your_name_match:
                    your_name = your_name_match.group(1).strip().strip('"\'')
                    if your_name.lower() != 'unknown':
                        result['your_name'] = your_name
                
                if verbose:
                    print(f"Inferred from LLM: archetype={result['archetype']}, other_person_name={result['other_person_name']}, your_name={result['your_name']}")
                
                return result
            else:
                # Old format: only return archetype string
                response = response.strip('"\'')
                
                # Verify if it's a valid archetype name
                if response in ALL_ARCHETYPES:
                    return response
                
                # Try to match (ignore case and prefix)
                for archetype_name in ALL_ARCHETYPES.keys():
                    if response.lower() == archetype_name.lower() or response.lower() in archetype_name.lower():
                        return archetype_name
                    # Check if it's display name
                    display_name = archetype_name.split('_', 1)[-1] if '_' in archetype_name else archetype_name
                    if response.lower() == display_name.lower():
                        return archetype_name
        
        if verbose:
            print(f"Warning: Failed to infer archetype from personality, falling back to Big-5 matching")
    except Exception as e:
        if verbose:
            print(f"Error inferring archetype: {str(e)}")
    
    if conversations:
        return {'archetype': None, 'other_person_name': None, 'your_name': None}
    return None


def fix_system_prompt_pronouns(system_prompt: str, user_name: str) -> str:
    """
    Fix pronoun issues in system prompt (Issue 5 fix)
    
    Fix pronouns throughout system prompt, all changed to second person (You/Your)
    Specifically fix pronouns in Your occupation and Your talking style sections
    Fix "Your name: you" issue, ensure using specific name
    
    Args:
        system_prompt: Original system prompt
        user_name: User name
    
    Returns:
        system prompt（）
    """
    if not system_prompt or not user_name:
        return system_prompt
    
    result = system_prompt
    
    # Step 0: Fix "Your name: you" issue (Issue 3 fix)
    # system prompt "Your name: you" value，
    name_patterns = [
        (r'(Your name:\s*)(you|yourself|your|your name)(\s|$|\n)', rf'\1{user_name}\3'),
        (r'(Character name:\s*)(you|yourself|your|your name)(\s|$|\n)', rf'\1{user_name}\3'),
    ]
    for pattern, replacement in name_patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # Step 1: Fix third person pronouns throughout system prompt (He/She/His/Her etc)
    # But keep first person (I/me/my) unchanged
    result = fix_pronouns_in_text(result, user_name)
    
    # 2: Handle Your occupation 
    occ_match = re.search(r'(Your occupation:\s*)(.+?)(?=\n-?\s*Your personality:|\n\n[A-Z][^:]*:|$)', 
                         result, re.DOTALL | re.IGNORECASE)
    if occ_match:
        prefix = occ_match.group(1)
        occupation_text = occ_match.group(2)
        fixed_occupation = fix_pronouns_in_text(occupation_text, user_name)
        result = result[:occ_match.start()] + prefix + fixed_occupation + result[occ_match.end():]
    
    # 3: Handle Your talking style 
    style_match = re.search(r'(Your talking style:\s*)(.+?)(?=\n\n[A-Z][^:]*:|$)', 
                           result, re.DOTALL | re.IGNORECASE)
    if style_match:
        prefix = style_match.group(1)
        style_text = style_match.group(2)
        fixed_style = fix_pronouns_in_text(style_text, user_name)
        result = result[:style_match.start()] + prefix + fixed_style + result[style_match.end():]
    
    # Step 4: Ensure all descriptive text uses second person
    #  "He will use..." -> "You will use..."
    #  "She speaks..." -> "You speak..."
    # fix_pronouns_in_textHandle
    
    return result


# ==================== R3: Big-5 Personality Trait Label Mapping ====================

def extract_big5_from_system(system_prompt: str, first_gpt_turn_value: Optional[str] = None) -> Dict[str, float]:
    big5 = {
        'openness': 0.5,
        'conscientiousness': 0.5,
        'extraversion': 0.5,
        'agreeableness': 0.5,
        'neuroticism': 0.5
    }
    
    # Try to extract Big-5 value from first GPT turn's Current State
    if first_gpt_turn_value:
        state_dict = extract_current_state_from_gpt_value(first_gpt_turn_value)
        if state_dict:
            emotion = state_dict.get('emotion', {})
            for key in big5.keys():
                if key in emotion:
                    big5[key] = float(emotion[key])
    
    # If cannot extract from gpt turn, try to extract from system prompt text
    # Here we can add logic to parse Big-5 value from system prompt text    
    # Example: Find patterns like "openness: 0.7" or "openness: high"
    
    return big5


# ==================== Big-5 Archetype Definitions (20 types, from information.md) ====================

# 20 Archetype definitions (including Big Five Text label and initial state)
ARCHETYPE_DEFINITIONS = {
    "A_Stable_Rational": {
        "A1_Analyst": {
            "label": "Rational Analyst",
            "Openness": "medium",
            "Conscientiousness": "high",
            "Extraversion": "low",
            "Agreeableness": "medium",
            "Neuroticism": "low",
            "initial_state": {"stress": 0.2, "trust": 0.47, "dominance": 0.17, "focus": 0.62}
        },
        "A2_RiskAversePlanner": {
            "label": "Risk Averse Planner",
            "Openness": "low",
            "Conscientiousness": "high",
            "Extraversion": "low",
            "Agreeableness": "low",
            "Neuroticism": "high",
            "initial_state": {"stress": 0.56, "trust": 0.15, "dominance": 0.2, "focus": 0.45}
        },
        "A3_CalmManager": {
            "label": "Calm Manager",
            "Openness": "medium",
            "Conscientiousness": "high",
            "Extraversion": "medium",
            "Agreeableness": "medium",
            "Neuroticism": "low",
            "initial_state": {"stress": 0.2, "trust": 0.47, "dominance": 0.32, "focus": 0.62}
        },
        "A4_Technocrat": {
            "label": "Technocrat",
            "Openness": "low",
            "Conscientiousness": "high",
            "Extraversion": "low",
            "Agreeableness": "low",
            "Neuroticism": "low",
            "initial_state": {"stress": 0.21, "trust": 0.3, "dominance": 0.25, "focus": 0.6}
        }
    },
    "B_Social_Driving": {
        "B1_Persuader": {
            "label": "Extroverted Persuader",
            "Openness": "high",
            "Conscientiousness": "medium",
            "Extraversion": "high",
            "Agreeableness": "medium",
            "Neuroticism": "low",
            "initial_state": {"stress": 0.2, "trust": 0.5, "dominance": 0.47, "focus": 0.55}
        },
        "B2_Advocate": {
            "label": "Passionate Advocate",
            "Openness": "high",
            "Conscientiousness": "medium",
            "Extraversion": "high",
            "Agreeableness": "high",
            "Neuroticism": "medium",
            "initial_state": {"stress": 0.36, "trust": 0.62, "dominance": 0.4, "focus": 0.5}
        },
        "B3_DominantLeader": {
            "label": "Dominant Leader",
            "Openness": "medium",
            "Conscientiousness": "high",
            "Extraversion": "high",
            "Agreeableness": "low",
            "Neuroticism": "low",
            "initial_state": {"stress": 0.21, "trust": 0.3, "dominance": 0.55, "focus": 0.6}
        },
        "B4_CharismaticRiskTaker": {
            "label": "Charismatic Risk Taker",
            "Openness": "high",
            "Conscientiousness": "low",
            "Extraversion": "high",
            "Agreeableness": "medium",
            "Neuroticism": "low",
            "initial_state": {"stress": 0.26, "trust": 0.5, "dominance": 0.47, "focus": 0.3}
        }
    },
    "C_Emotional_Empathic": {
        "C1_Empath": {
            "label": "Empathetic Listener",
            "Openness": "high",
            "Conscientiousness": "medium",
            "Extraversion": "low",
            "Agreeableness": "high",
            "Neuroticism": "medium",
            "initial_state": {"stress": 0.36, "trust": 0.62, "dominance": 0.17, "focus": 0.5}
        },
        "C2_Mediator": {
            "label": "Emotional Mediator",
            "Openness": "medium",
            "Conscientiousness": "high",
            "Extraversion": "medium",
            "Agreeableness": "high",
            "Neuroticism": "low",
            "initial_state": {"stress": 0.19, "trust": 0.62, "dominance": 0.25, "focus": 0.6}
        },
        "C3_AnxiousSupportSeeker": {
            "label": "Anxious Support Seeker",
            "Openness": "medium",
            "Conscientiousness": "low",
            "Extraversion": "medium",
            "Agreeableness": "high",
            "Neuroticism": "high",
            "initial_state": {"stress": 0.64, "trust": 0.43, "dominance": 0.15, "focus": 0.2}
        },
        "C4_OverAccommodating": {
            "label": "Over Accommodating",
            "Openness": "high",
            "Conscientiousness": "low",
            "Extraversion": "low",
            "Agreeableness": "high",
            "Neuroticism": "high",
            "initial_state": {"stress": 0.63, "trust": 0.45, "dominance": 0.02, "focus": 0.25}
        }
    },
    "D_Defensive_Conflict": {
        "D1_DefensiveSkeptic": {
            "label": "Defensive Skeptic",
            "Openness": "low",
            "Conscientiousness": "medium",
            "Extraversion": "low",
            "Agreeableness": "low",
            "Neuroticism": "high",
            "initial_state": {"stress": 0.56, "trust": 0.15, "dominance": 0.2, "focus": 0.4}
        },
        "D2_PassiveAggressive": {
            "label": "Passive Aggressive",
            "Openness": "medium",
            "Conscientiousness": "medium",
            "Extraversion": "low",
            "Agreeableness": "low",
            "Neuroticism": "high",
            "initial_state": {"stress": 0.55, "trust": 0.18, "dominance": 0.25, "focus": 0.42}
        },
        "D3_Confrontational": {
            "label": "Confrontational",
            "Openness": "low",
            "Conscientiousness": "high",
            "Extraversion": "high",
            "Agreeableness": "low",
            "Neuroticism": "medium",
            "initial_state": {"stress": 0.38, "trust": 0.08, "dominance": 0.5, "focus": 0.38}
        },
        "D4_Detached": {
            "label": "Detached",
            "Openness": "low",
            "Conscientiousness": "low",
            "Extraversion": "low",
            "Agreeableness": "low",
            "Neuroticism": "low",
            "initial_state": {"stress": 0.31, "trust": 0.23, "dominance": 0.2, "focus": 0.3}
        }
    },
    "E_Unstable_Robustness": {
        "E1_Volatile": {
            "label": "Volatile",
            "Openness": "high",
            "Conscientiousness": "low",
            "Extraversion": "high",
            "Agreeableness": "low",
            "Neuroticism": "high",
            "initial_state": {"stress": 0.66, "trust": 0.2, "dominance": 0.5, "focus": 0.2}
        },
        "E2_IdealisticUnstable": {
            "label": "Idealistic Unstable",
            "Openness": "high",
            "Conscientiousness": "low",
            "Extraversion": "medium",
            "Agreeableness": "high",
            "Neuroticism": "high",
            "initial_state": {"stress": 0.64, "trust": 0.5, "dominance": 0.2, "focus": 0.2}
        },
        "E3_Burnout": {
            "label": "Burnout",
            "Openness": "medium",
            "Conscientiousness": "high",
            "Extraversion": "low",
            "Agreeableness": "medium",
            "Neuroticism": "high",
            "initial_state": {"stress": 0.55, "trust": 0.33, "dominance": 0.12, "focus": 0.47}
        },
        "E4_Impulsive": {
            "label": "Impulsive",
            "Openness": "high",
            "Conscientiousness": "low",
            "Extraversion": "high",
            "Agreeableness": "low",
            "Neuroticism": "medium",
            "initial_state": {"stress": 0.49, "trust": 0.28, "dominance": 0.52, "focus": 0.28}
        }
    }
}

# Flattenedtened archetype list (for quick lookup)
ALL_ARCHETYPES = {}
for category, archetypes in ARCHETYPE_DEFINITIONS.items():
    for archetype_name, archetype_data in archetypes.items():
        ALL_ARCHETYPES[archetype_name] = archetype_data


def big5_text_to_value(text: str) -> float:
    """
    Convert Big Five text labels to numerical values
    
    Args:
        text: Text label（"low", "medium", "high"）
    
    Returns:
        Numerical value（0.25, 0.5, 0.75）
    """
    text_lower = text.lower().strip()
    if text_lower == "low":
        return 0.25
    elif text_lower == "medium":
        return 0.5
    elif text_lower == "high":
        return 0.75
    else:
        # Default value
        return 0.5


def big5_value_to_text(value: float) -> str:
    """
    Big FiveNumerical valueText label
    
    Args:
        value: Numerical value（0.0-1.0）
    
    Returns:
        Text label（"low", "medium", "high"）
    """
    if value < 0.33:
        return "low"
    elif value < 0.66:
        return "medium"
    else:
        return "high"


def find_closest_archetype(big5_text: Dict[str, str]) -> Tuple[str, Dict[str, float], Dict[str, str]]:
    """
    Find closest archetype based on Big-5 text labels
    
    Args:
        big5_text: Big-5 trait Text label dictionary {"Openness": "medium", "Conscientiousness": "high", ...}
    
    Returns:
        (archetype_name, initial_state, big5_text_dict) Tuple
    """
    # Text labelNumerical value
    big5_values = {}
    trait_mapping = {
        "Openness": "openness",
        "Conscientiousness": "conscientiousness",
        "Extraversion": "extraversion",
        "Agreeableness": "agreeableness",
        "Neuroticism": "neuroticism"
    }
    
    for key, value in big5_text.items():
        trait_key = trait_mapping.get(key, key.lower())
        big5_values[trait_key] = big5_text_to_value(value)
    
    min_distance = float('inf')
    closest_archetype = None
    closest_state = None
    closest_big5_text = None
    
    for archetype_name, archetype_data in ALL_ARCHETYPES.items():
        # Get archetype's Big Five text labels
        archetype_big5_text = {
            "Openness": archetype_data.get("Openness", "medium"),
            "Conscientiousness": archetype_data.get("Conscientiousness", "medium"),
            "Extraversion": archetype_data.get("Extraversion", "medium"),
            "Agreeableness": archetype_data.get("Agreeableness", "medium"),
            "Neuroticism": archetype_data.get("Neuroticism", "medium")
        }
        
        # Convert to Numerical value
        archetype_big5_values = {}
        for key, value in archetype_big5_text.items():
            trait_key = trait_mapping.get(key, key.lower())
            archetype_big5_values[trait_key] = big5_text_to_value(value)
        
        # Calculate Euclidean distance
        distance = 0.0
        for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            diff = big5_values.get(trait, 0.5) - archetype_big5_values.get(trait, 0.5)
            distance += diff * diff
        
        distance = math.sqrt(distance)
        
        if distance < min_distance:
            min_distance = distance
            closest_archetype = archetype_name
            closest_state = archetype_data['initial_state'].copy()
            closest_big5_text = archetype_big5_text.copy()
    
    return closest_archetype, closest_state, closest_big5_text


def calculate_initial_state_from_big5(big5_text: Dict[str, str]) -> Tuple[Dict[str, float], str, Dict[str, str]]:
    """
    Calculate initial state  value by matching closest archetype (based on 20 archetypes)
    
    Determine initial state by matching closest archetype
    
    Args:
        big5_text: Big-5 trait Text label dictionary {"Openness": "medium", ...}
    
    Returns:
        (initial state dictionary, archetype name, big5_text_dict) Tuple
        Initial state includes: stress, trust, dominance, focus
    """
    archetype_name, initial_state, big5_text_dict = find_closest_archetype(big5_text)
    
    # Ensure all values are in [0, 1] range
    for key in initial_state:
        initial_state[key] = max(0.0, min(1.0, initial_state[key]))
    
    return initial_state, archetype_name, big5_text_dict


def map_big5_to_tags(big5_values: Dict[str, float]) -> str:
    """
    Map Big-5 Numerical value to Text label (using unified threshold value)
    
    Args:
        big5_values: Big-5 trait Numerical value dictionary
    
    Returns:
        Text label string, Example "openness: high | conscientiousness: medium"
    """
    tags = []
    for trait, value in big5_values.items():
        if value < 0.33:
            level = "low"
        elif value < 0.66:
            level = "medium"
        else:
            level = "high"
        tags.append(f"{trait}: {level}")
    
    return " | ".join(tags)


# ==================== R5: State Update Logic ====================

def parse_plan_to_delta(plan_text: str) -> Dict[str, float]:
    """
    Parse State Update Plan text, extract state change delta
    
    Support unified format：Stress: increase a little
    
    Args:
        plan_text: State Update Plan text content
    
    Returns:
        State change dictionary，Example {"stress": 0.05, "trust": -0.05}
    """
    deltas = {}
    
    # Define mapping rules
    delta_map = {
        "increase a lot": 0.10,
        "increase a little": 0.05,
        "stable": 0.00,
        "decrease a little": -0.05,
        "decrease a lot": -0.10
    }
    
    # State dimensions (note case)
    state_dimensions = {
        'stress': 'Stress',
        'trust': 'Trust',
        'dominance': 'Dominance',
        'focus': 'Focus'
    }
    
    plan_lower = plan_text.lower()
    
    for dim_key, dim_capitalized in state_dimensions.items():
        # Method 1: Try to match unified format "Dimension: change_type"
        # Example: "Stress: increase a little"
        pattern1 = rf"{dim_capitalized}:\s*((?:increase|decrease|stable)[^.\n]*)"
        match1 = re.search(pattern1, plan_text, re.IGNORECASE | re.MULTILINE)
        
        if match1:
            change_text = match1.group(1).strip().lower()
            # Find matching delta value
            found = False
            for key, delta in delta_map.items():
                if key in change_text:
                    deltas[dim_key] = delta
                    found = True
                    break
            
            if not found:
                # If no exact match, try simple judgment
                if "increase" in change_text and "lot" in change_text:
                    deltas[dim_key] = 0.10
                elif "increase" in change_text:
                    deltas[dim_key] = 0.05
                elif "decrease" in change_text and "lot" in change_text:
                    deltas[dim_key] = -0.10
                elif "decrease" in change_text:
                    deltas[dim_key] = -0.05
                else:
                    deltas[dim_key] = 0.00
            continue
        
        # Method 2: Fall back to old format matching (compatibility)
        pattern2 = rf"{dim_key}[^.:]*?((?:increase|decrease|stable)[^.\n]*)"
        match2 = re.search(pattern2, plan_lower, re.IGNORECASE)
        
        if match2:
            change_text = match2.group(1).strip()
            for key, delta in delta_map.items():
                if key in change_text:
                    deltas[dim_key] = delta
                    break
            else:
                if "increase" in change_text:
                    deltas[dim_key] = 0.05
                elif "decrease" in change_text:
                    deltas[dim_key] = -0.05
                else:
                    deltas[dim_key] = 0.00
        else:
            # If dimension description not found, default to stable
            deltas[dim_key] = 0.00
    
    return deltas


# ==================== State update pipeline (authoritative implementation from task1.md) ====================

def apply_plan(x: float, action: str, delta: float, Delta: float) -> float:
    """
    Apply plan action to state value (authoritative implementation)
    
    Args:
        x: Current state value
        action: Plan action（"increase a lot", "increase a little", "stable", "decrease a little", "decrease a lot"）
        delta: Small change amount（0.06）
        Delta: Large change amount（0.18）
    
    Returns:
        Candidate value after applying plan
    """
    if action == "increase a lot":
        return x + Delta * (1 - x)
    if action == "increase a little":
        return x + delta * (1 - x)
    if action == "decrease a lot":
        return x - Delta * x
    if action == "decrease a little":
        return x - delta * x
    return x  # stable


def clip01(x: float) -> float:
    """Limit value to [0, 1] range"""
    return max(0.0, min(1.0, x))


def update_state(state: Dict[str, str], plan: Dict[str, str],
                 delta: float = 0.06, Delta: float = 0.18,
                 delta_T: float = 0.03, Delta_T: float = 0.10,   # trust（）
                 eta: float = 0.35, k_sf: float = 0.10) -> Dict[str, float]:
    """
    State update pipeline (authoritative implementation from task1.md)
    
    4-step pipeline：
    1. plan → candidate (with trust-buffer rules)
    2. EMA smoothing (use eta to control update inertia)
    3. Coupling correction (cross-dimensional coupling, such as stress affecting focus)
    4. clip to [0,1]
    
    Args:
        state: Current state dictionary {"stress": 0.7, "trust": 0.3, ...}
        plan: Plan dictionary {"stress": "increase a little", "trust": "stable", ...}
        delta: Small change amount (default 0.06)
        Delta: Large change amount (default 0.18)
        delta_T: Small change amount for trust (default 0.03, slower)
        Delta_T: Large change amount for trust (default 0.10, slower)
        eta: EMA smoothing coefficient (default 0.35, more stable, Issue C fix)
        k_sf: Stress-to-focus coupling coefficient (default 0.10)
    
    Returns:
        Updated state dictionary (float values)
    """
    s, t, d, f = state["stress"], state["trust"], state["dominance"], state["focus"]
    
    # Step 1: plan -> candidate (with trust-buffer for stress increases)
    # Issue E fix: adjust trust-buffer coefficient from (1-0.5*t) to (1-0.3*t), lower the minimum
    ds, Ds = delta, Delta
    if plan["stress"].startswith("increase"):
        ds = delta * (1 - 0.3 * t)  # Fix E: changed from 0.5 to 0.3
        Ds = Delta * (1 - 0.3 * t)  # Fix E: changed from 0.5 to 0.3
    
    s_plan = apply_plan(s, plan["stress"], ds, Ds)
    t_plan = apply_plan(t, plan["trust"],  delta_T, Delta_T)
    d_plan = apply_plan(d, plan["dominance"], delta,  Delta)
    f_plan = apply_plan(f, plan["focus"],     delta,  Delta)
    
    # Issue E fix: add natural decay mechanism for stress (slight decay when stable)
    if plan["stress"] == "stable":
        s_plan = s * 0.98  # Natural decay 2%
    
    # Step 2: EMA smoothing
    # Issue C fix: eta changed from 0.7 to 0.35 (more stable, reduce bucket crossing frequency)
    s2 = (1-eta)*s + eta*s_plan
    t2 = (1-eta)*t + eta*t_plan
    d2 = (1-eta)*d + eta*d_plan
    f2 = (1-eta)*f + eta*f_plan
    
    # Step 3: coupling A (stress eats focus)
    # Issue F fix: changed to proportional penalty, gentler; only deduct when plan doesn't have "increase a lot" focus
    if plan["focus"] != "increase a lot":
        # Use proportional penalty, gentler
        if s2 > 0.6:
            f2 = f2 * (1 - k_sf * (s2 - 0.6))  # Proportional penalty instead of hard deduction
    # If plan is "increase a lot" focus, don't deduct (avoid deducting when model clearly says "more focused")
    
    return {
        "stress": clip01(s2),
        "trust": clip01(t2),
        "dominance": clip01(d2),
        "focus": clip01(f2),
    }


def parse_plan_to_dict(plan_text: str) -> Dict[str, str]:
    """
    State Update Plan，Plan dictionary
    
    Args:
        plan_text: State Update Plan text content
    
    Returns:
        Plan dictionary，Example {"stress": "increase a little", "trust": "stable", ...}
    """
    plan = {}
    
    # State dimensions
    state_dimensions = {
        'stress': 'Stress',
        'trust': 'Trust',
        'dominance': 'Dominance',
        'focus': 'Focus'
    }
    
    # Allowed plan values
    allowed_values = ["increase a lot", "increase a little", "stable", "decrease a little", "decrease a lot"]
    
    for dim_key, dim_capitalized in state_dimensions.items():
        # Match format "Dimension: value"
        pattern = rf"{dim_capitalized}:\s*((?:increase|decrease|stable)[^.\n]*)"
        match = re.search(pattern, plan_text, re.IGNORECASE | re.MULTILINE)
        
        if match:
            value_text = match.group(1).strip().lower()
            # Matchvalue
            found = False
            for allowed_val in allowed_values:
                if allowed_val in value_text:
                    plan[dim_key] = allowed_val
                    found = True
                    break
            
            if not found:
                # If no exact match, try simple judgment
                if "increase" in value_text and "lot" in value_text:
                    plan[dim_key] = "increase a lot"
                elif "increase" in value_text:
                    plan[dim_key] = "increase a little"
                elif "decrease" in value_text and "lot" in value_text:
                    plan[dim_key] = "decrease a lot"
                elif "decrease" in value_text:
                    plan[dim_key] = "decrease a little"
                else:
                    plan[dim_key] = "stable"
        else:
            # If dimension description not found, default to stable
            plan[dim_key] = "stable"
    
    return plan


def calculate_new_state_float(previous_state: Dict[str, float], 
                              plan: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate new state float values using authoritative implementation
    
    Args:
        previous_state: Previous round's state float values
        plan: Plan dictionary {"stress": "increase a little", ...}
    
    Returns:
        (New state dictionary, state_delta dictionary) Tuple
    """
    # Convert float state to string format (for update_state)
    state_str = {k: str(v) for k, v in previous_state.items()}
    
    # Update state using authoritative implementation
    new_state = update_state(previous_state, plan)
    
    # Calculate delta
    state_delta = {
        k: new_state[k] - previous_state[k]
        for k in ['stress', 'trust', 'dominance', 'focus']
    }
    
    return new_state, state_delta


def map_float_to_tag(value: float) -> str:
    """
    Map float value to label (according to task1.md requirements)
    
    Using thresholds：
    - 0 <= x < 0.3 -> low
    - 0.3 <= x < 0.6 -> medium
    - x >= 0.6 -> high
    
    Args:
        value: Float value（0.0-1.0）
    
    Returns:
        Label: "low" | "medium" | "high"
    """
    if value < 0.3:
        return "low"
    elif value < 0.6:
        return "medium"
    else:
        return "high"


# ==================== R2: Monologue Regeneration (First Person) ====================

def build_relabel_prompt(
    system_prompt: str,
    conversation_history: List[Dict[str, Any]],
    current_turn_index: int,
    user_name: str,
    other_person_name: str,
    current_gpt_response: str = "",
    previous_state: Optional[Dict[str, float]] = None
) -> str:
    """
    Build prompt for regenerating first-person monologue
    
    Args:
        system_prompt: System prompt
        conversation_history: Conversation history
        current_turn_index: Current gpt turn index
        user_name: User name (first-person character)
        other_person_name: Other person name
        current_gpt_response: Current turn's gpt response (for reference, but not regenerated)
        previous_state: Previous round's state float values (optional, for providing state context)
    
    Returns:
        Built prompt
    """
    # Find human turn before current gpt turn (this is the message that triggered current gpt response)
    current_message = ""
    for i in range(current_turn_index - 1, -1, -1):
        turn = conversation_history[i]
        if turn.get('from') == 'human':
            current_message = turn.get('value', '')
            break
    
    # Build history context (only includes conversation before current message)
    history_text = ""
    for i, turn in enumerate(conversation_history[:current_turn_index]):
        role = turn.get('from', '')
        value = turn.get('value', '')
        if role == 'human':
            history_text += f"{other_person_name}: {value}\n"
        elif role in ['gpt', 'assistant']:
            # Extract Response section
            response_match = re.search(r'<Response>(.*?)</Response>', value, re.DOTALL)
            if response_match:
                history_text += f"{user_name}: {response_match.group(1).strip()}\n"
    
    # Simplify prompt, keep only key information
    # Limit history context length to avoid overly long prompt
    if len(history_text) > 2000:
        # Only keep recent conversation history
        lines = history_text.split('\n')
        history_text = '\n'.join(lines[-10:])  # Only keep last 10 lines
    
    # Build previous round state information (if provided)
    previous_state_info = ""
    if previous_state:
        # Convert Float value to Label format (using map_float_to_tag function defined in the file)
        # If stress is already high (>=0.6), same stimulus may produce greater reaction
        # If trust is already low (<0.3), more likely to produce distrust
        def _map_float_to_tag(value: float) -> str:
            if value < 0.3:
                return "low"
            elif value < 0.6:
                return "medium"
            else:
                return "high"
        
        state_tags = {
            "Stress": _map_float_to_tag(previous_state.get("stress", 0.5)),
            "Trust": _map_float_to_tag(previous_state.get("trust", 0.5)),
            "Dominance": _map_float_to_tag(previous_state.get("dominance", 0.5)),
            "Focus": _map_float_to_tag(previous_state.get("focus", 0.5))
        }
        stress_val = previous_state.get('stress', 0.5)
        trust_val = previous_state.get('trust', 0.5)
        dominance_val = previous_state.get('dominance', 0.5)
        focus_val = previous_state.get('focus', 0.5)
        
        previous_state_info = f"""
# Your Previous State (before this message):
Stress: {state_tags['Stress']} (value: {stress_val:.2f})
Trust: {state_tags['Trust']} (value: {trust_val:.2f})
Dominance: {state_tags['Dominance']} (value: {dominance_val:.2f})
Focus: {state_tags['Focus']} (value: {focus_val:.2f})

IMPORTANT: Consider your previous state when generating Observation Analysis, Self-Reflection, and State Update Plan. Your current emotional state affects how you interpret and respond to new messages. For example:
- If your Stress was already high ({state_tags['Stress']}), a new stressor might push it even higher, or you might be more sensitive to criticism.
- If your Trust was low ({state_tags['Trust']}), you might interpret ambiguous messages more negatively.
- If your Dominance was low ({state_tags['Dominance']}), you might feel more easily pushed around.
- Your state changes should be relative to where you were before, not absolute. Consider the baseline when deciding the magnitude of change.
"""
    
    # Build prompt, include current turn's response as reference (only for Self Reflection and State Update Plan)
    response_reference = ""
    if current_gpt_response:
        response_reference = f"""
# Your Response (for reference in Self-Reflection and State Update Plan only):
{current_gpt_response}

# Task:
You are {user_name}. Generate your internal monologue. IMPORTANT: <Observation Analysis> should NOT reference your response above. Only <Self-Reflection> and <State Update Plan> should consider your response, but through a two-step reasoning process to maintain first-person perspective.
"""
    else:
        response_reference = f"""
# Task:
You are {user_name}. Generate your internal monologue immediately after receiving the message above.
"""
    
    prompt = f"""# Context:
{system_prompt}

# Recent History:
{history_text}
{previous_state_info}
# Current Message from {other_person_name}:
{current_message}
{response_reference}

# CRITICAL RULES:

1. **FIRST-PERSON PERSPECTIVE ONLY**: Use "I notice...", "I feel...", "I think..." - NEVER third-person or God's-eye view. NEVER analyze your own response from an external perspective.

2. **OBSERVATION ANALYSIS** (2-3 sentences):
   - MUST start: "I notice {other_person_name} [specific quote/reference from Current Message]..."
   - MUST include: What did {other_person_name} actually say? How do I interpret it?
   - CONSIDER YOUR Previous Emotional State: Your previous emotional state (shown in Previous State above) affects how you perceive things. If you're already stressed or distrustful, you might interpret messages differently.
   - FORBIDDEN: References to your own response, "Analyze the situation", "Think about the response", placeholders, empty content
   - FORBIDDEN: God's-eye view or third-person analysis
   - Example: "I notice {other_person_name} mentioned struggling with January bills and seems worried. They're asking for help splitting costs fairly."

3. **SELF-REFLECTION** (use two-step reasoning if response is provided):
   - Step 1: <Reasoning> - Analyze how {other_person_name}'s message would affect your emotions/state, leading to the response you gave. Think about the emotional chain: their words → your internal reaction → your response. Consider how your previous state (shown above) influences your reaction.
   - Step 2: <First Person> - Convert the reasoning into FIRST-PERSON emotional experience. Express what you FEEL, not what you observe about yourself. FORBIDDEN: Any mention of "my response", "I said", "I responded", or analysis of your own words.
   - MUST be SPECIFIC: What emotion/thought does this trigger in me? How does this change my state? How does my current state (from Previous State) affect my reaction?
   - MUST connect to what {other_person_name} just said
   - FORBIDDEN: Generic templates, placeholders, third-person analysis of your response
   - Example structure:
     <Reasoning>
     [Analyze: Their words about X would make me feel Y, which explains why I responded with Z]
     </Reasoning>
     <First Person>
     [I feel Y because of X. This makes me want to...]
     </First Person>
   - Final output: Only the content from <First Person> (without the tags)

4. **STATE UPDATE PLAN** (use two-step reasoning if response is provided):
   - CRITICAL: Your plan must be NATURAL and RESPONSIVE - like how a real person's internal state would actually change in response to what happened. Think about what you would genuinely feel, not what a template says.
   - AVOID COLLAPSE: Do NOT default to "increase a little" or "stable" for everything. Real emotions fluctuate! If something significant happened, your state should reflect that intensity.
   - CONSIDER YOUR BASELINE: Your Previous State (shown above) is your starting point. State changes are RELATIVE to where you were:
     * If Stress was already high, a new stressor might push it "increase a lot" even if the stressor itself is moderate.
     * If Trust was already low, even a small negative signal might cause "decrease a little" or "decrease a lot".
     * If you're already at an extreme (very high or very low), changes might be smaller (you're near the limit) or larger (you're more sensitive).
   - Step 1: <Reasoning> - Think deeply: What did {other_person_name} actually do or say? How would that make you feel internally, given your current state? What state changes would naturally follow from that emotional experience? Consider both the intensity of the interaction AND your baseline state.
   - Step 2: <First Person> - Express your state changes as if you're planning them from within your own experience. Use natural, first-person language about how your internal state is shifting relative to where you were before.
   - FORBIDDEN: Any mention of "my response", "I said", "I responded", or analysis of your own words
   - ONLY four dimensions: Stress, Trust, Dominance, Focus
   - ONLY five values: "increase a lot", "increase a little", "stable", "decrease a little", "decrease a lot"
   - FORBIDDEN: Big-5 traits, other dimensions, placeholders, descriptions
   - NATURAL STATE CHANGES - Think about realistic emotional responses based on your personality and archetype:
     * **Stress**: Rises when you feel threatened, criticized, overwhelmed, or uncertain. Falls when you feel safe, understood, or relieved.
       - Example: If they blame you harshly → Stress likely "increase a lot"
       - Example: If they apologize sincerely → Stress likely "decrease a little" or "decrease a lot"
     * **Trust**: Rises when you feel respected, heard, or supported. Falls when you feel betrayed, dismissed, or manipulated.
       - Example: If they acknowledge your concerns → Trust likely "increase a little"
       - Example: If they dismiss your feelings → Trust likely "decrease a little" or "decrease a lot"
     * **Dominance**: Rises when you assert yourself, set boundaries, or take control. Falls when you feel pushed around, interrupted, or powerless.
       - Example: If you propose a solution → Dominance likely "increase a little"
       - Example: If they interrupt you repeatedly → Dominance likely "decrease a little" or "decrease a lot"
     * **Focus**: Rises when the conversation becomes structured, task-oriented, or requires problem-solving. Falls when you're distracted, emotional, or overwhelmed.
       - Example: If you start listing options → Focus likely "increase a little" or "increase a lot"
       - Example: If you're very stressed → Focus might decrease (stress can reduce focus)
   - IMPORTANT: State changes should be PROPORTIONAL to what happened:
     * Minor things (polite disagreement, small clarification) → mostly "stable" or "increase/decrease a little"
     * Major things (harsh criticism, sincere apology, major conflict) → likely "increase/decrease a lot"
     * Don't overreact to small things, but don't underreact to big things either!
   - Format (no indentation, one line per dimension):
     <Reasoning>
     [Think: What happened? How would that make me feel? What state changes feel natural?]
     </Reasoning>
     <First Person>
Stress: [increase a lot | increase a little | stable | decrease a little | decrease a lot]
Trust: [increase a lot | increase a little | stable | decrease a little | decrease a lot]
Dominance: [increase a lot | increase a little | stable | decrease a little | decrease a lot]
Focus: [increase a lot | increase a little | stable | decrease a little | decrease a lot]
     </First Person>
   - Final output: Only the content from <First Person> (without the tags and without the <Reasoning> section)

5. **LANGUAGE**: ALL content MUST be in ENGLISH only.

# Output Format:
<Internal Monologue>
  <Observation Analysis>
    [Your observation here - 2-3 sentences, specific to Current Message, NO reference to your response]
  </Observation Analysis>
  <Self-Reflection>
    <Reasoning>
    [If response provided: Analyze how their message affects your emotions, leading to your response]
    </Reasoning>
    <First Person>
    [Your first-person emotional experience - what you FEEL, not what you observe about yourself]
    </First Person>
  </Self-Reflection>
  <State Update Plan>
    <Reasoning>
    [If response provided: Analyze how their message affects your state dimensions]
    </Reasoning>
    <First Person>
Stress: [increase a lot | increase a little | stable | decrease a little | decrease a lot]
Trust: [increase a lot | increase a little | stable | decrease a little | decrease a lot]
Dominance: [increase a lot | increase a little | stable | decrease a little | decrease a lot]
Focus: [increase a lot | increase a little | stable | decrease a little | decrease a lot]
    </First Person>
  </State Update Plan>
</Internal Monologue>

# IMPORTANT: 
- Do NOT generate <Current State> or <Response>
- <Observation Analysis> should NOT reference your response
- Include <Reasoning> and <First Person> in <Self-Reflection> and <State Update Plan> if response is provided
- The final output will extract only the <First Person> content (without tags) for these sections

Output ONLY the <Internal Monologue> XML structure:"""
    
    return prompt


def normalize_state_update_plan(plan_text: str) -> str:
    """
    Normalize State Update Plan format (Issue 3 fix: prohibit Big5 words)
    
    Ensure format is fixed to 4 lines, no indentation, no list symbols, only 5 allowed enum values
    Automatically remove Big5-related vocabulary
    
    Args:
        plan_text: Original plan text
    
    Returns:
        Normalized plan text
    """
    if not plan_text:
        return ""
    
    # Check if contains Big5 words (Issue 3 fix)
    big5_keywords = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    has_big5 = any(keyword in plan_text.lower() for keyword in big5_keywords)
    
    if has_big5:
        # If contains Big5 words, only extract four dimensions, ignore Big5 lines
        # Remove lines containing Big5 words
        lines = plan_text.split('\n')
        filtered_lines = []
        for line in lines:
            line_lower = line.lower()
            if not any(keyword in line_lower for keyword in big5_keywords):
                filtered_lines.append(line)
        plan_text = '\n'.join(filtered_lines)
    
    # Extract values for 4 dimensions
    dimensions = ['Stress', 'Trust', 'Dominance', 'Focus']
    allowed_changes = ['increase a lot', 'increase a little', 'stable', 'decrease a little', 'decrease a lot']
    
    extracted = {}
    for dim in dimensions:
        # Try to match "Dimension: change_type"
        pattern = rf'{dim}:\s*((?:increase|decrease|stable)[^.\n]*)'
        match = re.search(pattern, plan_text, re.IGNORECASE | re.MULTILINE)
        if match:
            change_text = match.group(1).strip().lower()
            # Find matching change type
            for allowed in allowed_changes:
                if allowed in change_text:
                    extracted[dim] = allowed
                    break
            if dim not in extracted:
                # If no exact match, try to infer
                if 'increase' in change_text and 'lot' in change_text:
                    extracted[dim] = 'increase a lot'
                elif 'increase' in change_text:
                    extracted[dim] = 'increase a little'
                elif 'decrease' in change_text and 'lot' in change_text:
                    extracted[dim] = 'decrease a lot'
                elif 'decrease' in change_text:
                    extracted[dim] = 'decrease a little'
                else:
                    extracted[dim] = 'stable'
        else:
            # If not found, default to stable
            extracted[dim] = 'stable'
    
    # Build fixed format: 4 lines, no indentation
    lines = []
    for dim in dimensions:
        change = extracted.get(dim, 'stable')
        lines.append(f"{dim}: {change}")
    
    return "\n".join(lines)


def parse_internal_monologue(monologue_text: str) -> Dict[str, str]:
    """
    Parse Internal Monologue XML, extract three parts
    Self-ReflectionState Update Plan，<First Person>，<Reasoning>Label
    
    Args:
        monologue_text: Text containing XML
    
    Returns:
        Dictionary containing observation_analysis, self_reflection, state_update_plan
    """
    result = {
        'observation_analysis': '',
        'self_reflection': '',
        'state_update_plan': ''
    }
    
    # Extract Observation Analysis (without Reasoning/First Person structure)
    obs_match = re.search(r'<Observation Analysis>(.*?)</Observation Analysis>', 
                         monologue_text, re.DOTALL | re.IGNORECASE)
    if obs_match:
        obs_content = obs_match.group(1).strip()
        obs_content = re.sub(r'<Reasoning>.*?</Reasoning>', '', obs_content, flags=re.DOTALL | re.IGNORECASE)
        obs_content = re.sub(r'<First Person>.*?</First Person>', '', obs_content, flags=re.DOTALL | re.IGNORECASE)
        obs_content = obs_content.strip()
        # Remove leading literal \n strings (displayed as \\n in JSON)
        obs_content = re.sub(r'^\\n+', '', obs_content)
        result['observation_analysis'] = obs_content
    
    # Extract Self-Reflection, prioritize extracting <First Person> content
    self_match = re.search(r'<Self-Reflection>(.*?)</Self-Reflection>', 
                          monologue_text, re.DOTALL | re.IGNORECASE)
    if self_match:
        self_content = self_match.group(1).strip()
        
        # Try to extract <First Person> content
        first_person_match = re.search(r'<First Person>(.*?)</First Person>', 
                                      self_content, re.DOTALL | re.IGNORECASE)
        if first_person_match:
            # If <First Person> is found, only use this content
            self_reflection_content = first_person_match.group(1).strip()
            # Remove leading literal \n strings
            self_reflection_content = re.sub(r'^\\n+', '', self_reflection_content)
            result['self_reflection'] = self_reflection_content
        else:
            # If no <First Person>Label, remove <Reasoning> content, keep remaining content
            self_content = re.sub(r'<Reasoning>.*?</Reasoning>', '', self_content, flags=re.DOTALL | re.IGNORECASE)
            self_content = self_content.strip()
            # Remove leading literal \n strings
            self_content = re.sub(r'^\\n+', '', self_content)
            result['self_reflection'] = self_content
    
    # Extract State Update Plan, prioritize extracting <First Person> content
    plan_match = re.search(r'<State Update Plan>(.*?)</State Update Plan>', 
                          monologue_text, re.DOTALL | re.IGNORECASE)
    if plan_match:
        plan_content = plan_match.group(1).strip()
        
        # Try to extract <First Person> content
        first_person_match = re.search(r'<First Person>(.*?)</First Person>', 
                                      plan_content, re.DOTALL | re.IGNORECASE)
        if first_person_match:
            # If <First Person> is found, only use this content
            result['state_update_plan'] = first_person_match.group(1).strip()
        else:
            # If no <First Person>Label, remove <Reasoning> content, keep remaining content
            plan_content = re.sub(r'<Reasoning>.*?</Reasoning>', '', plan_content, flags=re.DOTALL | re.IGNORECASE)
            result['state_update_plan'] = plan_content.strip()
    
    return result


# ==================== R4: Dual-Track State Representation ====================

def extract_current_state_from_gpt_value(gpt_value: str) -> Optional[Dict[str, Any]]:
    """
    Extract current state from gpt's value field (JSON format)
    Use more robust parsing method
    
    Args:
        gpt_value: gpt turn's value field
    
    Returns:
        State dictionary containing emotion and relation, return None if parsing fails
    """
    state_match = re.search(r'<Current State>(.*?)</Current State>', 
                           gpt_value, re.DOTALL | re.IGNORECASE)
    if not state_match:
        return None
    
    state_content = state_match.group(1).strip()
    
    # Method 1: Try direct parsing
    try:
        state_dict = json.loads(state_content)
        return state_dict
    except json.JSONDecodeError:
        pass
    
    # Method 2: Find first { to last }, try parsing
    start = state_content.find("{")
    end = state_content.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = state_content[start:end+1]
        
        # Try to fix common JSON issues
        # Remove trailing commas
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        # Remove comments (if any)
        candidate = re.sub(r'//.*?$', '', candidate, flags=re.MULTILINE)
        candidate = re.sub(r'/\*.*?\*/', '', candidate, re.DOTALL)
        
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    
    return None


def extract_state_floats_from_state_dict(state_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract state Float value from state dictionary (stress, trust, dominance, focus)   
    
    Args:
        state_dict: State dictionary containing emotion and relation
    
    Returns:
        State Float value dictionary
    """
    state_floats = {
        'stress': 0.5,
        'trust': 0.5,
        'dominance': 0.5,
        'focus': 0.5
    }
    
    # Extract stress from emotion
    emotion = state_dict.get('emotion', {})
    if 'stress' in emotion:
        state_floats['stress'] = float(emotion['stress'])
    
    # Extract trust and dominance from relation
    relation = state_dict.get('relation', {})
    if 'Trust' in relation:
        state_floats['trust'] = float(relation['Trust'])
    elif 'trust' in relation:
        state_floats['trust'] = float(relation['trust'])
    
    if 'Dominance' in relation:
        state_floats['dominance'] = float(relation['Dominance'])
    elif 'dominance' in relation:
        state_floats['dominance'] = float(relation['dominance'])
    
    # focus can be derived from emotion's energy or other related dimensions
    # Here temporarily use default value, can be adjusted according to actual needs
    if 'energy' in emotion:
        state_floats['focus'] = float(emotion['energy'])
    elif 'conscientiousness' in emotion:
        state_floats['focus'] = float(emotion['conscientiousness'])
    
    return state_floats


def build_state_tags(state_floats: Dict[str, float]) -> str:
    """
    Build state Label string (fixed format: 4 lines, no indentation, force all 4 keys to appear)

    Args:
        state_floats: State Float value dictionary
    
    Returns:
        Fixed format pure Text label, Example:
        Stress: medium
        Trust: high
        Dominance: low
        Focus: high
    """
    # Force 4 dimensions in fixed order
    dimensions = ['stress', 'trust', 'dominance', 'focus']
    tags = []
    
    for dim in dimensions:
        value = state_floats.get(dim, 0.5)
        tag = map_float_to_tag(value)
        dim_capitalized = dim.capitalize()
        tags.append(f"{dim_capitalized}: {tag}")
    
    # Return fixed format: 4 lines, no indentation, one key per line
    return "\n".join(tags)


def rebuild_gpt_value(
    observation_analysis: str,
    self_reflection: str,
    state_update_plan: str,
    state_tags: str,
    state_floats: Dict[str, float],
    original_response: str
) -> str:
    """
    Rebuild gpt's value field (containing XML format text and metadata)
    
    Args:
        observation_analysis: Observation analysis
        self_reflection: Self-reflection
        state_update_plan: State update plan
        state_tags: State Label string
        state_floats: State Float value
        original_response: Original response text
    
    Returns:
        Rebuilt value string
    """
    # Build XML format text part (according to task1.md format requirements)
    # Clean content: remove leading and trailing whitespace, avoid duplicate newlines
    # Also remove literal \n strings (displayed as \\n in JSON)
    observation_analysis = observation_analysis.strip()
    observation_analysis = re.sub(r'^\\n+', '', observation_analysis)  # Remove leading literal \n
    self_reflection = self_reflection.strip()
    self_reflection = re.sub(r'^\\n+', '', self_reflection)  # Remove leading literal \n
    state_update_plan = state_update_plan.strip()
    original_response = original_response.strip()
    
    value = f"""<Internal Monologue>
  <Observation Analysis>
{observation_analysis}
  </Observation Analysis>

  <Self-Reflection>
{self_reflection}
  </Self-Reflection>

  <State Update Plan>
{state_update_plan}
  </State Update Plan>
</Internal Monologue>

<Current State>
{state_tags}
</Current State>

<Response>
{original_response}
</Response>
</assistant_end>"""
    
   
    value = re.sub(r'(<Observation Analysis>\n)\n+', r'\1', value)
    value = re.sub(r'(<Self-Reflection>\n)\n+', r'\1', value)
    value = re.sub(r'(<Observation Analysis>\n)\\n', r'\1', value)
    value = re.sub(r'(<Self-Reflection>\n)\\n', r'\1', value)
    
    return value


# ==================== Main Processing Functions ====================

def check_data_quality(
    data: Dict[str, Any],
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Check data quality, detect bad samples
    
    Judgment rules (marked as bad if any rule is hit):：
    1. <Observation Analysis> contains keywords：The dialogue / The agent / The interaction / The speaker
    2. <Observation Analysis> doesn't have first-person opening (like I notice / I think / I feel) and subject is not I
    3. <State Update Plan> mentions Big5 words：openness conscientiousness extraversion agreeableness neuroticism
    4. plan says increase but state/float doesn't change in same direction (threshold tolerance)
    
    Args:
        data: Conversation data
        verbose: Whether to output detailed information
    
    Returns:
        (is_valid, error_messages) Tuple
    """
    errors = []
    conversations = data.get('conversations', [])
    
    # Extract system information
    system_prompt = ""
    user_name = ""
    other_person_name = ""
    for conv in conversations:
        if conv.get('from') == 'system':
            system_prompt = conv.get('value', '')
            user_name = extract_user_name_from_system(system_prompt)
            other_person_name = extract_other_person_name_from_system(system_prompt)
            break
    
    # Check all gpt turns (unified format: assistant converted to gpt)
    for i, turn in enumerate(conversations):
        if turn.get('from') not in ['gpt', 'assistant']:
            continue
        # Unified format: convert assistant to gpt
        if turn.get('from') == 'assistant':
            turn['from'] = 'gpt'
        
        gpt_value = turn.get('value', '')
        
        # Extract Observation Analysis
        obs_match = re.search(r'<Observation Analysis>(.*?)</Observation Analysis>', gpt_value, re.DOTALL)
        if obs_match:
            observation = obs_match.group(1).strip()
            
            # Check 1: contains placeholder or empty talk (stricter check)
            placeholder_keywords = [
                'analyze the current situation', 'think about the response', 'prepare to update state',
                'analyze the situation', 'think about how to reply', 'update state accordingly',
                'analyze', 'think about', 'prepare', 'consider', 'reflect on'  # If appears alone and without specific content
            ]
            # Check if it's placeholder: if only contains these keywords without specific content
            observation_lower = observation.lower().strip()
            has_placeholder = False
            for keyword in placeholder_keywords:
                if keyword in observation_lower:
                    # If entire observation is mainly placeholder keywords, mark as error
                    if len(observation_lower) < 50 or observation_lower.count(keyword) / len(observation_lower.split()) > 0.3:
                        has_placeholder = True
                        break
            
            if has_placeholder or len(observation.strip()) < 20:  # Too short may also be placeholder
                errors.append(f"Turn {i}: Observation Analysis contains placeholder/template content or is too short")
            
            # Use word boundaries and context checking, avoid misjudging reasonable references in first-person as God's-eye-view
            god_eye_patterns = [
                # Obvious God's-eye-view patterns: sentences starting with these phrases
                r'^(?:The\s+)?(?:dialogue|agent|interaction|speaker|conversation)\s+(?:shows|indicates|demonstrates|reveals|suggests|means)',
                r'^(?:The\s+)?(?:dialogue|agent|interaction|speaker|conversation)\s+is\s+(?:that|when|where|how)',
                # God's-eye-view description: used as subject
                r'\b(?:The\s+)?(?:dialogue|agent|interaction|speaker|conversation)\s+(?:appears|seems|looks|feels)\s+',
            ]
            
            observation_lower = observation.lower()
            # Check if contains obvious God's-eye-view pattern
            has_god_eye_view = False
            for pattern in god_eye_patterns:
                if re.search(pattern, observation_lower, re.IGNORECASE | re.MULTILINE):
                    # Check if in first-person context (to avoid misjudgment)
                    # If sentence starts with "I notice"、"I see"、"I observe" etc., not God's-eye-view
                    first_person_intro = r'^(?:I\s+(?:notice|see|observe|notice\s+that|see\s+that|observe\s+that|can\s+see|can\s+notice))'
                    if not re.search(first_person_intro, observation_lower, re.IGNORECASE | re.MULTILINE):
                        has_god_eye_view = True
                        matched_keyword = re.search(pattern, observation_lower, re.IGNORECASE | re.MULTILINE).group(0)
                        errors.append(f"Turn {i}: Observation Analysis contains God's-eye view pattern: '{matched_keyword[:50]}...'")
                        break
            
            # If the above pattern Match not found, check simple keywords (but more strictly)
            if not has_god_eye_view:
                god_eye_keywords = ['the dialogue', 'the agent', 'the interaction', 'the speaker', 'the conversation']
                for keyword in god_eye_keywords:
                    # Use word boundary Match, avoid partial Match
                    pattern = rf'\b{re.escape(keyword)}\b'
                    if re.search(pattern, observation_lower, re.IGNORECASE):
                        # Check if in first-person context
                        # If keyword before has "I notice"、"I see" etc., not God's-eye-view
                        keyword_pos = observation_lower.find(keyword)
                        before_keyword = observation_lower[:keyword_pos]
                        # Check if there is first-person introduction before
                        has_first_person_before = bool(re.search(
                            r'(?:I\s+(?:notice|see|observe|can\s+see|can\s+notice)|I\s+notice\s+that|I\s+see\s+that)',
                            before_keyword, re.IGNORECASE
                        ))
                        # Check if keyword is used as subject (more likely God's-eye-view)
                        # If keyword followed by verb (like shows, indicates etc.), likely God's-eye-view
                        after_keyword = observation_lower[keyword_pos + len(keyword):keyword_pos + len(keyword) + 30]
                        is_subject_usage = bool(re.search(
                            r'\s+(?:shows|indicates|demonstrates|reveals|suggests|means|appears|seems|looks|feels|is|was|are|were)',
                            after_keyword, re.IGNORECASE
                        ))
                        
                        if not has_first_person_before and is_subject_usage:
                            errors.append(f"Turn {i}: Observation Analysis contains God's-eye view: '{keyword}' (used as subject)")
                            break
            
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', observation))
            if has_chinese:
                errors.append(f"Turn {i}: Observation Analysis contains Chinese characters (must be English only)")

            has_i_in_text = bool(re.search(r'\bI\s+', observation, re.IGNORECASE))
            # If no "I" at all, mark as error (but allow some edge cases)
            if not has_i_in_text and len(observation) > 20:  # Only check longer text
                # Check if contains obvious God's-eye-view keywords (more strict check)
                strict_god_eye = ['the dialogue', 'the agent', 'the interaction']
                has_strict_god_eye = any(keyword in observation.lower() for keyword in strict_god_eye)
                if has_strict_god_eye:
                    errors.append(f"Turn {i}: Observation Analysis lacks first-person perspective and contains God's-eye view")
        
        reflection_match = re.search(r'<Self-Reflection>(.*?)</Self-Reflection>', gpt_value, re.DOTALL)
        if reflection_match:
            reflection = reflection_match.group(1).strip()
            
            placeholder_keywords = [
                'think about how to reply', 'prepare to respond',
                'think about the response', 'prepare to reply', 'think about', 'prepare to update state',
                'analyze the current situation', 'think about the response', 'prepare to update',
                'prepare to', 'analyze', 'think about', 'prepare', 'consider', 'reflect on'
            ]
            has_placeholder = any(keyword.lower() in reflection.lower() for keyword in placeholder_keywords)
            if has_placeholder or len(reflection.strip()) < 15:
                errors.append(f"Turn {i}: Self-Reflection contains placeholder/template content or is too short")
            
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', reflection))
            if has_chinese:
                errors.append(f"Turn {i}: Self-Reflection contains Chinese characters (must be English only)")
            
            generic_phrases = ['this makes me', 'i feel', 'i think', 'this is']
            if all(phrase in reflection.lower() for phrase in generic_phrases[:2]) and len(reflection.strip()) < 30:
                errors.append(f"Turn {i}: Self-Reflection is too generic and lacks specific content")
        
        # Extract State Update Plan
        plan_match = re.search(r'<State Update Plan>(.*?)</State Update Plan>', gpt_value, re.DOTALL)
        if plan_match:
            plan = plan_match.group(1).strip()
            
            # Check if mentions Big5 trait
            big5_keywords = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
            for keyword in big5_keywords:
                if keyword.lower() in plan.lower():
                    errors.append(f"Turn {i}: State Update Plan mentions Big-5 trait: '{keyword}'")
                    break
            
            # Check if contains placeholder or empty talk (Issue 5 fix - more strict check)
            placeholder_keywords = [
                'prepare to update state', 'update state accordingly',
                'update state', 'prepare state update', 'analyze the situation',
                'think about', 'prepare to', 'analyze'
            ]
            has_placeholder = any(keyword.lower() in plan.lower() for keyword in placeholder_keywords)
            if has_placeholder:
                errors.append(f"Turn {i}: State Update Plan contains placeholder (must be specific discrete actions)")
            
            # Check if contains valid discrete actions (must contain all 4 dimensions)
            valid_change_types = ['increase a lot', 'increase a little', 'stable', 'decrease a little', 'decrease a lot']
            has_valid_action = any(change_type in plan.lower() for change_type in valid_change_types)
            if not has_valid_action and len(plan.strip()) > 0:
                # If plan is not empty but no valid discrete actions, may be placeholder
                errors.append(f"Turn {i}: State Update Plan does not contain valid discrete actions (must use: increase a lot/little, stable, decrease a little/lot)")
            
            # Check if contains all 4 dimensions
            required_dimensions = ['stress', 'trust', 'dominance', 'focus']
            plan_lower = plan.lower()
            missing_dimensions = [dim for dim in required_dimensions if dim not in plan_lower]
            if missing_dimensions:
                errors.append(f"Turn {i}: State Update Plan missing dimensions: {', '.join(missing_dimensions)}")
            
            plan_lines = [line.strip() for line in plan.split('\n') if line.strip()]
            if len(plan_lines) < 4:
                errors.append(f"Turn {i}: State Update Plan must have exactly 4 lines (one per dimension), found {len(plan_lines)}")
        
        if plan_match:
            plan = plan_match.group(1).strip()
            plan_deltas = parse_plan_to_delta(plan)
            
            metadata = data.get('_metadata', {})
            state_floats = metadata.get('state_float', {})
            turn_key = f"turn_{i+1}"  # turn starts from 1
            current_state = state_floats.get(turn_key, {})
            
            if i > 0:
                prev_turn_key = f"turn_{i}"
                prev_state = state_floats.get(prev_turn_key, {})
            else:
                prev_state = metadata.get('initial_state', {})
        
            for dim in ['stress', 'trust', 'dominance', 'focus']:
                delta = plan_deltas.get(dim, 0.0)
                prev_val = prev_state.get(dim, 0.5)
                curr_val = current_state.get(dim, 0.5)
                
                tolerance = 0.10  
                
                if delta > 0.05:  # increase a lot
                    if curr_val < prev_val - tolerance:
                        errors.append(f"Turn {i}: Plan says {dim} increase a lot, but state decreased significantly ({prev_val:.3f} -> {curr_val:.3f})")
                elif delta > 0.01:  # increase a little
                    if curr_val < prev_val - tolerance * 0.5:  # More lenient for small changes
                        errors.append(f"Turn {i}: Plan says {dim} increase, but state decreased ({prev_val:.3f} -> {curr_val:.3f})")
                elif delta < -0.05:  # decrease a lot
                    if curr_val > prev_val + tolerance:
                        errors.append(f"Turn {i}: Plan says {dim} decrease a lot, but state increased significantly ({prev_val:.3f} -> {curr_val:.3f})")
                elif delta < -0.01:  # decrease a little
                    if curr_val > prev_val + tolerance * 0.5:  # More lenient for small changes
                        errors.append(f"Turn {i}: Plan says {dim} decrease, but state increased ({prev_val:.3f} -> {curr_val:.3f})")
    
    is_valid = len(errors) == 0
    
    if verbose and errors:
        print(f"Data quality check failed with {len(errors)} errors:")
        for error in errors[:10]:  # Only show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    return is_valid, errors


def extract_user_name_from_system(system_prompt: str) -> str:
    """
    Extract User name from system prompt
    
    Args:
        system_prompt: System prompt
    
    Returns:
        User name, if invalid return empty string
    """
    # Try to match "Your name: ..." or similar pattern
    name_match = re.search(r'Your name:\s*([^\n]+)', system_prompt, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        # Clean possible parentheses content
        name = re.sub(r'\([^)]*\)', '', name).strip()
        # Check if it's invalid value ("you" etc.)
        if name.lower() in ['you', 'yourself', 'your', 'your name', '']:
            return ""  # Return empty string, indicating invalid
        return name
    
    # Try to match "Character name: ..."
    name_match = re.search(r'Character name:\s*([^\n]+)', system_prompt, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        name = re.sub(r'\([^)]*\)', '', name).strip()
        if name.lower() in ['you', 'yourself', 'your', 'your name', '']:
            return ""
        return name
    
    return ""  


def extract_other_person_name_from_system(system_prompt: str) -> str:
    # Try to match "Other person's name: ..." 
    name_match = re.search(r"Other person'?s name:\s*([^\n]+)", system_prompt, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        # Clean possible parentheses content
        name = re.sub(r'\([^)]*\)', '', name).strip()
        return name
    
    return ""


def extract_name_from_conversation_history(conversations: List[Dict[str, Any]], is_user: bool = True) -> Optional[str]:
    
    if not conversations:
        return None
    
    excluded_words = {'i', 'the', 'this', 'that', 'hey', 'oh', 'my', 'god', 
                   'january', 'february', 'march', 'april', 'may', 'june', 
                   'july', 'august', 'september', 'october', 'november', 'december',
                   'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                   'yes', 'no', 'ok', 'okay', 'thanks', 'thank', 'please', 'sorry'}
    
    # Extract name from Response
    if is_user:
        for turn in conversations:
            if turn.get('from') == 'human':
                value = turn.get('value', '')
                # Match "Hey [Name]"  "[Name],"
                patterns = [
                    r'Hey\s+([A-Z][a-z]+)',
                    r'Hi\s+([A-Z][a-z]+)',
                    r'([A-Z][a-z]+),\s+',
                    r'^([A-Z][a-z]+)\s+',  # Name at the beginning of the sentence
                ]
                for pattern in patterns:
                    match = re.search(pattern, value, re.IGNORECASE)
                    if match:
                        name = match.group(1).strip()
                        if name.lower() not in excluded_words and len(name) > 1:
                            return name
    else:
        # Extract name from Response
        for turn in conversations:
            if turn.get('from') in ['gpt', 'assistant']:
                value = turn.get('value', '')
                # Extract Response section
                response_match = re.search(r'<Response>(.*?)</Response>', value, re.DOTALL)
                if response_match:
                    response = response_match.group(1)
                    patterns = [
                        r'Hey\s+([A-Z][a-z]+)',
                        r'Hi\s+([A-Z][a-z]+)',
                        r'([A-Z][a-z]+),\s+',
                        r'^([A-Z][a-z]+)\s+',  # Name at the beginning of the sentence
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            name = match.group(1).strip()
                            if name.lower() not in excluded_words and len(name) > 1:
                                return name
    
    name_counts = {}
    for turn in conversations:
        value = turn.get('value', '')
        potential_names = re.findall(r'\b([A-Z][a-z]+)\b', value)
        for name in potential_names:
            if name.lower() not in excluded_words and len(name) > 1:
                name_counts[name] = name_counts.get(name, 0) + 1
    
    if name_counts:
        sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
        for name, count in sorted_names:
            if count >= 2:
                return name
    
    return None


def process_single_conversation(
    data: Dict[str, Any],
    llm,
    verbose: bool = False,
    skip_bad_quality: bool = True, 
    retry_bad_quality: bool = False,
    max_retries: int = 1,
    use_cache: bool = False 
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Handle single conversation, implement all requirements  
    
    Args:
        data: ShareGPT format Conversation data
        llm: LLM instance
        verbose: Whether to output detailed information
        skip_bad_quality: Whether to skip if quality check fails (no data returned)
        retry_bad_quality: Whether to retry if quality check fails
        max_retries: Maximum number of retries
    
    Returns:
        (Handled data, statistics, whether to skip)
        If skip_bad_quality=True and quality check fails, return (None, stats, True)
    """
    stats = {
        'api_calls': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'turns_processed': 0    ,
        'cache_stats': {
            'cache_created': 0,  
            'cache_reused': 0,  
            'cache_hits': 0,  
            'cache_deleted': 0,  
            'cache_errors': 0  
        } if use_cache else None
    }
    
    try:
        conversations = data.get('conversations', [])
        if not conversations:
            return data, stats, False
        
    
        conversations = normalize_conversation_length(conversations)
    
        system_prompt = ""
        for conv in conversations:
            if conv.get('from') == 'system':
                system_prompt = conv.get('value', '')
                break
        
        if not system_prompt:
            if verbose:
                print("Warning: No system prompt found")
            return data, stats, False
        
        user_name = extract_user_name_from_system(system_prompt)
        other_person_name = extract_other_person_name_from_system(system_prompt)
        if not user_name or user_name.lower() in ['you', 'yourself', 'your', 'your name']:
            if verbose:
                print(f"Warning: Invalid user name '{user_name}' in system prompt. Attempting to extract from conversation...")
            for turn in conversations:
                if turn.get('from') == 'human':
                    human_value = turn.get('value', '')
                    name_patterns = [
                        r',\s+([A-Z][a-z]+)(?:\s|$|\.|!)',  
                        r'([A-Z][a-z]+),\s+',  
                        r'You\'re\s+not\s+alone[^,]*,\s+([A-Z][a-z]+)',  
                    ]
                    for pattern in name_patterns:
                        match = re.search(pattern, human_value, re.IGNORECASE)
                        if match:
                            extracted_name = match.group(1).strip()
                            if extracted_name.lower() not in ['i', 'the', 'this', 'that', 'hey', 'oh', 'my', 'god']:
                                user_name = extracted_name
                                if verbose:
                                    print(f"  Extracted user name from human message: '{user_name}'")
                                break
                    if user_name and user_name.lower() not in ['you', 'yourself', 'your', 'your name']:
                        break
            
            # Extract name from Conversation history
            if not user_name or user_name.lower() in ['you', 'yourself', 'your', 'your name']:
                extracted_name = extract_name_from_conversation_history(conversations, is_user=True)
                if extracted_name:
                    user_name = extracted_name
                    if verbose:
                        print(f"  Extracted user name from conversation history: '{user_name}'")
            
            # If still not found, generate a default name (based on conversation ID or random)
            if not user_name or user_name.lower() in ['you', 'yourself', 'your', 'your name']:
                # Use part of the conversation ID or generate a name
                conv_id = data.get('id', '')
                if conv_id:
                    # Try to extract meaningful part from ID
                    user_name = f"Character_{conv_id[:8]}" if len(conv_id) > 8 else f"Character_{conv_id}"
                else:
                    user_name = "Character"
                if verbose:
                    print(f"  Could not extract name, using generated default: '{user_name}'")
        
        # Handle other person name
        if not other_person_name or other_person_name.lower() in ['the other person', 'other person', '']:
            if verbose:
                print(f"Warning: Invalid other person name '{other_person_name}' in system prompt. Attempting to extract from conversation history...")
            extracted_name = extract_name_from_conversation_history(conversations, is_user=False)
            if extracted_name:
                other_person_name = extracted_name
                if verbose:
                    print(f"  Extracted other person name from conversation: '{other_person_name}'")
            else:
                other_person_name = other_person_name or "the other person"
                if verbose:
                    print(f"  Could not extract other person name, using default: '{other_person_name}'")
        
        if not user_name or user_name.lower() in ['you', 'yourself', 'your', 'your name']:
            extracted_name = extract_name_from_conversation_history(conversations, is_user=True)
            if extracted_name:
                user_name = extracted_name
            else:
                # Use conversation ID to generate a name
                conv_id = data.get('id', '')
                if conv_id:
                    user_name = f"Character_{conv_id[:8]}" if len(conv_id) > 8 else f"Character_{conv_id}"
                else:
                    user_name = "Character"
            if verbose:
                print(f"  Fixed user_name to: '{user_name}'")
        
        # Extract occupation and personality (using original system_prompt, not fixed_system_prompt)
        occupation = extract_occupation_from_system(system_prompt)
        personality = extract_personality_from_system(system_prompt)
        talking_style = extract_talking_style_from_system(system_prompt)
        
        if not occupation:
            occupation = "[occupation not specified]"
        if not personality:
            personality = "[personality not specified]"
        if not talking_style:
            talking_style = "[talking style not specified]"
        
        archetype_name = None
        llm_inferred_names = {'other_person_name': None, 'your_name': None}
        if personality:
            if verbose:
                print(f"Inferring archetype from personality and conversation...")
            result = infer_archetype_from_personality(personality, llm, verbose, conversations=conversations)
            stats['api_calls'] += 1
            
            # Handle result (dictionary or string)
            if isinstance(result, dict):
                archetype_name = result.get('archetype')
                llm_inferred_names['other_person_name'] = result.get('other_person_name')  
                llm_inferred_names['your_name'] = result.get('your_name')  
            else:
                archetype_name = result  
        
        # If LLM inference fails, use default archetype
        if not archetype_name or archetype_name not in ALL_ARCHETYPES:
            if verbose:
                print(f"Warning: Failed to infer archetype, using default A1_Analyst")
            archetype_name = "A1_Analyst"
        
        # If LLM inferred names, and current name is Agent_A or Agent_B, use LLM inferred name
        if llm_inferred_names['other_person_name'] and (other_person_name == "Agent_A" or other_person_name == "Agent_B"):
            if verbose:
                print(f"Replacing other_person_name '{other_person_name}' with LLM inferred name '{llm_inferred_names['other_person_name']}'")
            other_person_name = llm_inferred_names['other_person_name']
        
        if llm_inferred_names['your_name'] and (user_name == "Agent_A" or user_name == "Agent_B"):
            if verbose:
                print(f"Replacing user_name '{user_name}' with LLM inferred name '{llm_inferred_names['your_name']}'")
            user_name = llm_inferred_names['your_name']
        
        archetype_data = ALL_ARCHETYPES[archetype_name]
        big5_text_final = {
            "Openness": archetype_data.get("Openness", "medium"),
            "Conscientiousness": archetype_data.get("Conscientiousness", "medium"),
            "Extraversion": archetype_data.get("Extraversion", "medium"),
            "Agreeableness": archetype_data.get("Agreeableness", "medium"),
            "Neuroticism": archetype_data.get("Neuroticism", "medium")
        }
        initial_state_floats = archetype_data['initial_state'].copy()
        
        if verbose:
            print(f"Archetype: {archetype_name}, Big-5: {big5_text_final}, Initial State: {initial_state_floats}")
        
        new_system_prompt = rebuild_system_prompt_with_archetype(
            original_system_prompt=system_prompt,
            user_name=user_name,
            other_person_name=other_person_name,
            occupation=occupation,
            personality=personality,
            talking_style=talking_style,
            archetype_name=archetype_name,
            big5_text=big5_text_final
        )
        
        if verbose:
            print(f"New system prompt (first 200 chars): {new_system_prompt[:200]}...")
        
        for conv in conversations:
            if conv.get('from') == 'system':
                conv['value'] = new_system_prompt
                if verbose:
                    print(f"Updated system prompt in conversations")
                break
        
        # Initialize current state  Float value (will be used at turn=1)
        # For turn=1, state_float_prev = initial_state
        # For turn>=2, state_float_prev = previous turn's state_float_post
        previous_state_floats = initial_state_floats.copy()

        # Session Cache related variables
        cache_response_id = None  # To store the ID of the previous response, for reuse of cache
        first_turn_cache_id = None  # To store the ID of the first turn, for reuse of subsequent turns
        cache_available = use_cache  # To track if cache is available (if first turn fails, subsequent turns will not be attempted)
        
        # Cache debug information
        if use_cache and verbose:
            print(f"\n{'='*60}")
            print(f"Session Cache mode enabled")
            print(f"Conversation ID: {data.get('id', data.get('conversation_id', 'unknown'))}")
            print(f"{'='*60}\n")
        
        # Handle each gpt turn
        processed_conversations = []
        for i, turn in enumerate(conversations):
            if turn.get('from') == 'system':
                processed_conversations.append(turn)
                continue
            
            if turn.get('from') not in ['gpt', 'assistant']:
                processed_conversations.append(turn)
                continue
            
            # Unified format: convert assistant to gpt
            if turn.get('from') == 'assistant':
                turn['from'] = 'gpt'
            
            # This is a gpt turn    , need to regenerate the monologue
            gpt_value = turn.get('value', '')
            # turn index starts from 1: the first gpt turn is turn=1, no turn=0
            turn_index = stats['turns_processed'] + 1  # turn index: 1, 2, 3, ...
            
            # Extract original Response (keep original response content as anchor)
            # Priority:
            # 1. If gpt_value is pure text (no XML Label), the entire value is response
            # 2. If gpt_value contains <Response>Label, extract from it
            # 3. If gpt_value contains other CoSTLabel but no <Response>, try to extract after Label or remove Label and then extract
            
            original_response = ""
            
            has_cost_tags = re.search(
                r'<(Internal Monologue|Current State|Response|Observation Analysis|Self-Reflection|State Update Plan|assistant_end)[\s>]',
                gpt_value, re.IGNORECASE
            )
            
            if not has_cost_tags:
                # Case 1: pure text format, the entire gpt_value is response
                original_response = gpt_value.strip()
                if verbose and turn_index == 1:
                    print(f"  [Turn {turn_index}] Detected pure text format, use the entire value as response")
            else:
                # Case 2: contains CoST format Label, try to extract from <Response>Label
                response_match = re.search(r'<Response>(.*?)</Response>', gpt_value, re.DOTALL | re.IGNORECASE)
                if response_match:
                    original_response = response_match.group(1).strip()
                    # If the extracted is a placeholder, continue to try other methods
                    if original_response == "[Original response not available]":
                        original_response = ""
                
                # Case 3: if there is no <Response>Label, try to extract from before </assistant_end> (exclude known Label)
                if not original_response or len(original_response) < 5:
                    end_match = re.search(r'</assistant_end>', gpt_value, re.IGNORECASE)
                    if end_match:
                        # Extract content before </assistant_end>, exclude known Label part
                        before_end = gpt_value[:end_match.start()].strip()
                        # Remove known Label part (using IGNORECASE to match case changes)
                        before_end = re.sub(r'<Internal Monologue>.*?</Internal Monologue>', '', before_end, flags=re.DOTALL | re.IGNORECASE)
                        before_end = re.sub(r'<Current State>.*?</Current State>', '', before_end, flags=re.DOTALL | re.IGNORECASE)
                        before_end = re.sub(r'<Response>.*?</Response>', '', before_end, flags=re.DOTALL | re.IGNORECASE)
                        before_end = re.sub(r'<Observation Analysis>.*?</Observation Analysis>', '', before_end, flags=re.DOTALL | re.IGNORECASE)
                        before_end = re.sub(r'<Self-Reflection>.*?</Self-Reflection>', '', before_end, flags=re.DOTALL | re.IGNORECASE)
                        before_end = re.sub(r'<State Update Plan>.*?</State Update Plan>', '', before_end, flags=re.DOTALL | re.IGNORECASE)
                        if before_end and len(before_end.strip()) > 5:
                            original_response = before_end.strip()
                
                # Case 4: try to extract content after the last known Label
                if not original_response or len(original_response) < 5:
                    last_tag_pos = -1
                    for tag in ['</Internal Monologue>', '</Current State>', '</Observation Analysis>', '</Self-Reflection>', '</State Update Plan>']:
                        match = gpt_value.rfind(tag)
                        if match > last_tag_pos:
                            last_tag_pos = match + len(tag)
                    
                    if last_tag_pos > 0:
                        remaining = gpt_value[last_tag_pos:].strip()
                        remaining = re.sub(r'^<Response>\s*', '', remaining, flags=re.IGNORECASE)
                        remaining = re.sub(r'\s*</Response>.*$', '', remaining, flags=re.IGNORECASE | re.DOTALL)
                        remaining = re.sub(r'\s*</assistant_end>.*$', '', remaining, flags=re.IGNORECASE | re.DOTALL)
                        if remaining and len(remaining) > 5:
                            original_response = remaining.strip()
            
            # If still cannot extract, use placeholder (but record warning for debugging)
            if not original_response or len(original_response) < 5:
                if verbose:
                    print(f"Warning: Could not extract response from turn {turn_index}, gpt_value preview: {gpt_value[:300]}")
                    print(f"  gpt_value length: {len(gpt_value)}, has CoST tags: {bool(has_cost_tags)}")
                    if has_cost_tags:
                        tag_pattern = r'<(Internal Monologue|Current State|Response|Observation Analysis|Self-Reflection|State Update Plan|assistant_end)[\s>]'
                        print(f"  Contains tags: {re.findall(tag_pattern, gpt_value, re.IGNORECASE)}")
                original_response = "[Original response not available]"
            
            # state_float_prev: For turn=1, use initial_state; for turn>=2, use previous turn's state_float_post
            state_float_prev = previous_state_floats.copy()
            
            # Fix names in human message before Handle
            if i > 0:
                prev_turn = conversations[i-1]
                if prev_turn.get('from') == 'human':
                    original_human_value = prev_turn.get('value', '')
                    fixed_human_value = fix_human_message_names(original_human_value, user_name, other_person_name)
                    if fixed_human_value != original_human_value:
                        # Update human message in conversations
                        prev_turn['value'] = fixed_human_value
                        if verbose:
                            print(f"  Fixed names in human message at turn {i-1}: replaced undefined name with '{user_name}'")
            
            # R2: Use LLM to regenerate first-person monologue (only generate Internal Monologue, not Current State and Response)
            # Use new system prompt, and pass current turn's response as reference
            prompt = build_relabel_prompt(
                system_prompt=new_system_prompt,
                conversation_history=conversations[:i+1],
                current_turn_index=i,
                user_name=user_name,
                other_person_name=other_person_name,
                current_gpt_response=original_response,  # Pass current turn's response as reference
                previous_state=state_float_prev  # Pass previous turn's state, help LLM generate more natural state changes
            )
            
            system_message = "You are an expert at generating first-person internal monologues."
            
            # Use Session Cache (if enabled and available)  
            if cache_available and hasattr(llm, 'generate_with_cache'):
                # For the first turn, create new cache (using system prompt)
                # For subsequent turns, reuse the cache of the first turn
                if turn_index == 1:
                    # For the first turn: create cache containing system prompt
                    if verbose:
                        print(f"\n[Turn {turn_index}] Create Session Cache")
                        print(f"  System message length: {len(system_message)} chars")
                        print(f"  Prompt length: {len(prompt)} chars")
                        print(f"  Total input: {len(system_message) + len(prompt)} chars")
                    
                    success, monologue_response, response_id, cache_info = llm.generate_with_cache(
                        system_message=system_message,
                        prompt=prompt,
                        previous_response_id=None,  # For the first turn, do not pass previous_response_id
                        use_cache=True,
                        verbose=verbose
                    )
                    
                    # Check if fallback to normal call
                    if cache_info and cache_info.get('fallback_to_normal'):
                        if verbose:
                            print(f"  ✓ Fallback to normal call mode")
                        # When fallback to normal call, do not try to use cache
                        cache_available = False
                        first_turn_cache_id = None
                    elif response_id:
                        first_turn_cache_id = response_id
                        cache_response_id = response_id
                        if stats.get('cache_stats'):
                            stats['cache_stats']['cache_created'] += 1
                        if verbose:
                            print(f"  ✓ Cache ")
                            print(f"  Cache ID: {response_id}")
                            if cache_info:
                                print(f"  Cache : {cache_info}")
                    else:
                        if stats.get('cache_stats'):
                            stats['cache_stats']['cache_errors'] += 1
                        if verbose:
                            print(f"  ✗ Cache （ response_id）")
                else:
                    # turn：turncache（user message）
                    # ：turncache，first_turn_cache_id
                    if verbose:
                        print(f"\n[Turn {turn_index}]  Session Cache")
                        print(f"  Previous cache ID: {first_turn_cache_id}")
                        print(f"  Prompt length: {len(prompt)} chars ( user message，system  cache )")
                    
                    #  turn ， turn 
                    if first_turn_cache_id is None:
                        if verbose:
                            print(f"  ⚠️   turn  cache，")
                        success, monologue_response = llm.generate(system_message, prompt)
                        response_id = None
                        cache_info = None
                    else:
                        success, monologue_response, response_id, cache_info = llm.generate_with_cache(
                            system_message=system_message,
                            prompt=prompt,
                            previous_response_id=first_turn_cache_id,  # turncache
                            use_cache=True,
                            verbose=verbose
                        )
                        
                        if cache_info and cache_info.get('fallback_to_normal'):
                            if verbose:
                                print(f"  ⚠️  Cache ，")
                            # ， cache
                            cache_available = False
                            first_turn_cache_id = None
                        elif response_id:
                            cache_response_id = response_id
                            if stats.get('cache_stats'):
                                stats['cache_stats']['cache_reused'] += 1
                                stats['cache_stats']['cache_hits'] += 1
                            if verbose:
                                print(f"  ✓ Cache ")
                                print(f"  New response ID: {response_id}")
                                if cache_info:
                                    print(f"  Cache : {cache_info}")
                        else:
                            if stats.get('cache_stats'):
                                stats['cache_stats']['cache_errors'] += 1
                            if verbose:
                                print(f"  ✗ Cache （ response_id）")
            else:
                # cache，
                if verbose and turn_index == 1:
                    print(f"\n[Turn {turn_index}] （ Cache）")
                success, monologue_response = llm.generate(system_message, prompt)
                response_id = None
                cache_info = None
            
            stats['api_calls'] += 1
            
            # ：500（verbose）
            if verbose and monologue_response:
                print(f"Turn {turn_index} LLM response preview (first 500 chars): {monologue_response[:500]}")
            
            if not success or not monologue_response:
                if verbose:
                    print(f"Warning: Failed to generate monologue for turn {turn_index}")
                    print(f"  success: {success}, monologue_response length: {len(monologue_response) if monologue_response else 0}")
                    if monologue_response:
                        print(f"  monologue_response preview: {monologue_response[:200]}")
                # LLM，Current StateLabel
                # Label
                state_dict = extract_current_state_from_gpt_value(gpt_value)
                if state_dict:
                    temp_state_floats = extract_state_floats_from_state_dict(state_dict)
                    state_tags = build_state_tags(temp_state_floats)
                    # Internal Monologue（）
                    internal_match = re.search(r'<Internal Monologue>(.*?)</Internal Monologue>', gpt_value, re.DOTALL)
                    if internal_match:
                        internal_monologue = internal_match.group(0)
                    else:
                        internal_monologue = "<Internal Monologue>\n  <Observation Analysis>\n    [LLM generation failed]\n  </Observation Analysis>\n  <Self-Reflection>\n    [LLM generation failed]\n  </Self-Reflection>\n  <State Update Plan>\n    [LLM generation failed]\n  </State Update Plan>\n</Internal Monologue>"
                    
                    new_gpt_value = f"""{internal_monologue}

<Current State>
{state_tags}
</Current State>

<Response>
  {original_response}
</Response>
</assistant_end>"""
                    new_turn = turn.copy()
                    new_turn['value'] = new_gpt_value
                    processed_conversations.append(new_turn)
                else:
                    # ，turn
                    processed_conversations.append(turn)
                continue
            
            monologue_parts = parse_internal_monologue(monologue_response)
            
            # ：
            if verbose:
                print(f"Turn {turn_index} parsed monologue parts:")
                print(f"  observation_analysis: {len(monologue_parts.get('observation_analysis', ''))} chars")
                print(f"  self_reflection: {len(monologue_parts.get('self_reflection', ''))} chars")
                print(f"  state_update_plan: {len(monologue_parts.get('state_update_plan', ''))} chars")
            
            # State Update Plan（2）
            if monologue_parts.get('state_update_plan'):
                monologue_parts['state_update_plan'] = normalize_state_update_plan(monologue_parts['state_update_plan'])
            
            if not all(monologue_parts.values()):
                if verbose:
                    print(f"Warning: Failed to parse monologue for turn {turn_index}")
                    print(f"  Missing parts: {[k for k, v in monologue_parts.items() if not v]}")
                    print(f"  Full response: {monologue_response[:1000]}")
                # ，Current StateLabel
                state_dict = extract_current_state_from_gpt_value(gpt_value)
                if state_dict:
                    temp_state_floats = extract_state_floats_from_state_dict(state_dict)
                    state_tags = build_state_tags(temp_state_floats)
                    # ，
                    observation = monologue_parts.get('observation_analysis', '[Failed to parse]')
                    reflection = monologue_parts.get('self_reflection', '[Failed to parse]')
                    plan = monologue_parts.get('state_update_plan', '[Failed to parse]')
                    
                    internal_monologue = f"""<Internal Monologue>
  <Observation Analysis>
    {observation}
  </Observation Analysis>
  <Self-Reflection>
    {reflection}
  </Self-Reflection>
  <State Update Plan>
    {plan}
  </State Update Plan>
</Internal Monologue>"""
                    
                    new_gpt_value = f"""{internal_monologue}

<Current State>
{state_tags}
</Current State>

<Response>
  {original_response}
</Response>
</assistant_end>"""
                    new_turn = turn.copy()
                    new_turn['value'] = new_gpt_value
                    processed_conversations.append(new_turn)
                else:
                    # ，turn
                    processed_conversations.append(turn)
                continue
            
            # R5: （）
            # plan
            plan_dict = parse_plan_to_dict(monologue_parts['state_update_plan'])
            
            # Update state using authoritative implementation（state_float_prev）
            new_state_floats, state_delta = calculate_new_state_float(
                previous_state=state_float_prev,
                plan=plan_dict
            )
            
            # state_float_post
            state_float_post = new_state_floats.copy()
            
            # previous_state_floats
            previous_state_floats = state_float_post.copy()
            
            # R4: Label（3：statestate_float）
            # ：state_tagsstate_float_post，LLM
            state_tags = build_state_tags(state_float_post)
            
            # gpt value
            # ：state_update_plan
            normalized_plan = normalize_state_update_plan(monologue_parts['state_update_plan'])
            new_gpt_value = rebuild_gpt_value(
                observation_analysis=monologue_parts['observation_analysis'],
                self_reflection=monologue_parts['self_reflection'],
                state_update_plan=normalized_plan,  # plan
                state_tags=state_tags,  # state_float_post，
                state_floats=state_float_post,
                original_response=original_response
            )
            
            # turn
            new_turn = turn.copy()
            new_turn['value'] = new_gpt_value
            
            # assistant turnmetadata（task1.md）
            # state_float_prev（turn=1initial_state，turn>=2state_float_post）
            new_turn['metadata'] = {
                "turn_index": turn_index,
                "state_float_prev": state_float_prev.copy(),
                "state_plan": plan_dict.copy(),
                "state_delta": state_delta.copy(),
                "state_float_post": state_float_post.copy()
            }
            
            processed_conversations.append(new_turn)
            stats['turns_processed'] += 1
        
        # ，task1.md
        result_data = data.copy()
        result_data['conversations'] = processed_conversations
        
        # conversation_id（id，id）
        if 'id' in data:
            result_data['conversation_id'] = data['id']
        elif 'conversation_id' in data:
            result_data['conversation_id'] = data['conversation_id']
        else:
            result_data['conversation_id'] = ensure_conversation_id(data)
        
        # id，conversation_id（）
        if 'id' in result_data:
            del result_data['id']
        
        # metadata（task1.md）
        result_data['metadata'] = {
            "archetype": archetype_name,
            "big5": {
                "openness": big5_text_final.get("Openness", "medium").lower(),
                "conscientiousness": big5_text_final.get("Conscientiousness", "medium").lower(),
                "extraversion": big5_text_final.get("Extraversion", "medium").lower(),
                "agreeableness": big5_text_final.get("Agreeableness", "medium").lower(),
                "neuroticism": big5_text_final.get("Neuroticism", "medium").lower()
            },
            "initial_state": initial_state_floats.copy()
        }
        
        # ： turn  ShareGPT （gpt  human）
        for conv in result_data['conversations']:
            if conv.get('from') == 'assistant':
                conv['from'] = 'gpt'
            elif conv.get('from') == 'user':
                conv['from'] = 'human'
        
        # Delete Session Cache（cachecache）
        if use_cache and cache_available and first_turn_cache_id and hasattr(llm, 'delete_cache'):
            # cache
            cache_ids_to_delete = []
            if first_turn_cache_id:
                cache_ids_to_delete.append(first_turn_cache_id)
            if cache_response_id and cache_response_id != first_turn_cache_id:
                cache_ids_to_delete.append(cache_response_id)
            
            if verbose:
                print(f"\n{'='*60}")
                print(f" Session Cache")
                print(f"   cache : {len(cache_ids_to_delete)}")
                if cache_ids_to_delete:
                    print(f"  Cache IDs: {cache_ids_to_delete}")
            
            for cache_id in cache_ids_to_delete:
                if llm.delete_cache(cache_id):
                    if stats.get('cache_stats'):
                        stats['cache_stats']['cache_deleted'] += 1
                    if verbose:
                        print(f"  ✓ : {cache_id}")
                else:
                    if stats.get('cache_stats'):
                        stats['cache_stats']['cache_errors'] += 1
                    if verbose:
                        print(f"  ✗ : {cache_id}")
            
            if verbose:
                print(f"{'='*60}\n")
        
        #  Cache （ cache）
        if use_cache and stats.get('cache_stats') and verbose:
            cache_stats = stats['cache_stats']
            print(f"\n{'='*60}")
            print(f"Session Cache ")
            print(f"  Cache : {cache_stats['cache_created']}")
            print(f"  Cache : {cache_stats['cache_reused']}")
            print(f"  Cache : {cache_stats['cache_hits']}")
            print(f"  Cache : {cache_stats['cache_deleted']}")
            print(f"  Cache : {cache_stats['cache_errors']}")
            if cache_stats['cache_reused'] > 0:
                hit_rate = (cache_stats['cache_hits'] / cache_stats['cache_reused']) * 100 if cache_stats['cache_reused'] > 0 else 0
                print(f"  Cache : {hit_rate:.1f}%")
            print(f"{'='*60}\n")
        
        # 1: Handle（turn）
        is_valid, quality_errors = check_data_quality(result_data, verbose=verbose)
        if 'metadata' not in result_data:
            result_data['metadata'] = {}
        result_data['metadata']['quality_check'] = {
            'is_valid': is_valid,
            'errors': quality_errors[:20] if not is_valid else []  # 20
        }
        
        # 2: Handle（prompt，）
        # ：，（）
        should_skip = False
        if not is_valid:
            if verbose:
                print(f"❌ Data quality check FAILED for conversation {result_data.get('id', 'unknown')}")
                print(f"  Errors: {len(quality_errors)} issues found")
                # 3
                for error in quality_errors[:3]:
                    print(f"    - {error}")
            
            # ：（prompt）
            # skip_bad_quality=True，
            # skip_bad_quality=False，（）
            if skip_bad_quality:
                # skip_bad_quality，None
                if verbose:
                    print(f"  🗑️  DELETING bad sample (--skip_bad_quality enabled)")
                return None, stats, True
            else:
                # skip_bad_quality，
                # ，bad
                if verbose:
                    print(f"  ⚠️  WARNING: Bad sample will be saved but marked as invalid")
                    print(f"     Recommendation: Use --skip_bad_quality to DELETE bad samples (bad samples harm training more than bad prompts)")
                # should_skip=True，
                should_skip = True
            
            if retry_bad_quality and max_retries > 0:
                # retry_bad_quality，
                # ：，Handle
                if verbose:
                    print(f"  Quality check failed, but retry is enabled (will be handled externally)")
        
        return result_data, stats, should_skip
        
    except Exception as e:
        # cache（cache）
        if use_cache and 'cache_available' in locals() and cache_available and 'first_turn_cache_id' in locals() and first_turn_cache_id and hasattr(llm, 'delete_cache'):
            cache_ids_to_delete = []
            if first_turn_cache_id:
                cache_ids_to_delete.append(first_turn_cache_id)
            if 'cache_response_id' in locals() and cache_response_id and cache_response_id != first_turn_cache_id:
                cache_ids_to_delete.append(cache_response_id)
            
            for cache_id in cache_ids_to_delete:
                if llm.delete_cache(cache_id):
                    if stats.get('cache_stats'):
                        stats['cache_stats']['cache_deleted'] += 1
                    if verbose:
                        print(f"  ✓ : {cache_id}")
                else:
                    if stats.get('cache_stats'):
                        stats['cache_stats']['cache_errors'] += 1
                    if verbose:
                        print(f"  ✗ : {cache_id}")
        
        if verbose:
            print(f"Error processing conversation: {str(e)}")
            import traceback
            print(traceback.format_exc())
        return data, stats, False


# ==================== R6: Handle ====================

def _process_single_conversation_worker(
    data: Dict[str, Any],
    llm,
    verbose: bool,
    stats_lock: threading.Lock,
    total_stats: Dict[str, Any],
    result_queue: queue.Queue,
    skip_bad_quality: bool = True,  # ：prompt
    retry_bad_quality: bool = False,
    max_retries: int = 1,
    use_cache: bool = False  #  Session Cache
) -> None:
    """
    ：Handle（Handle）
    
    Args:
        data: Conversation data（id）
        llm: LLM instance
        verbose: Whether to output detailed information
        stats_lock: 
        total_stats: 
        result_queue: 
    """
    line_index = data.get("_line_index", 0)
    conv_id = data.get("id", "")
    
    if not conv_id:
        if verbose:
            print(f"Warning: Conversation at line {line_index} has no ID, skipping")
        result_queue.put((conv_id, data, {"api_calls": 0, "turns_processed": 0}, line_index, "No ID"))
        return
    
    try:
        relabeled_data, turn_stats, should_skip = process_single_conversation(
            data, llm, verbose, skip_bad_quality, retry_bad_quality, max_retries, use_cache
        )
        
        # （）- （prompt）
        if should_skip or relabeled_data is None:
            with stats_lock:
                total_stats["skipped_quality"] = total_stats.get("skipped_quality", 0) + 1
                total_stats["deleted_bad_samples"] = total_stats.get("deleted_bad_samples", 0) + 1
            result_queue.put((conv_id, None, turn_stats, line_index, "DELETED (quality check failed - bad sample harms training)"))
            if verbose:
                print(f"  🗑️  DELETED bad sample: {conv_id} (quality check failed - bad samples harm training more than bad prompts)")
            return
        
        # id
        if 'id' not in relabeled_data:
            relabeled_data['id'] = conv_id
        
        # （）
        with stats_lock:
            total_stats["api_calls"] += turn_stats["api_calls"]
            total_stats["input_tokens"] += turn_stats.get("input_tokens", 0)
            total_stats["output_tokens"] += turn_stats.get("output_tokens", 0)
            total_stats["total_tokens"] += turn_stats.get("total_tokens", 0)
            total_stats["total_turns"] += turn_stats["turns_processed"]
            total_stats["total_conversations"] += 1
            #  cache 
            if use_cache and turn_stats.get("cache_stats"):
                cache_stats = turn_stats["cache_stats"]
                if total_stats.get("cache_stats"):
                    total_stats["cache_stats"]["cache_created"] += cache_stats.get("cache_created", 0)
                    total_stats["cache_stats"]["cache_reused"] += cache_stats.get("cache_reused", 0)
                    total_stats["cache_stats"]["cache_hits"] += cache_stats.get("cache_hits", 0)
                    total_stats["cache_stats"]["cache_deleted"] += cache_stats.get("cache_deleted", 0)
                    total_stats["cache_stats"]["cache_errors"] += cache_stats.get("cache_errors", 0)
        
        # （IDline_index）
        result_queue.put((conv_id, relabeled_data, turn_stats, None, line_index))
        
    except Exception as e:
        if verbose:
            print(f"Error processing conversation {conv_id} (line {line_index}): {str(e)}")
            import traceback
            print(traceback.format_exc())
        result_queue.put((conv_id, data, {"api_calls": 0, "turns_processed": 0}, line_index, str(e)))


def _write_thread_worker(
    result_queue: queue.Queue,
    output_file: str,
    total_lines: int,
    processed_counter: Dict[str, int],
    processed_lock: threading.Lock,
    all_tasks_done: threading.Event,
    verbose: bool,
    pbar,
    id_to_line_index: Optional[Dict[str, int]] = None,
    total_stats: Optional[Dict[str, Any]] = None,
    stats_lock: Optional[threading.Lock] = None
):
    """
    ：Handle（ID）
    
    Args:
        result_queue: 
        output_file: 
        total_lines: 
        processed_counter: Handle
        processed_lock: Handle
        all_tasks_done: 
        verbose: Whether to output detailed information
        pbar: 
        id_to_line_index: ID（）
        total_stats: （）
        stats_lock: 
    """
    results = {}  # conversation_id -> (data, stats, error, line_index)
    written_ids = set()  # ID
    
    while True:
        try:
            # （）
            try:
                conv_id, data, stats, error, line_index = result_queue.get(timeout=1.0)
            except queue.Empty:
                # ，
                if all_tasks_done.is_set():
                    break
                continue
            
            results[conv_id] = (data, stats, error, line_index)
            
            # line_index
            with processed_lock:
                # （line_index）
                pending_results = {
                    conv_id: result for conv_id, result in results.items()
                    if conv_id not in written_ids
                }
                
                if pending_results:
                    # line_index
                    sorted_results = sorted(
                        pending_results.items(),
                        key=lambda x: x[1][3]  # line_index
                    )
                    
                    for conv_id, (data, stats, error, line_index) in sorted_results:
                        try:
                            with open(output_file, 'a', encoding='utf-8') as f:
                                # dataNone（quality check），id，checkpoint
                                if data is None:
                                    placeholder = {"id": conv_id, "_deleted": True, "_reason": error or "quality check failed"}
                                    f.write(json.dumps(placeholder, ensure_ascii=False) + '\n')
                                else:
                                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
                            
                            written_ids.add(conv_id)
                            results.pop(conv_id, None)
                            
                            if error:
                                processed_counter["failed"] += 1
                            else:
                                processed_counter["processed"] += 1
                            
                            # （）
                            if pbar:
                                processed = processed_counter["processed"]
                                failed = processed_counter["failed"]
                                total = processed + failed
                                
                                desc = f" [:{processed} :{failed}]"
                                pbar.set_description(desc)
                                pbar.update(1)
                        except Exception as e:
                            if verbose:
                                print(f"Error writing conversation {conv_id}: {str(e)}")
                            processed_counter["failed"] += 1
        
        except Exception as e:
            if verbose:
                print(f"Error in write thread: {str(e)}")
            break
    
    # （line_index）
    with processed_lock:
        remaining_results = {
            conv_id: result for conv_id, result in results.items()
            if conv_id not in written_ids
        }
        
        if remaining_results:
            sorted_results = sorted(
                remaining_results.items(),
                key=lambda x: x[1][3]  # line_index
            )
            
            for conv_id, (data, stats, error, line_index) in sorted_results:
                try:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                    
                    if error:
                        processed_counter["failed"] += 1
                    else:
                        processed_counter["processed"] += 1
                    
                    if pbar:
                        processed = processed_counter["processed"]
                        failed = processed_counter["failed"]
                        desc = f" [:{processed} :{failed}]"
                        pbar.set_description(desc)
                        pbar.update(1)
                except Exception as e:
                    if verbose:
                        print(f"Error writing remaining conversation {conv_id}: {str(e)}")
                    processed_counter["failed"] += 1


def relabel_jsonl_file_multi_worker(
    input_file: str,
    output_file: str,
    llm,
    max_workers: int = 4,
    verbose: bool = False,
    max_lines: Optional[int] = None,
    filter_file: Optional[str] = None,
    resume: bool = True,
    skip_bad_quality: bool = True,  # ：prompt
    retry_bad_quality: bool = False,
    max_retries: int = 1,
    use_cache: bool = False  #  Session Cache
):
    """
    
    Args:
        input_file: JSON/JSONL
        output_file: JSONL
        llm: LLM instance
        max_workers: 
        verbose: Whether to output detailed information
        max_lines: Handle（）
        filter_file: （）
        resume: checkpoint
        skip_bad_quality: ，（）
        retry_bad_quality: ，（，Handle）
        max_retries: 
    """
    from tqdm import tqdm
    
    all_data = []
    read_failed = 0
    
    input_path = Path(input_file)
    id_to_line_index = {}  # ID，
    
    # ：JSON，JSONL
    if input_path.suffix == '.json':
        # JSON
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                if not isinstance(data_list, list):
                    print(f"Error: JSON file does not contain a list/array")
                    return
                for idx, item in enumerate(data_list):
                    item['_line_index'] = idx
                    # ID
                    item = add_id_to_data(item, idx)
                    conv_id = item.get('id', '')
                    if conv_id:
                        id_to_line_index[conv_id] = idx
                    all_data.append(item)
        except Exception as e:
            print(f"Error reading JSON file: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return
    else:
        # JSONL，JSON
        # JSON
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                # JSON
                try:
                    content = f.read()
                    data_list = json.loads(content)
                    if isinstance(data_list, list):
                        # JSON
                        if verbose:
                            print(f"Detected JSON array format in .jsonl file, converting...")
                        for idx, item in enumerate(data_list):
                            item['_line_index'] = idx
                            # ID
                            item = add_id_to_data(item, idx)
                            conv_id = item.get('id', '')
                            if conv_id:
                                id_to_line_index[conv_id] = idx
                            all_data.append(item)
                    else:
                        # ，JSONL
                        raise ValueError("Not a JSON array")
                except (json.JSONDecodeError, ValueError):
                    # JSON，JSONL
                    if verbose:
                        print(f"Trying to read as JSONL format...")
                    f.seek(0)
                    for idx, line in enumerate(f):
                        line_stripped = line.strip()
                        if not line_stripped:
                            continue
                        
                        try:
                            data = json.loads(line_stripped)
                            if not isinstance(data, dict):
                                read_failed += 1
                                if verbose:
                                    print(f"Warning: Line {idx+1} is not a JSON object, skipping")
                                continue
                            
                            data['_line_index'] = idx
                            # ID
                            data = add_id_to_data(data, idx)
                            conv_id = data.get('id', '')
                            if conv_id:
                                id_to_line_index[conv_id] = idx
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            read_failed += 1
                            if verbose:
                                line_preview = line_stripped[:100] if len(line_stripped) > 100 else line_stripped
                                print(f"Warning: Failed to parse line {idx+1}: {str(e)}")
                                print(f"  Line preview: {line_preview}...")
                                print(f"  Line length: {len(line_stripped)} chars")
                        except Exception as e:
                            read_failed += 1
                            if verbose:
                                print(f"Warning: Unexpected error parsing line {idx+1}: {str(e)}")
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return
    
    total_lines = len(all_data)
    
    if read_failed > 0:
        print(f"Warning: Failed to parse {read_failed} lines from input file")
        print(f"Successfully loaded {total_lines} conversations")
    
    if max_lines:
        all_data = all_data[:max_lines]
        total_lines = len(all_data)
    
    # Handle
    skipped_by_filter = 0
    if filter_file:
        filter_ids = set()
        try:
            with open(filter_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            filter_data = json.loads(line)
                            conv_id = filter_data.get('conversation_id') or filter_data.get('id')
                            if conv_id:
                                filter_ids.add(str(conv_id))
                        except:
                            pass
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load filter file: {str(e)}")
        
        if filter_ids:
            filtered_data = []
            filtered_id_to_line = {}
            for data in all_data:
                conv_id = data.get('id', '')
                if conv_id and str(conv_id) in filter_ids:
                    filtered_data.append(data)
                    if conv_id in id_to_line_index:
                        filtered_id_to_line[conv_id] = id_to_line_index[conv_id]
                else:
                    skipped_by_filter += 1
            all_data = filtered_data
            id_to_line_index = filtered_id_to_line
            total_lines = len(all_data)
    
    # Handlecheckpoint（IDMatch）
    skipped_by_checkpoint = 0
    null_ids_count = 0  # quality checknull
    if resume and Path(output_file).exists():
        processed_ids = set()
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            processed_data = json.loads(line)
                            # null（quality check）
                            if processed_data is None:
                                # nullHandle，processed_ids
                                continue
                            
                            # （_deletedTrue）
                            if processed_data.get('_deleted', False):
                                # quality check，Handle，processed_ids
                                null_ids_count += 1
                                continue
                            
                            conv_id = processed_data.get('id') or processed_data.get('conversation_id')
                            if conv_id:
                                # nullprocessed_ids
                                processed_ids.add(str(conv_id))
                        except:
                            pass
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load checkpoint: {str(e)}")
        
        if processed_ids:
            remaining_data = []
            remaining_id_to_line = {}
            for data in all_data:
                conv_id = data.get('id', '')
                if conv_id and str(conv_id) in processed_ids:
                    skipped_by_checkpoint += 1
                else:
                    # Handlenull（Handle）
                    remaining_data.append(data)
                    if conv_id in id_to_line_index:
                        remaining_id_to_line[conv_id] = id_to_line_index[conv_id]
            all_data = remaining_data
            id_to_line_index = remaining_id_to_line
            total_lines = len(all_data)
            
            if verbose and skipped_by_checkpoint > 0:
                print(f"Checkpoint:  {skipped_by_checkpoint} Handle")
            if verbose and null_ids_count > 0:
                print(f"Checkpoint:  {null_ids_count} quality check，Handle")
    
    if total_lines == 0:
        print("No data to process")
        return
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not resume or not output_path.exists():
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
    
    total_stats = {
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_turns": 0,
        "total_conversations": 0,
        "skipped_quality": 0,
        "deleted_bad_samples": 0,  # （prompt）
        "cache_stats": {
            "cache_created": 0,
            "cache_reused": 0,
            "cache_hits": 0,
            "cache_deleted": 0,
            "cache_errors": 0
        } if use_cache else None
    }
    
    stats_lock = threading.Lock()
    processed_counter = {
        "processed": 0,
        "failed": 0,
        "next_write_index": 0
    }
    processed_lock = threading.Lock()
    
    result_queue = queue.Queue()
    all_tasks_done = threading.Event()
    
    # （，）
    pbar = tqdm(
        total=total_lines,
        desc="",
        unit="",
        unit_scale=False,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        disable=False
    )
    
    # （ID）
    write_thread = threading.Thread(
        target=_write_thread_worker,
        args=(result_queue, output_file, total_lines, processed_counter, 
              processed_lock, all_tasks_done, verbose, pbar, id_to_line_index,
              total_stats, stats_lock),
        daemon=False
    )
    write_thread.start()
    
    if verbose:
        print(f"Write thread started (async write enabled)")
    
    print(f"\nHandle {total_lines} ...")
    print(f": {max_workers} workers")
    if read_failed > 0:
        print(f":  {read_failed} ")
    print()
    
    # Handle
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for data in all_data:
            future = executor.submit(
                _process_single_conversation_worker,
                data=data,
                llm=llm,
                verbose=verbose,
                stats_lock=stats_lock,
                total_stats=total_stats,
                result_queue=result_queue,
                skip_bad_quality=skip_bad_quality,
                retry_bad_quality=retry_bad_quality,
                max_retries=max_retries,
                use_cache=use_cache
            )
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                if verbose:
                    print(f"Unexpected error in worker: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
        
        all_tasks_done.set()
    
    if verbose:
        print("\n...")
    write_thread.join(timeout=300)
    if write_thread.is_alive():
        pbar.write("\n⚠️  : ")
    elif verbose:
        print("✓ ")
    
    pbar.close()
    print()
    
    processed = processed_counter["processed"]
    failed = processed_counter["failed"]
    
    print(f"{'='*70}")
    print(f"{'Handle':^70}")
    print(f"{'='*70}")
    print(f"✓ Handle: {processed:,} ")
    print(f"✗ Handle: {failed:,} ")
    if read_failed > 0:
        print(f"⚠  : {read_failed:,} （，Handle）")
    if resume and skipped_by_checkpoint > 0:
        print(f"⏭  Handle: {skipped_by_checkpoint:,} （checkpoint）")
    if filter_file and skipped_by_filter > 0:
        print(f"🔍 : {skipped_by_filter:,} （）")
    if skip_bad_quality and total_stats.get("deleted_bad_samples", 0) > 0:
        print(f"❌ : {total_stats.get('deleted_bad_samples', 0):,} （ - prompt）")
    elif total_stats.get("skipped_quality", 0) > 0:
        print(f"⚠️  : {total_stats.get('skipped_quality', 0):,} （，--skip_bad_quality）")
    print(f"⚙  : {max_workers} workers")
    print(f"\n📊 API:")
    print(f"  : {total_stats['api_calls']:,}")
    if processed > 0:
        print(f"  : {total_stats['api_calls'] / processed:.2f} ")
    if total_stats['total_turns'] > 0:
        print(f"  turn: {total_stats['api_calls'] / total_stats['total_turns']:.2f} ")
    print(f"\n💾 Tokens:")
    print(f"  tokens: {total_stats['input_tokens']:,}")
    print(f"  tokens: {total_stats['output_tokens']:,}")
    print(f"  tokens: {total_stats['total_tokens']:,}")
    if processed > 0:
        print(f"  : {total_stats['total_tokens'] / processed:,.0f} tokens")
    if total_stats['total_turns'] > 0:
        print(f"  turn: {total_stats['total_tokens'] / total_stats['total_turns']:,.0f} tokens")
    
    #  Cache （ cache）
    if use_cache and total_stats.get("cache_stats"):
        cache_stats = total_stats["cache_stats"]
        print(f"\n🚀 Session Cache :")
        print(f"  Cache : {cache_stats['cache_created']:,}")
        print(f"  Cache : {cache_stats['cache_reused']:,}")
        print(f"  Cache : {cache_stats['cache_hits']:,}")
        print(f"  Cache : {cache_stats['cache_deleted']:,}")
        print(f"  Cache : {cache_stats['cache_errors']:,}")
        if cache_stats['cache_reused'] > 0:
            hit_rate = (cache_stats['cache_hits'] / cache_stats['cache_reused']) * 100
            print(f"  Cache : {hit_rate:.1f}%")
        if cache_stats['cache_created'] > 0:
            reuse_rate = (cache_stats['cache_reused'] / cache_stats['cache_created']) * 100
            print(f"  Cache : {reuse_rate:.1f}% (/)")
        if total_stats['total_turns'] > 0:
            avg_cache_per_turn = cache_stats['cache_reused'] / total_stats['total_turns']
            print(f"  turncache: {avg_cache_per_turn:.2f} ")
    
    print(f"{'='*70}")
    
    return total_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CoST（Ark）")
    parser.add_argument("--input", type=str, required=True, help="JSON/JSONL")
    parser.add_argument("--output", type=str, required=True, help="JSONL")
    parser.add_argument("--api_key", type=str, default=None, 
                        help="Ark API Key（，ARK_API_KEY）")
    parser.add_argument("--model_name", type=str, default="ep-20260119001210-xsndk", 
                        help=" (: ep-20260119001210-xsndk，)")
    parser.add_argument("--base_url", type=str, default="https://ark.cn-beijing.volces.com/api/v3",
                        help="APIURL（: https://ark.cn-beijing.volces.com/api/v3）")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Handleworker（4）")
    parser.add_argument("--verbose", action="store_true", help="")
    parser.add_argument("--max_lines", type=int, default=None, help="Handle（）")
    parser.add_argument("--filter_file", type=str, default=None,
                        help="（，，Handle）")
    parser.add_argument("--no_resume", action="store_true",
                        help="checkpoint（Handle）")
    parser.add_argument("--skip_bad_quality", action="store_true",
                        help="，（）。（prompt）")
    parser.add_argument("--keep_bad_quality", action="store_true",
                        help="，（，）。，--skip_bad_quality")
    parser.add_argument("--retry_bad_quality", action="store_true",
                        help="，（，Handle）")
    parser.add_argument("--max_retries", type=int, default=1,
                        help="（1，--retry_bad_quality）")
    parser.add_argument("--use_cache", action="store_true",
                        help=" Session Cache （conversationturnsystem promptcache）")
    
    args = parser.parse_args()
    
    # Ark LLM
    llm = ArkLLMWrapper(
        api_key=args.api_key,
        model_name=args.model_name,
        base_url=args.base_url
    )
    
    # skip_bad_quality（prompt）
    # --keep_bad_quality，skip_bad_quality
    skip_bad = (args.skip_bad_quality or not args.keep_bad_quality)  # ，--keep_bad_quality
    
    relabel_jsonl_file_multi_worker(
        input_file=args.input,
        output_file=args.output,
        llm=llm,
        max_workers=args.max_workers,
        verbose=args.verbose,
        max_lines=args.max_lines,
        filter_file=args.filter_file,
        resume=not args.no_resume,
        skip_bad_quality=skip_bad,
        retry_bad_quality=args.retry_bad_quality,
        max_retries=args.max_retries,
        use_cache=args.use_cache
    )
