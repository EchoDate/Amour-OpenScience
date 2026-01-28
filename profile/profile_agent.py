"""
Profile Agent: Agent Integrated with Profile
Based on HiAgent's VanillaAgent, but integrates Profile in Memory Retrieval and Prompt Construction stages
"""

import json
import re
from typing import Optional, Tuple, Dict, Any, List
from .base_agent import BaseAgent
from .profile import Profile
from .state_tracker import StateTracker
from .social_profiling import SocialProfiler
from .memory_enhancer import MemoryEnhancer
from .prompt_enhancer import PromptEnhancer
from .utils import extract_agent_id_from_text, detect_interaction_type


class ProfileAgent(BaseAgent):
    """
    Agent Integrated with Profile
    
    Integrate Profile into HiAgent's process:
    1. Memory Retrieval: Use MemoryEnhancer to enhance memory retrieval
    2. Prompt Construction: Use PromptEnhancer to enhance prompts
    3. State Update: Use StateTracker to update emotional state
    4. Relation Update: Use SocialProfiler to update relationships
    """
    
    def __init__(self,
                 llm_model,
                 profile: Profile,
                 memory_size: int = 100,
                 examples: List[str] = None,
                 instruction: str = "",
                 init_prompt_path: Optional[str] = None,
                 system_message: str = "You are a helpful assistant.",
                 need_goal: bool = False,
                 check_actions: Optional[str] = None,
                 check_inventory: Optional[str] = None,
                 use_parser: bool = True,
                 use_llm_for_emotion: bool = True):
        """
        Initialize ProfileAgent
        
        Args:
            llm_model: LLM model (needs to implement generate(system_message, prompt) method)
            profile: Agent's Profile
            memory_size: Memory size
            examples: Example list
            instruction: Instruction
            init_prompt_path: Initial prompt path (JSON file)
            system_message: System message
            need_goal: Whether goal is needed
            check_actions: Action check prompt
            check_inventory: Inventory check prompt
            use_parser: Whether to use parser
            use_llm_for_emotion: Whether to use LLM for emotion analysis (default True, dynamic properties should be judged by LLM)
        """
        super().__init__()
        
        self.llm_model = llm_model
        self.memory_size = memory_size
        self.profile = profile
        self.use_parser = use_parser
        
        # Initialize Profile-related components
        self.state_tracker = StateTracker(profile, use_llm=use_llm_for_emotion, llm_model=llm_model)
        # Use LLM for relationship analysis by default (dynamic properties should be judged by LLM)
        self.social_profiler = SocialProfiler(profile, use_llm=True, llm_model=llm_model)
        self.memory_enhancer = MemoryEnhancer(profile)
        self.prompt_enhancer = PromptEnhancer(profile)

        self.thought = " "
        
        # Load prompt configuration
        if init_prompt_path is not None:
            with open(init_prompt_path, 'r') as f:
                self.init_prompt_dict = json.load(f)
            self.instruction = self.init_prompt_dict.get("instruction", "")
            self.examples = self.init_prompt_dict.get("examples", [])
        else:
            self.instruction = instruction
            self.examples = examples or []
            self.init_prompt_dict = {
                "examples": self.examples,
                "instruction": self.instruction,
                "system_msg": system_message
            }
        
        self.max_context_length = getattr(llm_model, 'context_length', 4096)
        self.need_goal = need_goal
        self.check_actions = check_actions
        self.check_inventory = check_inventory
        self.example_prompt = None
        
        # Set separators (for different LLM formats)
        if hasattr(llm_model, 'xml_split'):
            self.split = llm_model.xml_split
        else:
            self.split = {
                "example": [""],
                "text": [""],
                "rule": [""],
                "system_msg": [""],
                "instruction": [""],
                "goal": [""]
            }
        
        # Currently interacting Agent ID (for relationship awareness)
        self.current_agent_id: Optional[str] = None
    
    def reset(self, goal: str, init_obs: str, init_act: Optional[str] = None):
        """Reset Agent state"""
        super().reset(goal, init_obs, init_act)
        self.current_agent_id = None
    
    def update(self, action: str, state: str):
        """
        Update Agent state and Profile
        
        Args:
            action: Action executed
            state: Observed state
        """
        super().update(action, state)
        
        # 1. Extract interaction information
        agent_id = extract_agent_id_from_text(state) or self.current_agent_id
        interaction_type = detect_interaction_type(state, llm_model=self.llm_model)
        
        # 2. Update emotional state (State Tracker)
        self.state_tracker.update_emotion(
            last_dialogue=action,
            current_event=state,
            interaction_type=interaction_type
        )
        
        # 3. Update relationship (Social Profiler)
        if agent_id:
            self.social_profiler.update_relation(
                agent_id=agent_id,
                interaction=state,
                interaction_type=interaction_type
            )
            self.current_agent_id = agent_id
    
    def make_prompt(self,
                   need_goal: bool = False,
                   check_actions: Optional[str] = None,
                   check_inventory: Optional[str] = None,
                   system_message: str = '',
                   target_agent_id: Optional[str] = None) -> str:
        """
        Build prompt (integrated with Profile enhancement)
        
        Args:
            need_goal: Whether goal is needed
            check_actions: Action check prompt
            check_inventory: Inventory check prompt
            system_message: System message
            target_agent_id: Target Agent ID
        
        Returns:
            Enhanced prompt
        """
        # 1. Memory Retrieval: Use Memory Enhancer to retrieve memories
        enhanced_memory = self.memory_enhancer.retrieve_with_profile(
            memory=self.memory,
            current_agent_id=target_agent_id or self.current_agent_id,
            max_items=self.memory_size
        )
        
        # 2. Build base prompt (based on HiAgent's VanillaAgent)
        query = ""
        query += self.split["instruction"][0] + self.instruction + self.split["instruction"][-1]
        
        if isinstance(self.examples, str):
            self.examples = [self.examples]
        
        if len(self.examples) > 0:
            query += "\nHere are examples:\n" + self.split["example"][0]
            for example in self.examples:
                query += example + "\n"
            query += self.split["example"][-1]
        
        if need_goal or self.need_goal:
            query += self.split["goal"][0] + "You should perform actions to accomplish the goal: " + \
                     (self.goal or "") + "\n" + self.split["goal"][-1]
        
        if check_actions is not None or self.check_actions is not None:
            query += "You should use the following commands for help when your action cannot be understood: " + \
                     (check_actions or self.check_actions) + "\n"
        
        if check_inventory is not None or self.check_inventory is not None:
            query += "You should use the following commands for help when your action cannot be understood: inventory\n"
        
        # Use enhanced memory
        history = enhanced_memory[-self.memory_size:] if len(enhanced_memory) > self.memory_size else enhanced_memory
        input_prompt = query + "\n".join([item[0] + ": " + item[1] for item in history])
        input_prompt += "\nAction: "
        
        # 3. Prompt Construction: Use Prompt Enhancer to enhance prompt
        enhanced_prompt = self.prompt_enhancer.enhance_prompt(
            base_prompt=input_prompt,
            target_agent_id=target_agent_id or self.current_agent_id,
            include_profile=True,
            include_emotion=True,
            include_relation=True
        )
        
        # 4. Check token length (if LLM supports it)
        if hasattr(self.llm_model, 'num_tokens_from_messages'):
            messages = [
                {"role": "system", "content": system_message or self.init_prompt_dict.get('system_msg', '')},
                {"role": "user", "content": enhanced_prompt}
            ]
            num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
            max_tokens = getattr(self.llm_model, 'max_tokens', 512)
            
            while num_of_tokens > self.max_context_length - max_tokens:
                # Reduce historical memory
                if len(history) > 1:
                    history = history[1:]
                    input_prompt = query + "\n".join([item[0] + ": " + item[1] for item in history])
                    input_prompt += "\nAction: "
                    enhanced_prompt = self.prompt_enhancer.enhance_prompt(
                        base_prompt=input_prompt,
                        target_agent_id=target_agent_id or self.current_agent_id,
                        include_profile=True,
                        include_emotion=True,
                        include_relation=True
                    )
                    messages = [
                        {"role": "system", "content": system_message or self.init_prompt_dict.get('system_msg', '')},
                        {"role": "user", "content": enhanced_prompt}
                    ]
                    num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
                else:
                    break
        
        return enhanced_prompt
    
    def action_parser_for_special_llms(self, action: str) -> str:
        """
        Parse action output from special LLMs
        
        Args:
            action: Raw action string
        
        Returns:
            Parsed action
        """
        origin_action = action
        if 'action' in action.lower():
            action_temp = action.split('\n')
            for act in action_temp:
                if "next action" in act and ':' in act:
                    idx = action_temp.index(act)
                    while idx + 1 < len(action_temp):
                        if action_temp[idx + 1]:
                            action = action_temp[idx + 1]
                            break
                        idx += 1
                if act.split(':')[0].lower().endswith('with action input'):
                    action = act
                    break
                if 'action' in act.lower() and ':' in act:
                    action_temp = ':'.join(act.split(':')[1:])
                    if action_temp != "":
                        action = action_temp
                        break
                if 'action' in act.lower() and 'is to' in act:
                    action_temp = act.split('is to')[1]
                    if action_temp != "":
                        action = action_temp
                        break
        
        action = action.strip()
        action = action.strip("'/")
        action = action.split('\n')[0]
        return action
    
    def run(self,
            init_prompt_dict: Optional[Dict] = None,
            target_agent_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Run Agent, generate next action
        
        Args:
            init_prompt_dict: Initial prompt dictionary (optional, for override)
            target_agent_id: Target Agent ID (optional)
        
        Returns:
            (success, action): Whether successful, generated action
        """
        # Override prompt configuration (if provided)
        if init_prompt_dict is not None:
            self.init_prompt_dict = init_prompt_dict
            self.instruction = init_prompt_dict.get('instruction', self.instruction)
            self.examples = init_prompt_dict.get('examples', self.examples)
        
        system_message = self.init_prompt_dict.get('system_msg', 'You are a helpful assistant.')
        
        # Build enhanced prompt
        input_prompt = self.make_prompt(
            need_goal=self.need_goal,
            check_actions=self.check_actions,
            check_inventory=self.check_inventory,
            system_message=system_message,
            target_agent_id=target_agent_id
        )
        
        self.log_example_prompt(input_prompt)
        
        # Call LLM to generate action
        if hasattr(self.llm_model, 'generate'):
            success, action = self.llm_model.generate(system_message, input_prompt)
        else:
            # If LLM model doesn't have generate method, try direct call
            try:
                response = self.llm_model(input_prompt)
                success, action = True, response
            except Exception as e:
                success, action = False, f"Error: {str(e)}"
        
        # Parse action (if needed)
        if success and self.use_parser:
            action = self.action_parser_for_special_llms(action)
        
        return success, action
    
    def log_example_prompt(self, prompt: str):
        """Log example prompt"""
        self.example_prompt = prompt
    
    def get_example_prompt(self) -> Optional[str]:
        """Get example prompt"""
        return self.example_prompt
    
    def get_profile_summary(self) -> str:
        """Get Profile summary"""
        return self.profile.get_summary()
    
    def get_emotion_state(self) -> Dict[str, float]:
        """Get current emotional state"""
        return self.profile.emotion_vector.to_dict()
    
    def get_relations_summary(self) -> str:
        """Get summary of all relationships"""
        return self.social_profiler.get_all_relations_summary()
    
    @classmethod
    def from_config(cls,
                   llm_model,
                   profile: Profile,
                   config: Dict[str, Any]) -> 'ProfileAgent':
        """
        Create ProfileAgent from configuration
        
        Args:
            llm_model: LLM model
            profile: Profile object
            config: Configuration dictionary
        
        Returns:
            ProfileAgent instance
        """
        memory_size = config.get("memory_size", 100)
        instruction = config.get("instruction", "")
        examples = config.get("examples", [])
        init_prompt_path = config.get("init_prompt_path", None)
        system_message = config.get("system_message", "You are a helpful assistant.")
        check_actions = config.get("check_actions", None)
        check_inventory = config.get("check_inventory", None)
        use_parser = config.get("use_parser", True)
        need_goal = config.get("need_goal", False)
        use_llm_for_emotion = config.get("use_llm_for_emotion", True)
        
        return cls(
            llm_model=llm_model,
            profile=profile,
            memory_size=memory_size,
            examples=examples,
            instruction=instruction,
            init_prompt_path=init_prompt_path,
            system_message=system_message,
            need_goal=need_goal,
            check_actions=check_actions,
            check_inventory=check_inventory,
            use_parser=use_parser,
            use_llm_for_emotion=use_llm_for_emotion
        )

