"""
Memory Enhancer: Enhance Memory Retrieval
Consider Profile information when retrieving memory
"""

from typing import List, Dict, Any, Optional, Tuple
from .profile import Profile


class MemoryEnhancer:
    """
    Enhance Memory Retrieval with Profile information
    
    When retrieving memory, consider:
    1. Current state of Profile
    2. Relationship with current conversation agent
    3. Impact of emotional state on memory selection
    """
    
    def __init__(self, profile: Profile):
        """
        Initialize Memory Enhancer
        
        Args:
            profile: Agent's Profile
        """
        self.profile = profile
    
    def retrieve_with_profile(self, 
                             memory: List[Tuple[str, str]], 
                             current_agent_id: Optional[str] = None,
                             max_items: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Retrieve relevant memories based on Profile
        
        Args:
            memory: Original memory list, format: [("Action", "..."), ("Observation", "..."), ...]
            current_agent_id: Current interacting Agent ID (if any)
            max_items: Maximum number of items to return
        
        Returns:
            Enhanced memory list
        """
        if not memory:
            return memory
        
        # If current_agent_id is provided, prioritize memories related to that agent
        if current_agent_id:
            relevant_memory = self._filter_by_agent(memory, current_agent_id)
            if relevant_memory:
                memory = relevant_memory + [m for m in memory if m not in relevant_memory]
        
        # Adjust memory selection based on emotional state
        # E.g.: High stress may make negative memories more accessible
        memory = self._weight_by_emotion(memory)
        
        # Limit number of returns
        if max_items and len(memory) > max_items:
            memory = memory[:max_items]
        
        return memory
    
    def _filter_by_agent(self, 
                        memory: List[Tuple[str, str]], 
                        agent_id: str) -> List[Tuple[str, str]]:
        """
        Filter memories related to a specific Agent
        """
        relevant = []
        agent_id_lower = agent_id.lower()
        
        for item in memory:
            content = item[1].lower()
            # Check if content contains agent_id
            if agent_id_lower in content:
                relevant.append(item)
        
        return relevant
    
    def _weight_by_emotion(self, memory: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Weight and sort memories based on emotional state
        
        Emotional state affects memory selection:
        - High Neuroticism: May be more likely to recall negative memories
        - High Trust: May be more likely to recall positive memories
        - High Stress: May be more likely to recall stress-related events
        """
        emotion = self.profile.emotion_vector
        
        # Simple weighting strategy: adjust memory order based on emotional state
        # More complex weighting algorithms can be implemented as needed
        
        # If stress is high, prioritize recent memories (as they may be more relevant)
        if emotion.stress > 0.7:
            # Keep original order, but prioritize recent ones
            return memory
        
        # If trust is high, can retain more historical memories
        if emotion.trust > 0.7:
            return memory
        
        # By default, maintain original order
        return memory
    
    def add_profile_context_to_memory(self, 
                                      memory: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Add Profile context before memory
        
        Args:
            memory: Original memory list
        
        Returns:
            Memory list with added Profile context
        """
        profile_context = [
            ("Profile", self.profile.get_summary())
        ]
        
        return profile_context + memory
    
    def get_relation_aware_memory(self,
                                 memory: List[Tuple[str, str]],
                                 agent_id: str) -> List[Tuple[str, str]]:
        """
        Get relationship-aware memory
        
        Select relevant memories based on relationship with specific Agent
        
        Args:
            memory: Original memory list
            agent_id: Target Agent ID
        
        Returns:
            Relationship-aware memory list
        """
        from .social_profiling import SocialProfiler
        
        profiler = SocialProfiler(self.profile)
        relation = profiler.get_relational_profile(agent_id)
        
        # If trust is high, can include more memories related to that agent
        if relation["Trust"] > 0.7:
            # Prioritize memories containing that agent
            agent_memory = self._filter_by_agent(memory, agent_id)
            other_memory = [m for m in memory if m not in agent_memory]
            return agent_memory + other_memory
        
        # If trust is low, may only keep essential memories
        elif relation["Trust"] < 0.3:
            # Only keep recent memories
            return memory[-10:] if len(memory) > 10 else memory
        
        # Default case
        return memory

