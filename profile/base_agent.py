"""
Base Agent: Base Agent Class
Based on HiAgent design, but without dependency on its framework
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Dict


class BaseAgent(ABC):
    """
    Base Agent Class
    Based on HiAgent's BaseAgent design
    """
    
    def __init__(self):
        """Initialize Agent"""
        self.goal = None
        self.init_obs = None
        self.memory: List[Tuple[str, str]] = []
        self.steps = 0
        self.done = False
    
    def reset(self, goal: str, init_obs: str, init_act: Optional[str] = None):
        """
        Reset Agent state
        
        Args:
            goal: Goal
            init_obs: Initial observation
            init_act: Initial action (optional)
        """
        self.goal = goal
        self.init_obs = init_obs
        self.memory = [("Action", init_act), ('Observation', self.init_obs)] if init_act \
            else [('Observation', self.init_obs)]
        self.steps = 0
        self.done = False
    
    def update(self, action: str, state: str):
        """
        Update Agent state
        
        Args:
            action: Action executed
            state: Observed state
        """
        self.steps += 1
        self.memory.append(("Action", action))
        self.memory.append(("Observation", state))
    
    @abstractmethod
    def run(self, **kwargs) -> Tuple[bool, str]:
        """
        Run Agent, generate next action
        
        Returns:
            (success, action): Whether successful, generated action
        """
        pass
    
    def get_memory(self) -> List[Tuple[str, str]]:
        """
        Get current memory
        
        Returns:
            Memory list
        """
        return self.memory
    
    def get_memory_size(self) -> int:
        """
        Get memory size
        
        Returns:
            Number of memory items
        """
        return len(self.memory)

