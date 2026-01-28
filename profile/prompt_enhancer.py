"""
Prompt Enhancer: Enhance Prompt Construction
Add Profile constraints to prompts
"""

from typing import Optional, Dict, Any
from .profile import Profile
from .state_tracker import StateTracker
from .social_profiling import SocialProfiler


class PromptEnhancer:
    """
    Enhance Prompt Construction with Profile constraints
    
    Add to prompt:
    1. Profile information (static traits + current emotional state)
    2. Relationship constraints (if responding to a specific agent)
    3. Response style guidance
    """
    
    def __init__(self, profile: Profile):
        """
        Initialize Prompt Enhancer
        
        Args:
            profile: Agent's Profile
        """
        self.profile = profile
        self.state_tracker = StateTracker(profile)
        self.social_profiler = SocialProfiler(profile)
    
    def enhance_prompt(self, 
                      base_prompt: str,
                      target_agent_id: Optional[str] = None,
                      include_profile: bool = True,
                      include_emotion: bool = True,
                      include_relation: bool = True) -> str:
        """
        Enhance prompt with Profile information
        
        Args:
            base_prompt: Base prompt
            target_agent_id: Target Agent ID (if responding to a specific agent)
            include_profile: Whether to include Profile information
            include_emotion: Whether to include emotional state
            include_relation: Whether to include relationship constraints
        
        Returns:
            Enhanced prompt
        """
        enhanced_parts = []
        
        # 1. Add Profile information
        if include_profile:
            profile_section = self._build_profile_section()
            if profile_section:
                enhanced_parts.append(profile_section)
        
        # 2. Add emotional state guidance
        if include_emotion:
            emotion_section = self._build_emotion_section()
            if emotion_section:
                enhanced_parts.append(emotion_section)
        
        # 3. Add relationship constraints (if responding to a specific agent)
        if include_relation and target_agent_id:
            relation_section = self._build_relation_section(target_agent_id)
            if relation_section:
                enhanced_parts.append(relation_section)
        
        # Combine enhanced parts
        if enhanced_parts:
            profile_context = "\n\n".join(enhanced_parts)
            enhanced_prompt = f"{profile_context}\n\n{base_prompt}"
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt
    
    def _build_profile_section(self) -> str:
        """
        构建 Profile 信息部分
        """
        static = self.profile.static_traits
        section = "## Character Profile\n"
        section += f"Name: {static.name}\n"
        section += f"Occupation: {static.occupation}\n"
        section += f"Age: {static.age}\n"
        if static.personality:
            section += f"Personality: {static.personality}\n"
        if static.talking_style:
            section += f"Talking Style: {static.talking_style}\n"
        if static.core_values:
            section += f"Core Values: {', '.join(static.core_values)}\n"
        if static.mbti:
            section += f"MBTI: {static.mbti}\n"
        
        return section
    
    def _build_emotion_section(self) -> str:
        """
        Build emotional state section
        """
        emotion = self.profile.emotion_vector
        guidance = self.state_tracker.get_emotion_guidance()
        
        section = "## Current Emotional State\n"
        section += f"Openness: {emotion.openness:.2f}\n"
        section += f"Conscientiousness: {emotion.conscientiousness:.2f}\n"
        section += f"Extraversion: {emotion.extraversion:.2f}\n"
        section += f"Agreeableness: {emotion.agreeableness:.2f}\n"
        section += f"Neuroticism: {emotion.neuroticism:.2f}\n"
        section += f"Trust: {emotion.trust:.2f}\n"
        section += f"Stress: {emotion.stress:.2f}\n"
        section += f"Energy: {emotion.energy:.2f}\n"
        
        if guidance:
            section += f"\nResponse Guidance: {guidance}\n"
        
        return section
    
    def _build_relation_section(self, agent_id: str) -> str:
        """
        构建关系约束部分
        """
        relation = self.social_profiler.get_relational_profile(agent_id)
        constraint = self.social_profiler.get_relation_constraint_prompt(agent_id)
        
        section = "## Relationship Context\n"
        section += f"Relationship with {agent_id}:\n"
        if 'RelationType' in relation:
            section += f"  - Relation Type: {relation['RelationType']}\n"
        section += f"  - Intimacy: {relation['Intimacy']:.2f}\n"
        section += f"  - Trust: {relation['Trust']:.2f}\n"
        section += f"  - Dominance: {relation['Dominance']:.2f}\n"
        section += f"\n{constraint}\n"
        
        return section
    
    def enhance_system_message(self, 
                              base_system_message: str,
                              target_agent_id: Optional[str] = None) -> str:
        """
        Enhance system message with Profile information
        
        Args:
            base_system_message: Base system message
            target_agent_id: Target Agent ID
        
        Returns:
            Enhanced system message
        """
        profile_info = self._build_profile_section()
        
        enhanced = f"{base_system_message}\n\n{profile_info}"
        
        if target_agent_id:
            relation_info = self._build_relation_section(target_agent_id)
            enhanced += f"\n{relation_info}"
        
        return enhanced
    
    def get_response_style_instruction(self, target_agent_id: Optional[str] = None) -> str:
        """
        Get response style guidance
        
        Args:
            target_agent_id: Target Agent ID
        
        Returns:
            Response style guidance text
        """
        instructions = []
        
        # Guidance based on emotional state
        emotion_guidance = self.state_tracker.get_emotion_guidance()
        if emotion_guidance:
            instructions.append(emotion_guidance)
        
        # Guidance based on relationship
        if target_agent_id:
            relation = self.social_profiler.get_relational_profile(target_agent_id)
            
            if relation["Trust"] < 0.3:
                instructions.append("Due to low trust, respond more cautiously and reservedly")
            elif relation["Trust"] > 0.7:
                instructions.append("Due to high trust, you can respond more openly and sincerely")
            
            if relation["Intimacy"] > 0.5:
                instructions.append("Due to intimate relationship, you can use a more casual tone")
            elif relation["Intimacy"] < -0.3:
                instructions.append("Due to distant relationship, use a more formal tone")
            
            if relation["Dominance"] < -0.5:
                instructions.append("Since the other party dominates in the relationship, respond with more respect and humility")
            elif relation["Dominance"] > 0.5:
                instructions.append("Since you dominate in the relationship, you can respond more confidently")
        
        # Guidance based on static traits
        if self.profile.static_traits.talking_style:
            instructions.append(f"Maintain your talking style: {self.profile.static_traits.talking_style}")
        
        return "; ".join(instructions) if instructions else ""

