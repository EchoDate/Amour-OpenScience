"""
Social Profiling: Profile Based on "Social Perspective"
Implements dual-layer Profile (Inner Profile + Relational Profile)
"""

from typing import Dict, Optional, Any
from .profile import Profile, Relation, RELATION_TYPES


class SocialProfiler:
    """
    Social Perspective Profile Manager
    
    Dual-layer Profile:
    1. Inner Profile (self layer): Who am I (personality, goals)
    2. Relational Profile (relationship layer): Specific views on each Agent
       {agent_id: {Intimacy: float, Trust: float, Dominance: float}}
    """
    
    def __init__(self, profile: Profile, use_llm: bool = True, llm_model=None):
        """
        Initialize social perspective analyzer
        
        Args:
            profile: Profile to manage
            use_llm: Whether to use LLM for relationship analysis (default True)
            llm_model: LLM model (if use_llm=True)
        """
        self.profile = profile
        self.use_llm = use_llm
        self.llm_model = llm_model
    
    def get_inner_profile(self) -> Dict[str, Any]:
        """
        Get Inner Profile (self layer)
        
        Returns:
            Dictionary containing self-awareness
        """
        inner = {
            "name": self.profile.static_traits.name,
            "occupation": self.profile.static_traits.occupation,
            "core_values": self.profile.static_traits.core_values,
            "personality": self.profile.static_traits.personality,
            "mbti": self.profile.static_traits.mbti,
            "talking_style": self.profile.static_traits.talking_style,
            "current_emotion": self.profile.emotion_vector.to_dict()
        }
        
        # Merge custom inner_profile
        inner.update(self.profile.inner_profile)
        
        return inner
    
    def get_relational_profile(self, agent_id: str) -> Dict[str, Any]:
        """
        Get Relational Profile for a specific Agent
        
        Args:
            agent_id: Target Agent ID
        
        Returns:
            Relationship dimension dictionary, e.g.: {Intimacy: 0.8, Trust: 0.2, Dominance: -0.5, RelationType: "friend"}
        """
        relation = self.profile.get_relation(agent_id)
        result = {
            "Intimacy": relation.intimacy,
            "Trust": relation.trust,
            "Dominance": relation.dominance
        }
        if relation.relation_type is not None:
            result["RelationType"] = relation.relation_type
        return result
    
    def update_relation(self, 
                       agent_id: str, 
                       interaction: str,
                       interaction_type: Optional[str] = None,
                       relation_type: Optional[str] = None):
        """
        Update relationship based on interaction
        
        Args:
            agent_id: Interacting Agent ID
            interaction: Interaction content
            interaction_type: Interaction type (such as "criticism", "praise", "conflict", etc.)
            relation_type: Relationship type (such as "ex", "teacher_student", "lover", etc.), will set if provided
        """
        relation = self.profile.get_relation(agent_id)
        
        # Set relationship type if provided
        if relation_type is not None:
            relation.set_relation_type(relation_type)
        
        # If using LLM and LLM model is provided, use LLM to judge relationship changes
        if self.use_llm and self.llm_model is not None:
            updated_relation = self._update_relation_with_llm(
                relation, agent_id, interaction, interaction_type
            )
        else:
            # Otherwise use rules/keyword matching
            updated_relation = self._update_relation_with_rules(
                relation, interaction, interaction_type
            )
        
        # Limit values to reasonable range
        updated_relation.clamp()
        
        # Update relationship in profile
        self.profile.update_relation(agent_id, updated_relation)
    
    def _update_relation_with_llm(self,
                                  relation: Relation,
                                  agent_id: str,
                                  interaction: str,
                                  interaction_type: Optional[str] = None) -> Relation:
        """
        Use LLM to judge relationship changes
        
        Args:
            relation: Current relation object
            agent_id: Target Agent ID
            interaction: Interaction content
            interaction_type: Interaction type (optional)
        
        Returns:
            Updated relation object
        """
        # Get current relationship state
        current_intimacy = relation.intimacy
        current_trust = relation.trust
        current_dominance = relation.dominance
        
        # Build prompt
        system_message = "You are an expert at analyzing social relationships. Analyze how interactions affect relationship dimensions."
        
        prompt = f"""Analyze how the following interaction affects the relationship between the current agent and {agent_id}.

Current Relationship State:
- Intimacy: {current_intimacy:.2f} (range: -1.0 to 1.0, where -1 is very distant, 1 is very intimate)
- Trust: {current_trust:.2f} (range: 0.0 to 1.0, where 0 is no trust, 1 is complete trust)
- Dominance: {current_dominance:.2f} (range: -1.0 to 1.0, where -1 means the other party dominates, 1 means you dominate)

Interaction: {interaction}
Interaction Type: {interaction_type if interaction_type else "Not specified"}

Based on the interaction, determine the changes to each relationship dimension.
Respond with a JSON object containing the changes (deltas) for each dimension, e.g.:
{{
    "intimacy": -0.05,
    "trust": -0.1,
    "dominance": 0.0
}}

Values should be reasonable (typically between -0.3 and 0.3 for each dimension).
Respond with ONLY the JSON object, no other text."""

        try:
            # Call LLM
            if hasattr(self.llm_model, 'generate'):
                success, response = self.llm_model.generate(system_message, prompt)
                if not success:
                    # If LLM call fails, fall back to rule-based method
                    return self._update_relation_with_rules(relation, interaction, interaction_type)
            else:
                # If LLM model doesn't have generate method, try direct call
                try:
                    response = self.llm_model(prompt)
                    success = True
                except Exception:
                    # If call fails, fall back to rule-based method
                    return self._update_relation_with_rules(relation, interaction, interaction_type)
            
            if not success:
                return self._update_relation_with_rules(relation, interaction, interaction_type)
            
            # Parse LLM response (try to extract JSON)
            import json
            import re
            
            # Try to extract JSON object
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    deltas = json.loads(json_str)
                except json.JSONDecodeError:
                    # If JSON parsing fails, fall back to rule-based method
                    return self._update_relation_with_rules(relation, interaction, interaction_type)
            else:
                # If no JSON found, fall back to rule-based method
                return self._update_relation_with_rules(relation, interaction, interaction_type)
            
            # Apply changes
            updated_relation = Relation(
                intimacy=max(-1.0, min(1.0, current_intimacy + deltas.get("intimacy", 0.0))),
                trust=max(0.0, min(1.0, current_trust + deltas.get("trust", 0.0))),
                dominance=max(-1.0, min(1.0, current_dominance + deltas.get("dominance", 0.0))),
                relation_type=relation.relation_type
            )
            
            return updated_relation
            
        except Exception as e:
            # If any error occurs, fall back to rule-based method
            import warnings
            warnings.warn(f"LLM-based relation update failed: {e}. Falling back to rule-based method.")
            return self._update_relation_with_rules(relation, interaction, interaction_type)
    
    def _update_relation_with_rules(self,
                                   relation: Relation,
                                   interaction: str,
                                   interaction_type: Optional[str] = None) -> Relation:
        """
        Update relationship using rules/keyword matching (backward compatible method)
        
        Args:
            relation: Current relation object
            interaction: Interaction content
            interaction_type: Interaction type (optional)
        
        Returns:
            Updated relation object
        """
        # Update relationship based on interaction type
        if interaction_type:
            if interaction_type == "criticism":
                # Being criticized -> trust decreases, intimacy may decrease
                relation.trust -= 0.1
                relation.intimacy -= 0.05
            elif interaction_type == "praise":
                # Being praised -> trust and intimacy increase
                relation.trust += 0.1
                relation.intimacy += 0.05
            elif interaction_type == "conflict":
                # Conflict -> trust decreases, intimacy decreases
                relation.trust -= 0.15
                relation.intimacy -= 0.1
            elif interaction_type == "friendly":
                # Friendly interaction -> trust and intimacy increase
                relation.trust += 0.1
                relation.intimacy += 0.1
            elif interaction_type == "help":
                # Help -> trust increases
                relation.trust += 0.15
                relation.intimacy += 0.05
            elif interaction_type == "betrayal":
                # Betrayal -> trust sharply decreases
                relation.trust -= 0.3
                relation.intimacy -= 0.2
        
        # Analyze based on interaction text (if type not provided)
        else:
            interaction_lower = interaction.lower()
            
            # Detect keywords
            if any(kw in interaction_lower for kw in ["thank", "appreciate", "help", "谢谢", "感谢", "帮助"]):
                relation.trust += 0.05
                relation.intimacy += 0.03
            elif any(kw in interaction_lower for kw in ["sorry", "apologize", "forgive", "抱歉", "对不起", "原谅"]):
                relation.trust += 0.03
            elif any(kw in interaction_lower for kw in ["blame", "fault", "wrong", "责怪", "错误"]):
                relation.trust -= 0.08
                relation.intimacy -= 0.05
        
        return relation
    
    def get_relation_constraint_prompt(self, agent_id: str) -> str:
        """
        Generate relationship constraint prompt fragment
        
        Example: "Based on your current Profile, you have very low trust but high intimacy with Agent B (possibly a toxic friend).
              Please respond under this relationship constraint."
        
        Args:
            agent_id: Target Agent ID
        
        Returns:
            Text description of relationship constraints
        """
        relation = self.profile.get_relation(agent_id)
        
        constraints = []
        
        # Trust description
        if relation.trust < 0.3:
            constraints.append("very low trust")
        elif relation.trust < 0.5:
            constraints.append("relatively low trust")
        elif relation.trust > 0.7:
            constraints.append("very high trust")
        else:
            constraints.append("moderate trust")
        
        # Intimacy description
        if relation.intimacy < -0.5:
            constraints.append("distant relationship")
        elif relation.intimacy < 0:
            constraints.append("neutral relationship")
        elif relation.intimacy > 0.5:
            constraints.append("very intimate relationship")
        elif relation.intimacy > 0:
            constraints.append("relatively intimate relationship")
        
        # Dominance description
        if relation.dominance > 0.5:
            constraints.append("you dominate in the relationship")
        elif relation.dominance < -0.5:
            constraints.append("the other party dominates in the relationship")
        else:
            constraints.append("relatively equal relationship")
        
        # Generate relationship type judgment
        # Prioritize explicitly set relationship type, otherwise infer
        relation_type = relation.relation_type or self._infer_relation_type(relation)
        
        prompt = f"Based on your current Profile, you have "
        prompt += ", ".join(constraints)
        prompt += f" with {agent_id}"
        if relation_type:
            prompt += f" (relation type: {relation_type})"
        prompt += ". Please respond under this relationship constraint."
        
        return prompt
    
    def _infer_relation_type(self, relation: Relation) -> Optional[str]:
        """
        Infer relationship type based on relationship dimensions
        
        Returns:
            Relationship type description, such as "toxic_friend", "close_friend", "rival", etc.
        """
        # High intimacy + low trust = toxic friend
        if relation.intimacy > 0.5 and relation.trust < 0.3:
            return "toxic_friend"
        
        # High intimacy + high trust = close friend
        if relation.intimacy > 0.5 and relation.trust > 0.7:
            return "close_friend"
        
        # Low intimacy + low trust = rival/enemy
        if relation.intimacy < -0.3 and relation.trust < 0.3:
            return "rival"
        
        # High trust + moderate intimacy = partner
        if relation.trust > 0.7 and -0.3 < relation.intimacy < 0.5:
            return "partner"
        
        return None
    
    def get_all_relations_summary(self) -> str:
        """
        Get summary of all relationships
        
        Returns:
            Text description of all relationships
        """
        if not self.profile.relations:
            return "No relationship records"
        
        summaries = []
        for agent_id, relation in self.profile.relations.items():
            rel_profile = self.get_relational_profile(agent_id)
            summary = f"{agent_id}: Intimacy={rel_profile['Intimacy']:.2f}, "
            summary += f"Trust={rel_profile['Trust']:.2f}, "
            summary += f"Dominance={rel_profile['Dominance']:.2f}"
            if 'RelationType' in rel_profile:
                summary += f", RelationType={rel_profile['RelationType']}"
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def set_relation_type(self, agent_id: str, relation_type: Optional[str]):
        """
        Set relationship type with specified Agent
        
        Args:
            agent_id: Target Agent ID
            relation_type: Relationship type (such as "ex", "teacher_student", "lover", etc.), or None to clear
        """
        relation = self.profile.get_relation(agent_id)
        relation.set_relation_type(relation_type)
        self.profile.update_relation(agent_id, relation)

