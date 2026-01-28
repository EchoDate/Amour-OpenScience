"""
State Tracker: State Tracker
Update V_emotion (emotion vector) based on conversational interactions
"""

from typing import Dict, Optional, Tuple
from .profile import Profile, EmotionVector
import re


class StateTracker:
    """
    State Tracker: Update emotion vector based on conversational interactions
    
    Mechanism:
    Input: Previous round of dialogue + current environmental event
    Process: Calculate how current event changes V_emotion
    Output: Updated Profile for guiding Response generation
    """
    
    def __init__(self, profile: Profile, use_llm: bool = False, llm_model=None):
        """
        Initialize state tracker
        
        Args:
            profile: Profile to track
            use_llm: Whether to use LLM for emotion analysis (otherwise use rules)
            llm_model: LLM model (if use_llm=True)
        """
        self.profile = profile
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.last_thought = ""  # Store recent thought process
    
    def update_emotion(self, 
                      last_dialogue: str, 
                      current_event: str,
                      interaction_type: Optional[str] = None) -> Tuple[EmotionVector, str]:
        """
        Update emotion vector based on dialogue and events
        
        Args:
            last_dialogue: Previous round of dialogue content
            current_event: Current environmental event
            interaction_type: Interaction type (such as "criticism", "praise", "conflict", etc.)
        
        Returns:
            (Updated emotion vector, thought process) tuple
        """
        emotion = self.profile.emotion_vector
        
        if self.use_llm and self.llm_model:
            # Use LLM to analyze emotional changes
            updated_emotion, thought = self._update_with_llm(last_dialogue, current_event)
        else:
            # Use rules/keywords to analyze emotional changes
            updated_emotion, thought = self._update_with_rules(last_dialogue, current_event, interaction_type)
        
        # Apply update
        self.profile.emotion_vector = updated_emotion
        self.last_thought = thought  # Save thought process

        updated_emotion.clamp()  # Ensure values are in reasonable range
        
        return updated_emotion, thought
    
    def _update_with_rules(self, 
                          last_dialogue: str, 
                          current_event: str,
                          interaction_type: Optional[str] = None) -> Tuple[EmotionVector, str]:
        """
        Update emotion vector using rules
        
        Rule examples:
        - Being criticized/insulted -> Neuroticism increases, Trust decreases
        - Being praised -> Agreeableness increases, Trust increases
        - Conflict -> Stress increases, Neuroticism increases
        - Friendly interaction -> Trust increases, Energy increases
        """
        emotion = EmotionVector(
            openness=self.profile.emotion_vector.openness,
            conscientiousness=self.profile.emotion_vector.conscientiousness,
            extraversion=self.profile.emotion_vector.extraversion,
            agreeableness=self.profile.emotion_vector.agreeableness,
            neuroticism=self.profile.emotion_vector.neuroticism,
            trust=self.profile.emotion_vector.trust,
            stress=self.profile.emotion_vector.stress,
            energy=self.profile.emotion_vector.energy
        )
        
        # Merge dialogue and event text for analysis
        text = (last_dialogue + " " + current_event).lower()
        
        # Detect negative emotion keywords
        negative_keywords = [
            "stupid", "idiot", "wrong", "bad", "hate", "angry", "mad",
            "笨蛋", "白痴", "错误", "糟糕", "讨厌", "生气", "愤怒"
        ]
        positive_keywords = [
            "good", "great", "excellent", "love", "happy", "nice", "wonderful",
            "好", "棒", "优秀", "爱", "开心", "不错", "美好"
        ]
        conflict_keywords = [
            "fight", "argue", "disagree", "conflict", "against",
            "争吵", "争论", "不同意", "冲突", "反对"
        ]
        
        # Calculate keyword matching
        negative_count = sum(1 for kw in negative_keywords if kw in text)
        positive_count = sum(1 for kw in positive_keywords if kw in text)
        conflict_count = sum(1 for kw in conflict_keywords if kw in text)
        
        # Update emotions based on matching results
        if negative_count > 0:
            # Being criticized/insulted
            emotion.neuroticism += 0.1 * negative_count
            emotion.trust -= 0.1 * negative_count
            emotion.stress += 0.1 * negative_count
            emotion.agreeableness -= 0.05 * negative_count
        
        if positive_count > 0:
            # Being praised/friendly interaction
            emotion.agreeableness += 0.1 * positive_count
            emotion.trust += 0.1 * positive_count
            emotion.energy += 0.1 * positive_count
            emotion.stress -= 0.05 * positive_count
        
        if conflict_count > 0:
            # Conflict
            emotion.stress += 0.15 * conflict_count
            emotion.neuroticism += 0.1 * conflict_count
            emotion.trust -= 0.1 * conflict_count
        
        # If interaction type is provided, apply directly
        if interaction_type:
            if interaction_type == "criticism":
                emotion.neuroticism += 0.2
                emotion.trust -= 0.15
                emotion.stress += 0.2
            elif interaction_type == "praise":
                emotion.agreeableness += 0.2
                emotion.trust += 0.15
                emotion.energy += 0.15
            elif interaction_type == "conflict":
                emotion.stress += 0.25
                emotion.neuroticism += 0.15
                emotion.trust -= 0.15
            elif interaction_type == "friendly":
                emotion.trust += 0.15
                emotion.energy += 0.15
                emotion.agreeableness += 0.1
        
        # Generate rule-based thought process
        thought_parts = []
        if negative_count > 0:
            thought_parts.append(f"Detected {negative_count} negative emotion keywords, which may increase neuroticism and stress, reduce trust.")
        if positive_count > 0:
            thought_parts.append(f"Detected {positive_count} positive emotion keywords, which may increase agreeableness and trust, boost energy.")
        if conflict_count > 0:
            thought_parts.append(f"Detected {conflict_count} conflict keywords, which may increase stress and neuroticism, reduce trust.")
        if interaction_type:
            thought_parts.append(f"Interaction type is {interaction_type}, applying corresponding emotion change rules.")
        
        thought = " ".join(thought_parts) if thought_parts else "Using rule-based method to analyze emotional changes, no obvious emotional trigger words detected."
        
        return emotion, thought
    
    def _update_with_llm(self, last_dialogue: str, current_event: str) -> Tuple[EmotionVector, str]:
        """
        Use LLM to analyze emotional changes
        
        This method uses LLM to analyze the impact of dialogue and events on emotions
        Let LLM perform natural language reasoning (thought) first, then perform state changes
        """
        # Get current emotional state
        current_emotion = self.profile.emotion_vector
        
        # Build prompt - require LLM to provide thought process first, then state changes
        system_message = "You are an expert at analyzing emotional states. Analyze how interactions affect emotional dimensions. First, provide your reasoning in natural language, then provide the emotional state changes."
        
        prompt = f"""Analyze how the following dialogue and event affect the emotional state.

Current Emotional State:
- Openness: {current_emotion.openness:.2f}
- Conscientiousness: {current_emotion.conscientiousness:.2f}
- Extraversion: {current_emotion.extraversion:.2f}
- Agreeableness: {current_emotion.agreeableness:.2f}
- Neuroticism: {current_emotion.neuroticism:.2f}
- Trust: {current_emotion.trust:.2f}
- Stress: {current_emotion.stress:.2f}
- Energy: {current_emotion.energy:.2f}

Dialogue: {last_dialogue}
Event: {current_event}

Please analyze this step by step:

1. First, provide your reasoning (thought) in natural language about how this dialogue and event might affect the emotional state. Consider the context, tone, and implications.

2. Then, provide a JSON object containing the changes (deltas) for each dimension, e.g.:
{{
    "openness": 0.05,
    "conscientiousness": -0.02,
    "extraversion": 0.1,
    "agreeableness": -0.1,
    "neuroticism": 0.15,
    "trust": -0.1,
    "stress": 0.2,
    "energy": -0.05
}}

Please respond in the following format:

THOUGHT: [Your natural language reasoning here. Structure your thought as follows:]
- Observation Analysis: What did you observe from the dialogue and event? What is the tone, context, and key information?
- Self-Reflection: How does this make you feel? What are your internal reactions and emotional responses?
- State Update Plan: Based on your analysis, which emotional dimensions should change and why?

Example THOUGHT:
"Observation Analysis: The neighbor is dismissing the emotional value of the trees and focusing purely on utility and safety. The tone is dismissive and practical, showing little regard for emotional attachment.
Self-Reflection: This feels cold to me. I feel my values are being trivialized. My stress is rising because I fear he doesn't care about the heritage and emotional significance I attach to these trees.
State Update Plan: Increase stress (due to feeling dismissed), increase neuroticism (due to emotional sensitivity), and decrease trust (due to feeling misunderstood)."

CHANGES: {{
    "openness": ...,
    "conscientiousness": ...,
    "extraversion": ...,
    "agreeableness": ...,
    "neuroticism": ...,
    "trust": ...,
    "stress": ...,
    "energy": ...
}}

Values should be between -1.0 and 1.0. Provide the THOUGHT first, then the CHANGES JSON object."""

        try:
            # Call LLM
            if hasattr(self.llm_model, 'generate'):
                success, response = self.llm_model.generate(system_message, prompt)
                if not success:
                    # If LLM call fails, fall back to rule-based method
                    return self._update_with_rules(last_dialogue, current_event, None)
            else:
                # If LLM model doesn't have generate method, try direct call
                try:
                    response = self.llm_model(prompt)
                    success = True
                except Exception:
                    # If call fails, fall back to rule-based method
                    return self._update_with_rules(last_dialogue, current_event, None)
            
            if not success:
                return self._update_with_rules(last_dialogue, current_event, None)
            
            # Parse LLM response (extract thought and JSON)
            import json
            import re
            
            # Extract THOUGHT section
            thought = ""
            thought_match = re.search(r'THOUGHT:\s*(.+?)(?=CHANGES:|$)', response, re.DOTALL | re.IGNORECASE)
            if thought_match:
                thought = thought_match.group(1).strip()
            else:
                # If THOUGHT tag not found, try extracting all text before CHANGES as thought
                changes_match = re.search(r'CHANGES:', response, re.IGNORECASE)
                if changes_match:
                    thought = response[:changes_match.start()].strip()
                    # Remove possible tags
                    thought = re.sub(r'^(THOUGHT|thought|思考|推理):\s*', '', thought, flags=re.IGNORECASE)
                else:
                    thought = "LLM analyzed the impact of dialogue and events on emotions."
            
            # Try to extract JSON object
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    deltas = json.loads(json_str)
                except json.JSONDecodeError:
                    # If JSON parsing fails, fall back to rule-based method
                    return self._update_with_rules(last_dialogue, current_event, None)
            else:
                # If no JSON found, fall back to rule-based method
                return self._update_with_rules(last_dialogue, current_event, None)
            
            # Apply changes
            updated_emotion = EmotionVector(
                openness=max(0.0, min(1.0, current_emotion.openness + deltas.get("openness", 0.0))),
                conscientiousness=max(0.0, min(1.0, current_emotion.conscientiousness + deltas.get("conscientiousness", 0.0))),
                extraversion=max(0.0, min(1.0, current_emotion.extraversion + deltas.get("extraversion", 0.0))),
                agreeableness=max(0.0, min(1.0, current_emotion.agreeableness + deltas.get("agreeableness", 0.0))),
                neuroticism=max(0.0, min(1.0, current_emotion.neuroticism + deltas.get("neuroticism", 0.0))),
                trust=max(0.0, min(1.0, current_emotion.trust + deltas.get("trust", 0.0))),
                stress=max(0.0, min(1.0, current_emotion.stress + deltas.get("stress", 0.0))),
                energy=max(0.0, min(1.0, current_emotion.energy + deltas.get("energy", 0.0)))
            )
            
            return updated_emotion, thought
            
        except Exception as e:
            # If any error occurs, fall back to rule-based method
            import warnings
            warnings.warn(f"LLM-based emotion update failed: {e}. Falling back to rule-based method.")
            return self._update_with_rules(last_dialogue, current_event, None)
    
    def get_emotion_guidance(self) -> str:
        """
        Generate response style guidance based on current emotional state
        
        For example: If Neuroticism is high, response should be more defensive
        """
        emotion = self.profile.emotion_vector
        guidance = []
        
        if emotion.neuroticism > 0.7:
            guidance.append("Current emotion is quite sensitive, responses may be more defensive")
        elif emotion.neuroticism < 0.3:
            guidance.append("Current emotion is relatively stable, responses are more composed")
        
        if emotion.stress > 0.7:
            guidance.append("Current stress is high, responses may be shorter or more impatient")
        elif emotion.stress < 0.3:
            guidance.append("Current stress is low, responses are more relaxed and natural")
        
        if emotion.trust > 0.7:
            guidance.append("High trust in the current interlocutor, responses are more open and sincere")
        elif emotion.trust < 0.3:
            guidance.append("Low trust in the current interlocutor, responses are more cautious and reserved")
        
        if emotion.energy > 0.7:
            guidance.append("Current energy is high, responses are more positive and active")
        elif emotion.energy < 0.3:
            guidance.append("Current energy is low, responses may be shorter or tired")
        
        return "; ".join(guidance) if guidance else "Emotional state is normal, respond in your usual style"

