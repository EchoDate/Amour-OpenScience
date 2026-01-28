"""
Modeling for Profile.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json


# Predefined relationship type list
RELATION_TYPES = [
    "ex",              # ex-boyfriend/ex-girlfriend
    "teacher_student", # teacher-student
    "lover",           # lover/romantic partner
    "friend",          # friend
    "colleague",       # colleague
    "family",          # family member
    "classmate",       # classmate
    "partner",         # business partner
    "rival",           # rival/competitor
    "stranger",        # stranger
]


@dataclass
class StaticTraits:
    """T_static"""
    name: str = ""
    occupation: str = ""
    core_values: List[str] = field(default_factory=list)
    gender: str = ""
    age: int = 0
    interests: List[str] = field(default_factory=list)
    mbti: str = ""
    personality: str = ""
    talking_style: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "occupation": self.occupation,
            "core_values": self.core_values,
            "gender": self.gender,
            "age": self.age,
            "interests": self.interests,
            "mbti": self.mbti,
            "personality": self.personality,
            "talking_style": self.talking_style
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StaticTraits':
        return cls(
            name=data.get("name", ""),
            occupation=data.get("occupation", ""),
            core_values=data.get("core_values", []),
            gender=data.get("gender", ""),
            age=data.get("age", 0),
            interests=data.get("interests", []),
            mbti=data.get("mbti", ""),
            personality=data.get("personality", ""),
            talking_style=data.get("talking_style", "")
        )


@dataclass
class EmotionVector:
    """V_emotion: Dynamic state vector"""
    # Big Five personality dimensions
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    
    # Emotional state dimensions (optional)
    trust: float = 0.5
    stress: float = 0.5
    energy: float = 0.5
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
            "trust": self.trust,
            "stress": self.stress,
            "energy": self.energy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'EmotionVector':
        return cls(
            openness=data.get("openness", 0.5),
            conscientiousness=data.get("conscientiousness", 0.5),
            extraversion=data.get("extraversion", 0.5),
            agreeableness=data.get("agreeableness", 0.5),
            neuroticism=data.get("neuroticism", 0.5),
            trust=data.get("trust", 0.5),
            stress=data.get("stress", 0.5),
            energy=data.get("energy", 0.5)
        )
    
    def clamp(self):
        """clamp the values to [0, 1]"""
        self.openness = max(0.0, min(1.0, self.openness))
        self.conscientiousness = max(0.0, min(1.0, self.conscientiousness))
        self.extraversion = max(0.0, min(1.0, self.extraversion))
        self.agreeableness = max(0.0, min(1.0, self.agreeableness))
        self.neuroticism = max(0.0, min(1.0, self.neuroticism))
        self.trust = max(0.0, min(1.0, self.trust))
        self.stress = max(0.0, min(1.0, self.stress))
        self.energy = max(0.0, min(1.0, self.energy))


@dataclass
class Relation:
    """Relation"""
    intimacy: float = 0.5  # Intimacy [-1, 1]
    trust: float = 0.5      # Trust [0, 1]
    dominance: float = 0.0  # Dominance [-1, 1], positive means I dominate, negative means the other party dominates
    relation_type: Optional[str] = None  # Relationship type (static, won't change during conversation), such as "ex", "teacher_student", "lover", etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "intimacy": self.intimacy,
            "trust": self.trust,
            "dominance": self.dominance
        }
        if self.relation_type is not None:
            result["relation_type"] = self.relation_type
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relation':
        """Create from dictionary"""
        return cls(
            intimacy=data.get("intimacy", 0.5),
            trust=data.get("trust", 0.5),
            dominance=data.get("dominance", 0.0),
            relation_type=data.get("relation_type", None)
        )
    
    def set_relation_type(self, relation_type: Optional[str]):
        """
        Set relationship type
        
        Args:
            relation_type: Relationship type (such as "ex", "teacher_student", "lover", etc.), or None to clear
                          If the provided relationship type is not in RELATION_TYPES, it will still be set but a warning will be issued
        """
        if relation_type is not None and relation_type not in RELATION_TYPES:
            import warnings
            warnings.warn(
                f"Relationship type '{relation_type}' is not in predefined list {RELATION_TYPES}, but will still be set."
            )
        self.relation_type = relation_type
    
    def clamp(self):
        """clamp the values to [-1, 1]"""
        self.intimacy = max(-1.0, min(1.0, self.intimacy))
        self.trust = max(0.0, min(1.0, self.trust))
        self.dominance = max(-1.0, min(1.0, self.dominance))


class Profile:
    """
    Profile = (T_static, V_emotion, Relation)
    
    """
    
    def __init__(self, 
                 static_traits: Optional[StaticTraits] = None,
                 emotion_vector: Optional[EmotionVector] = None,
                 relations: Optional[Dict[str, Relation]] = None,
                 inner_profile: Optional[Dict[str, Any]] = None,
                 relational_profile: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize Profile
        
        Args:
            static_traits: Static traits
            emotion_vector: Emotion vector
            relations: Relationship map {agent_id: Relation}
            inner_profile: Inner Profile (self layer)
            relational_profile: Relational layer Profile
        """
        self.static_traits = static_traits or StaticTraits()
        self.emotion_vector = emotion_vector or EmotionVector()
        self.relations = relations or {}  # {agent_id: Relation}
        self.inner_profile = inner_profile or {}
        self.relational_profile = relational_profile or {}
    
    def get_relation(self, agent_id: str) -> Relation:
        """Get relationship with specified Agent, create default relationship if not exists"""
        if agent_id not in self.relations:
            self.relations[agent_id] = Relation()
        return self.relations[agent_id]
    
    def update_relation(self, agent_id: str, relation: Relation):
        """Update relationship with specified Agent"""
        self.relations[agent_id] = relation
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for serialization)"""
        return {
            "static_traits": self.static_traits.to_dict(),
            "emotion_vector": self.emotion_vector.to_dict(),
            "relations": {
                agent_id: rel.to_dict() 
                for agent_id, rel in self.relations.items()
            },
            "inner_profile": self.inner_profile,
            "relational_profile": self.relational_profile
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Profile':
        """Create from dictionary (for deserialization)"""
        static_traits = StaticTraits.from_dict(data.get("static_traits", {}))
        emotion_vector = EmotionVector.from_dict(data.get("emotion_vector", {}))
        relations = {
            agent_id: Relation.from_dict(rel_data)
            for agent_id, rel_data in data.get("relations", {}).items()
        }
        return cls(
            static_traits=static_traits,
            emotion_vector=emotion_vector,
            relations=relations,
            inner_profile=data.get("inner_profile", {}),
            relational_profile=data.get("relational_profile", {})
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Profile':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_summary(self) -> str:
        """Get text summary of Profile (for prompt)"""
        summary = f"Name: {self.static_traits.name}\n"
        summary += f"Occupation: {self.static_traits.occupation}\n"
        summary += f"Age: {self.static_traits.age}\n"
        summary += f"Personality: {self.static_traits.personality}\n"
        summary += f"Talking Style: {self.static_traits.talking_style}\n"
        summary += f"Current Emotional State:\n"
        summary += f"  - Openness: {self.emotion_vector.openness:.2f}\n"
        summary += f"  - Extraversion: {self.emotion_vector.extraversion:.2f}\n"
        summary += f"  - Neuroticism: {self.emotion_vector.neuroticism:.2f}\n"
        summary += f"  - Stress: {self.emotion_vector.stress:.2f}\n"
        return summary

