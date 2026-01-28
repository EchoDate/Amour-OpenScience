import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from templates import system_prompt


@dataclass
class State:
    full_text: str
    emotion: Optional[Dict[str, Any]] = None
    relation: Optional[Dict[str, Any]] = None
    def __str__(self):
        return self.full_text

@dataclass
class InternalMonologue:
    full_text: str
    observation_analysis: Optional[str] = None
    self_reflection: Optional[str] = None
    state_update_plan: Optional[str] = None
    def __str__(self):
        return self.full_text

@dataclass
class Turn:
    role: str
    message: str
    full_text: Optional[str] = None
    state: Optional[State] = None
    internal_monologue: Optional[InternalMonologue] = None

@dataclass
class CharacterInfo:
    name: str
    occupation: Optional[str] = None
    personality: Optional[str] = None
    talking_style: Optional[str] = None

@dataclass
class SystemSetting:
    scenario: str
    topic: str
    your_character_info: CharacterInfo
    other_character_info: CharacterInfo
    history_turns: List[Turn]=None

@dataclass
class Conversation:
    id: str
    system_setting: SystemSetting
    turns: List[Turn]

@dataclass
class ConversationDataset:
    name: str
    conversations: List[Conversation]

def load_dict_from_json(file: Path, remove_last_turn: bool = False) -> ConversationDataset:
    dataset = ConversationDataset(name=file.stem, conversations=[])
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for dct in data:
        sys_dct = dct["system_setting"]
        your_character_info = CharacterInfo(name=sys_dct["your_character_info"]["name"], occupation=sys_dct["your_character_info"].get("occupation", None), personality=sys_dct["your_character_info"].get("personality", None), talking_style=sys_dct["your_character_info"].get("talking_style", None))
        other_character_info = CharacterInfo(name=sys_dct["other_character_info"]["name"], occupation=sys_dct["other_character_info"].get("occupation", None), personality=sys_dct["other_character_info"].get("personality", None), talking_style=sys_dct["other_character_info"].get("talking_style", None))
        history_turns = []
        for turn in sys_dct["history_turns"]:
            history_turns.append(Turn(role=turn["role"], message=turn["message"], full_text=turn["full_text"], state=turn.get("state", None), internal_monologue=turn.get("internal_monologue", None)))
        sys = SystemSetting(scenario=sys_dct["scenario"], topic=sys_dct["topic"], your_character_info=your_character_info, other_character_info=other_character_info, history_turns=history_turns)
        turns = []
        for turn in dct["turns"]:
            turns.append(Turn(role=turn["role"], message=turn["message"], full_text=turn["full_text"], state=turn.get("state", None), internal_monologue=turn.get("internal_monologue", None)))
        if remove_last_turn:
            turns.pop()
        conv = Conversation(id=dct["id"], system_setting=sys, turns=turns)
        dataset.conversations.append(conv)
    return dataset

def parse_bw_tags(gpt_value: str, tag: str) -> str:
    match = re.search(fr'<{tag}>([\s\S]*?)</{tag}>', gpt_value, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_gpt_turn_from_response(response: str) -> Turn:
    state = parse_bw_tags(response, "Current State")
    internal_monologue = parse_bw_tags(response, "Internal Monologue")
    response_text = parse_bw_tags(response, "Response")
    if state is not None:
        state = State(full_text=state)
    if internal_monologue is not None:
        internal_monologue = InternalMonologue(full_text=internal_monologue)
    return Turn(role="gpt", message=response_text, state=state, 
         internal_monologue=internal_monologue, full_text=response)

def parse_system_prompt(system_value: str) -> SystemSetting:
    result = SystemSetting(scenario=None, topic=None, your_character_info=None, other_character_info=None)
    
    scenario_match = re.search(r'Scenario:\s*(.*?)(?:\nTopic:|$)', system_value, re.DOTALL)
    if scenario_match:
        result.scenario = scenario_match.group(1).strip()
    
    topic_match = re.search(r'Topic:\s*(.+?)(?:\n\nCharacter Information:|$)', system_value, re.DOTALL)
    if topic_match:
        result.topic = topic_match.group(1).strip()
    
    other_character_info = CharacterInfo(name=None, occupation=None, personality=None, talking_style=None)
    your_character_info = CharacterInfo(name=None, occupation=None, personality=None, talking_style=None)
    other_name_match = re.search(r'Other person\'s name:\s*(.+?)(?:\n|$)', system_value)
    if other_name_match:
        other_character_info.name = other_name_match.group(1).strip()
    
    your_name_match = re.search(r'Your name:\s*(.+?)(?:\n|$)', system_value)
    if your_name_match:
        your_character_info.name = your_name_match.group(1).strip()
    
    occupation_match = re.search(r'Your occupation:\s*(.+?)(?:\n|$)', system_value)
    if occupation_match:
        your_character_info.occupation = occupation_match.group(1).strip()
    
    personality_match = re.search(r'Your personality:\s*(.+?)(?:\n- Your talking style:|$)', system_value, re.DOTALL)
    if personality_match:
        your_character_info.personality = personality_match.group(1).strip()
    
    talking_style_match = re.search(r'Your talking style:\s*(.+?)(?:\n\nBased on|$)', system_value, re.DOTALL)
    if talking_style_match:
        your_character_info.talking_style = talking_style_match.group(1).strip()
    
    result.other_character_info = other_character_info
    result.your_character_info = your_character_info
    return result

