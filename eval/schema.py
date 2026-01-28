from dataclasses import dataclass, field
from dacite import from_dict
from typing import List, Optional
from pathlib import Path
import json
from templates import system_prompt, state_matrics_template, gpt_turn_template
import re
from archetype import big5_mapping
from copy import deepcopy
import math


class ParseError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

@dataclass
class Big5:
    openness: str
    conscientiousness: str
    extraversion: str
    agreeableness: str
    neuroticism: str
    label: Optional[str] = None

@dataclass
class StateMetrics:
    stress: float
    trust: float
    dominance: float
    focus: float

    def is_in_range(self, other: "StateMetrics", epsilon: float = 0.01) -> bool:
        ranges = {
            "low": (0.0, 0.3),
            "medium": (0.3, 0.6),
            "high": (0.6, 1.000001),
        }
        stress_range_lb, stress_range_ub = ranges[other.stress]
        trust_range_lb, trust_range_ub = ranges[other.trust]
        dominance_range_lb, dominance_range_ub = ranges[other.dominance]
        focus_range_lb, focus_range_ub = ranges[other.focus]
        return stress_range_lb <= self.stress <= stress_range_ub and \
            trust_range_lb <= self.trust <= trust_range_ub and \
                dominance_range_lb <= self.dominance <= dominance_range_ub and \
                    focus_range_lb <= self.focus <= focus_range_ub
    def is_equal(self, other: "StateMetrics", epsilon: float = 0.01) -> bool:
        return math.isclose(self.stress, other.stress, abs_tol=epsilon) and \
            math.isclose(self.trust, other.trust, abs_tol=epsilon) and \
                math.isclose(self.dominance, other.dominance, abs_tol=epsilon) and \
                    math.isclose(self.focus, other.focus, abs_tol=epsilon)
    @staticmethod
    def get_state_type(state: str) -> str | None:
        valid_states = ["low", "medium", "high"]
        for valid_state in valid_states:
            if valid_state in state:
                return valid_state
        return None
    def previous_state_from_value(self, value: str) -> dict:
        try:
            content = parse_bw_tags(value, "Previous Dialogue State")
            if content is None:
                regex = r'<Previous Dialogue State>(.*?)<Internal Monologue>'
                match = re.search(regex, value, re.S)
                if match:
                    content = match.group(1).strip()
            self.stress = content.split("Stress:")[1].split("Trust:")[0].strip()
            self.trust = content.split("Trust:")[1].split("Dominance:")[0].strip()
            self.dominance = content.split("Dominance:")[1].split("Focus:")[0].strip()
            self.focus = content.split("Focus:")[1].strip().strip()
            pattern = r'-?\d+\.?\d*'
            self.stress = re.search(pattern, self.stress)[0]
            self.trust = re.search(pattern, self.trust)[0]
            self.dominance = re.search(pattern, self.dominance)[0]
            self.focus = re.search(pattern, self.focus)[0]
            self.stress = float(self.stress)
            self.trust = float(self.trust)
            self.dominance = float(self.dominance)
            self.focus = float(self.focus)
        except Exception as e:
            raise ParseError(f"Error ({e}) parsing previous state from value: {value}")
    
    def current_state_from_value(self, value: str):
        try:
            state = parse_bw_tags(value, "Current State")
            self.stress = state.split("Stress:")[1].split("Trust:")[0]
            self.trust = state.split("Trust:")[1].split("Dominance:")[0]
            self.dominance = state.split("Dominance:")[1].split("Focus:")[0]
            self.focus = state.split("Focus:")[1].strip()
            self.stress = self.get_state_type(self.stress)
            self.trust = self.get_state_type(self.trust)
            self.dominance = self.get_state_type(self.dominance)
            self.focus = self.get_state_type(self.focus)
        except Exception as e:
            raise ParseError(f"Error ({e}) parsing current state from value: {value}")
    def to_value(self) -> str:
        return state_matrics_template.format(state_metrics=self)
    def reverse(self):
        self.stress = 1 - self.stress
        self.trust = 1 - self.trust
        self.dominance = 1 - self.dominance
        self.focus = 1 - self.focus

def _reverse_state_plan(value: str) -> str:
    if "increase" in value:
        return value.replace("increase", "decrease")
    elif "decrease" in value:
        return value.replace("decrease", "increase")
    else:
        return "stable"


@dataclass
class StatePlan:
    stress: str
    trust: str
    dominance: str
    focus: str

    @staticmethod
    def get_change_type(change: str) -> str:
        valid_changes = ["increase a little", "decrease a little", "stable", "increase a lot", "decrease a lot"]
        for valid_change in valid_changes:
            if valid_change in change:
                return valid_change
        raise ParseError(f"Error parsing state plan from value: {value}")

    def from_value(self, value: str):
        try:
            plan = parse_bw_tags(value, "State Update Plan")
            self.stress = plan.split("Stress:")[1].split("Trust:")[0]
            self.trust = plan.split("Trust:")[1].split("Dominance:")[0]
            self.dominance = plan.split("Dominance:")[1].split("Focus:")[0]
            self.focus = plan.split("Focus:")[1].strip()
            self.stress = self.get_change_type(self.stress)
            self.trust = self.get_change_type(self.trust)
            self.dominance = self.get_change_type(self.dominance)
            self.focus = self.get_change_type(self.focus)
        except Exception as e:
            raise ParseError(f"Error ({e}) parsing state plan from value: {value}")


    def reverse(self):
        self.stress = _reverse_state_plan(self.stress)
        self.trust = _reverse_state_plan(self.trust)
        self.dominance = _reverse_state_plan(self.dominance)
        self.focus = _reverse_state_plan(self.focus)


@dataclass
class TurnMetadata:
    turn_index: int
    state_float_prev: StateMetrics
    state_plan: StatePlan
    state_delta: StateMetrics
    state_float_post: StateMetrics
    def to_value(self) -> str:
        return 
    def reverse(self):
        self.state_plan.reverse()
        self.state_float_post.reverse()

@dataclass
class GlobalMetadata:
    archetype: str
    big5: Big5
    initial_state: StateMetrics


@dataclass
class ConversationTurn:
    role: str  
    value: str
    metadata: Optional[TurnMetadata] = None 

@dataclass
class ConversationData:
    conversation_id: str
    metadata: GlobalMetadata
    conversations: List[ConversationTurn]

@dataclass
class ConversationDataset:
    conversations: List[ConversationData]
    file_path: Path

@dataclass
class OneRoundResult:
    conversation_id: str
    predicted_response: str
    true_response: str
    error_message: Optional[str] = None

@dataclass
class OneRoundResults:
    dataset_path: str
    results: List[OneRoundResult]

@dataclass
class CharacterInfo:
    name: str
    occupation: Optional[str] = None
    personality: Optional[str] = None
    archetype: Optional[str] = None
    talking_style: Optional[str] = None

@dataclass
class SystemSetting:
    scenario: str
    topic: str
    your_character_info: CharacterInfo
    other_character_info: CharacterInfo
    def from_value(self, system_prompt: str) -> None:
        try:
            self.scenario = system_prompt.split("Scenario:")[1].split("Topic:")[0].strip()
            self.topic = system_prompt.split("Topic:")[1].split("Character Information:")[0].strip()
            your_character_info_name = system_prompt.split("Your name:")[1].split("\n")[0]
            your_character_info_occupation = system_prompt.split("Your occupation:")[1].split("\n")[0]
            your_character_info_personality = system_prompt.split("Your personality:")[1].split("\n")[0]
            your_character_info_archetype = system_prompt.split("Your archetype:")[1].split("Big Five:")[0].strip()
            your_character_info_talking_style = system_prompt.split("Your talking style:")[1].split("\n")[0]
            self.your_character_info = CharacterInfo(
                name=your_character_info_name,
                occupation=your_character_info_occupation,
                personality=your_character_info_personality,
                archetype=your_character_info_archetype,
                talking_style=your_character_info_talking_style
            )
        except Exception as e:
            raise Exception(f"Error parsing system setting: {e}")
        try:
            other_character_info_name = system_prompt.split("Other person's name:")[1].split("(use this name when")[0].strip()
            self.other_character_info = CharacterInfo(
                name=other_character_info_name,
                occupation=None,
                personality=None,
                archetype=None,
                talking_style=None
            )
        except Exception as e:
            raise Exception(f"Error parsing system setting: {e}")

    def to_value(self, switch_agent: bool = False, applied_template: str = system_prompt) -> str:
        if switch_agent:
            big5=Big5(**big5_mapping[self.other_character_info.archetype])
            switch_setting = deepcopy(self)
            switch_setting.your_character_info = self.other_character_info
            switch_setting.other_character_info = self.your_character_info
            return applied_template.format(system_setting=switch_setting, big5=big5)
        else:
            big5=Big5(**big5_mapping[self.your_character_info.archetype])
            return applied_template.format(system_setting=self, big5=big5)

@dataclass
class SystemSettingData:
    system_setting: SystemSetting
    system_setting_id: str

@dataclass
class SystemSettingDataset:
    system_settings: List[SystemSettingData]
    file_path: Path

@dataclass
class DialogResult:
    system_setting_id: str
    system_setting: SystemSetting
    dialog: List[ConversationTurn]
    parsed_dialog: Optional[List[ConversationTurn]] = None
    error_message: Optional[str] = None

def apply_plan(x, action, delta, Delta):
    if action == "increase a lot":
        return x + Delta * (1 - x)
    if action == "increase a little":
        return x + delta * (1 - x)
    if action == "decrease a lot":
        return x - Delta * x
    if action == "decrease a little":
        return x - delta * x
    return x  # stable

def clip01(x):
    return max(0.0, min(1.0, x))

def update_state(state: StateMetrics, plan: StatePlan,
                 delta=0.06, Delta=0.18,
                 delta_T=0.03, Delta_T=0.10,   # trust更慢（推荐）
                 eta=0.7, k_sf=0.10) -> StateMetrics:
    s, t, d, f = state.stress, state.trust, state.dominance, state.focus

    # Step 1: plan -> candidate (with trust-buffer for stress increases)
    ds, Ds = delta, Delta
    if plan.stress.startswith("increase"):
        ds = delta * (1 - 0.5 * t)
        Ds = Delta * (1 - 0.5 * t)

    s_plan = apply_plan(s, plan.stress, ds, Ds)
    t_plan = apply_plan(t, plan.trust,  delta_T, Delta_T)
    d_plan = apply_plan(d, plan.dominance, delta,  Delta)
    f_plan = apply_plan(f, plan.focus,     delta,  Delta)

    # Step 2: EMA smoothing
    s2 = (1-eta)*s + eta*s_plan
    t2 = (1-eta)*t + eta*t_plan
    d2 = (1-eta)*d + eta*d_plan
    f2 = (1-eta)*f + eta*f_plan

    # Step 3: coupling A (stress eats focus)
    f2 = f2 - k_sf * max(0.0, s2 - 0.6)

    return StateMetrics(
        stress=clip01(s2),
        trust=clip01(t2),
        dominance=clip01(d2),
        focus=clip01(f2)
    )

def parse_turn_metadata_from_response(response: str, last_turn_metadata: TurnMetadata) -> TurnMetadata:
    update_state_plan = StatePlan("", "", "", "")
    update_state_plan.from_value(response)
    # update_state_plan = StatePlan(stress=stress, trust=trust, dominance=dominance, focus=focus)
    state_float_post = update_state(last_turn_metadata.state_float_post, update_state_plan)
    state_delta = StateMetrics(stress=state_float_post.stress - last_turn_metadata.state_float_post.stress, 
                               trust=state_float_post.trust - last_turn_metadata.state_float_post.trust, 
                               dominance=state_float_post.dominance - last_turn_metadata.state_float_post.dominance,
                               focus=state_float_post.focus - last_turn_metadata.state_float_post.focus)
    metadata = TurnMetadata(
        turn_index=last_turn_metadata.turn_index + 1,
        state_float_prev=last_turn_metadata.state_float_post,
        state_plan=update_state_plan,
        state_delta=state_delta,
        state_float_post=state_float_post
    )
    return metadata

def parse_bw_tags(response: str, tag: str) -> str | None:
    if tag == "Response":
        # regex = r"<Response>(.*?)(?:</Response>|</assistant_end>|\ufffd|$)"
        allowed_chars = r"[\x00-\x7f\u2000-\u206f\u3000-\u303f\uff00-\uffef]"
        regex = rf"<Response>((?:(?!</Response>|</assistant_end>){allowed_chars})*)"
        match = re.search(regex, response, re.S)
        # if match:
        #     return match.group(1).strip()
        # else:
        #     regex = r"<Response>(.*?)(?:</assistant_end>|\ufffd|$)"
        # match = re.search(regex, response, re.S)
        if match:
            return match.group(1).strip()
        else:
            return None
    else:
        regex = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(regex, response, re.S)
        if match:
            return match.group(1).strip()
        else:
            return None


def build_dataset(file: Path) -> ConversationDataset:
    dataset = ConversationDataset(conversations=[], file_path=file)
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for dct in data:
        dct = {
            "conversation_id": dct["id"], 
            "metadata": dct["metadata"], 
            "conversations": dct["conversations"]
            }
        dataset.conversations.append(from_dict(ConversationData, dct))
    return dataset

