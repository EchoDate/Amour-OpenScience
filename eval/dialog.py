
import json
from typing import List, Optional
from pathlib import Path
from schema import ConversationTurn, parse_turn_metadata_from_response, CharacterInfo, parse_bw_tags, StateMetrics, TurnMetadata, GlobalMetadata, Big5, TurnMetadata
from schema import SystemSettingData, SystemSettingDataset, DialogResult, SystemSetting
from generate import Config, call_llm
from templates import generate_first_user_message_prompt, generate_first_user_message_system_prompt, system_prompt_for_dialog
from archetype import initial_state_mapping, big5_mapping
import argparse
from dataclasses import asdict
from dacite import from_dict
from copy import deepcopy
from tqdm import tqdm
import random


class DialogError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class DialogSystem:
    
    def __init__(
        self,
        system_setting: SystemSetting,
        gemini_config: Config,
        baseline_config: Config,
        max_turns: int = 10
    ):
    # system_setting is described from amour's perspective, your_character_info is amour's, other_character_info is gemini's
        self.system_setting = system_setting
        self.gemini_config = gemini_config
        self.baseline_config = baseline_config
        self.max_turns = max_turns
        other_archetype = self.system_setting.other_character_info.archetype
        your_archetype = self.system_setting.your_character_info.archetype
        if other_archetype is None or your_archetype is None:
            raise Exception("Archetype is not set")
        self._gemini_global_metadata = GlobalMetadata(
            archetype=other_archetype,
            big5=Big5(**big5_mapping[other_archetype]),
            initial_state=StateMetrics(**initial_state_mapping[other_archetype])
        )
        self._baseline_global_metadata = GlobalMetadata(
            archetype=your_archetype,
            big5=Big5(**big5_mapping[your_archetype]),
            initial_state=StateMetrics(**initial_state_mapping[your_archetype])
        )
        self._gemini_system_prompt = self.system_setting.to_value(switch_agent=True, applied_template=system_prompt_for_dialog)
        self._baseline_system_prompt = self.system_setting.to_value(switch_agent=False, applied_template=system_prompt_for_dialog)
        self.messages: List[ConversationTurn] = [] # Store turns in conversation
    
    def _build_messages(self, for_whom: ["gemini", "baseline"]) -> List[ConversationTurn]:
        human_role = "gemini" if for_whom == "baseline" else "baseline"
        # gpt_role = "baseline" if for_whom == "baseline" else "gemini"
        
        messages = []
        system_value = self._gemini_system_prompt if for_whom == "gemini" else self._baseline_system_prompt
        messages.append(ConversationTurn(role="system", value=system_value, metadata=None))
        for turn in self.messages:
            new_turn = deepcopy(turn)
            if new_turn.role == human_role:
                new_turn.role = "user"
                response = parse_bw_tags(new_turn.value, "Response")
                if response is None:
                    raise DialogError(f"Response is not found in the turn: {new_turn.value}")
                new_turn.value = response
            else:
                new_turn.role = "assistant"
            messages.append(new_turn)
        return messages

    def _parse_turn(self, response: str, for_whom: ["gemini", "baseline"]) -> ConversationTurn:
        if len(self.messages) == 1:
            if for_whom == "gemini":
                raise Exception("First turn should be from gemini")
            last_turn_metadata = TurnMetadata(
                turn_index=0,
                state_float_prev=None,
                state_plan=None,
                state_delta=None,
                state_float_post=self._baseline_global_metadata.initial_state
            )
        else:
            last_turn_metadata = self.messages[-2].metadata
        metadata = parse_turn_metadata_from_response(response, last_turn_metadata)
        turn = ConversationTurn(role=for_whom, value=response, metadata=metadata)
        return turn
    
    def run_dialog(self, initial_message: Optional[str] = None) -> List[ConversationTurn]:
        # Initialize conversation history
        if initial_message is None:
            messages = [
                {"role": "system", "content": generate_first_user_message_system_prompt},
                {"role": "user", "content": generate_first_user_message_prompt.format(system_setting=self.system_setting, 
                your_big5=self._gemini_global_metadata.big5, others_big5=self._baseline_global_metadata.big5)}
            ]
            first_message =call_llm(
                messages=messages,
                config=self.gemini_config
            )
        else:
            first_message = initial_message

        first_turn_metadata = TurnMetadata(
            turn_index=1,
            state_float_prev=None,
            state_plan=None,
            state_delta=None,
            state_float_post=self._gemini_global_metadata.initial_state
        )   
        first_turn = ConversationTurn(role="gemini", value=f"<Response>{first_message}</Response>", metadata=first_turn_metadata)
        self.messages.append(first_turn)
        
        current_role = "baseline"
        call_gemini = False
        
        for turn_num in range(self.max_turns):
            current_role = "gemini" if call_gemini else "baseline"
            try:
                messages = self._build_messages(current_role)
            except DialogError as e:
                self.messages.append(ConversationTurn(role=current_role, value=e.message, metadata=None))
                break
            response = None
            try:
                response = call_llm(messages, self.gemini_config if call_gemini else self.baseline_config)
                new_turn = self._parse_turn(response, current_role)
                self.messages.append(new_turn)
                call_gemini = not call_gemini
            except Exception as e:
                raise DialogError(f"Error: {e}, turn number: {turn_num}, response: {response}, messages: {[asdict(turn) for turn in self.messages]}")
        
        return self.messages
    
    def save_to_file(self, output_path: Path) -> None:
        """Save dialog system state to file"""
        data = {
            "system_setting": asdict(self.system_setting),
            "messages": [asdict(turn) for turn in self.messages]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def parse_dialog(self) -> List[ConversationTurn]:
        parsed_dialog = []
        for turn in self.messages:
            parsed_value = parse_bw_tags(turn.value, "Response")
            if parsed_value is None:
                raise DialogError(f"Response is not found in the turn: {turn.value}")
            role = self.gemini_config.model_name if turn.role == "gemini" else self.baseline_config.model_name
            parsed_dialog.append(ConversationTurn(role=role, value=parsed_value, metadata=None))

        return parsed_dialog
    

def build_system_setting_dataset(file_path: Path) -> SystemSettingDataset:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    system_settings = []
    for dct in data:
        system_setting_data = from_dict(SystemSettingData, dct)
        system_settings.append(system_setting_data)
    return SystemSettingDataset(system_settings=system_settings, file_path=file_path)

def main(args: argparse.Namespace):
    amour_config = Config(
        api_key=args.baseline_api_key,
        base_url=args.baseline_base_url,
        model_name=args.baseline_model_name,
        temperature=args.baseline_temperature,
        max_tokens=args.baseline_max_tokens,
        top_p=args.baseline_top_p,
        human_role=args.baseline_human_role,
        gpt_role=args.baseline_gpt_role
    )
    gemini_config = Config(
        api_key=args.gemini_api_key,
        base_url=args.gemini_base_url,
        model_name=args.gemini_model_name,
        temperature=args.gemini_temperature,
        max_tokens=args.gemini_max_tokens,
        top_p=args.gemini_top_p,
        human_role=args.gemini_human_role,
        gpt_role=args.gemini_gpt_role
    )
    dataset = build_system_setting_dataset(Path(args.system_settings_file))
    results = {
        "dataset_path": args.system_settings_file,
        "results": []
    }
    dialog_results = []
    for system_setting_data in tqdm(dataset.system_settings):
        max_turns = random.randint(5, 10)
        dialog = DialogSystem(
            system_setting=system_setting_data.system_setting,
            gemini_config=gemini_config,
            baseline_config=amour_config,
            max_turns=max_turns
        )
        error_message = None
        parsed_messages = None
        try:
            dialog.run_dialog()
            parsed_messages = dialog.parse_dialog()
        except DialogError as e:
            print(e.message)
            error_message = e.message
        dialog_result = DialogResult(
            system_setting_id=system_setting_data.system_setting_id,
            system_setting=system_setting_data.system_setting,
            dialog=dialog.messages,
            parsed_dialog=parsed_messages,
            error_message=error_message
        )
        dialog_results.append(dialog_result)
        # dialog.run_dialog("Hey, Yang Chaoyue, I heard you’ve been getting a lot of attention lately for your straightforward comments about relationships. What’s your take on how people perceive you?")
        # dialog.save_to_file(Path(args.output_path))
    results["results"] = [asdict(result) for result in dialog_results]
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_api_key", type=str, required=True)
    parser.add_argument("--baseline_base_url", type=str, required=True)
    parser.add_argument("--baseline_model_name", type=str, required=True)
    parser.add_argument("--baseline_temperature", type=float, required=True)
    parser.add_argument("--baseline_max_tokens", type=int, required=True)
    parser.add_argument("--baseline_top_p", type=float, required=True)
    parser.add_argument("--baseline_human_role", type=str, required=True)
    parser.add_argument("--baseline_gpt_role", type=str, required=True)
    parser.add_argument("--gemini_api_key", type=str, required=True)
    parser.add_argument("--gemini_base_url", type=str, required=True)
    parser.add_argument("--gemini_model_name", type=str, required=True)
    parser.add_argument("--gemini_temperature", type=float, required=True)
    parser.add_argument("--gemini_max_tokens", type=int, required=True)
    parser.add_argument("--gemini_top_p", type=float, required=True)
    parser.add_argument("--gemini_human_role", type=str, default="user")
    parser.add_argument("--gemini_gpt_role", type=str, default="assistant")
    parser.add_argument("--system_settings_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
