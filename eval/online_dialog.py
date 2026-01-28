from generate import call_llm, Config
from schema import ConversationTurn, SystemSetting, CharacterInfo, parse_bw_tags, Big5
from archetype import big5_mapping
from templates import system_prompt
from copy import deepcopy
import json


config = Config(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model_name="amour",
    temperature=1,
    max_tokens=2048,
    top_p=0.8
)


def main():
    system_setting = SystemSetting(
        scenario="Two college students meet at a cafe and start a conversation.",
        topic="Recent plans",
        other_character_info=CharacterInfo(
            name="Alex",
            occupation="college student",
            personality="lazy, offensive, but open to correction",
            archetype="A4_Technocrat",
            talking_style="Direct and unvarnished",
        ),
        your_character_info=CharacterInfo(
            name="Amour",
            occupation="college student",
            personality="enthusiastic, outgoing but sometimes aggressive",
            archetype="D3_Confrontational",
            talking_style="Cynical and snarky. Such as \"I will help you do that, then you will have so much work to do!\"",
        )
    )
    your_big5 = Big5(**big5_mapping[system_setting.your_character_info.archetype])
    system_value = system_prompt.format(system_setting=system_setting, big5=your_big5)
    history = [{"role": "system", "content": system_value}]
    parsed_history = deepcopy(history)
    for i in range(12):
        print(f"Turn {i+1}:")
        user_message = input("Enter your message: ")
        message = {"role": "user", "content": user_message}
        parsed_history.append(message)
        history.append(message)
        response = call_llm(history, config)
        parsed_response = parse_bw_tags(response, "Response")
        print(parsed_response)
        history.append({"role": "gpt", "content": response})
        parsed_history.append({"role": "gpt", "content": parsed_response})
    print(parsed_history)
    with open("history.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    main()