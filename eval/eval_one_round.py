from generate import generate_one_round_for_dataset, Config
from schema import ConversationDataset, build_dataset
from pathlib import Path
import argparse
from dataclasses import asdict
import json

def main(config: Config, json_file: Path, output_file: Path):
    dataset = build_dataset(json_file)
    results = generate_one_round_for_dataset(config, dataset)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(asdict(results), f, indent=2)

def run_one_conversation(config: Config, json_file: Path, output_file: Path, id: int):
    dataset = build_dataset(json_file)
    dataset.conversations = [dataset.conversations[id]]
    results = generate_one_round_for_dataset(config, dataset)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(asdict(results), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model_name", type=str, default="amour")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--json_file", type=str, default="../data/test_annotation_full_v1_amour.json")
    parser.add_argument("--output_file", type=str, default="./results/one_round_test.json")
    parser.add_argument("--human_role", type=str, default="human")
    parser.add_argument("--gpt_role", type=str, default="gpt")
    args = parser.parse_args()
    config = Config(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        human_role=args.human_role,
        gpt_role=args.gpt_role
    )
    main(config, Path(args.json_file), Path(args.output_file))
    # run_one_conversation(config, Path(args.json_file), Path(args.output_file), 0)