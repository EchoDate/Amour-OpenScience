from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import evaluate  # type: ignore
from schema import parse_bw_tags, SystemSetting, Big5, build_dataset
from archetype import big5_mapping
from templates import llm_judge_one_round_prompt
from generate import call_llm, Config
from tqdm import tqdm
import os
# Display download progress
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"


def load_preds_refs(json_path: Path | str, skip_errors: bool = True) -> Tuple[List[str], List[str], List[str], int, int]:
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    used_ids: List[str] = []
    preds: List[str] = []
    refs: List[str] = []
    skipped = 0

    for item in results:
        conv_id = item.get("conversation_id", "")
        err = item.get("error_message", None)
        pred = parse_bw_tags(item.get("predicted_response", ""), "Response")
        ref = parse_bw_tags(item.get("true_response", ""), "Response")

        if skip_errors and err:
            skipped += 1
            continue
        if pred is None or ref is None:
            skipped += 1
            continue

        used_ids.append(conv_id)
        preds.append(pred)
        refs.append(ref)

    return used_ids, preds, refs, len(results), skipped

def metrics_for_one_round_results_json(
    refs: List[str],
    preds: List[str],
    use_stemmer: bool = True,
    use_aggregator: bool = True,
) -> Dict[str, Any]:
    rouge = evaluate.load("rouge")
    belu = evaluate.load("sacrebleu")
    bertscore = evaluate.load("bertscore")
    return {
        "rouge": rouge.compute(predictions=preds, references=refs, use_stemmer=use_stemmer, use_aggregator=use_aggregator)["rougeL"],
        "bleu": belu.compute(predictions=preds, references=refs, smooth_method="floor")["bleu"],
        "bertscore": bertscore.compute(predictions=preds, references=refs)["f1"],
    }

def load_two_results_json(json_path_1: str | Path, json_path_2: str | Path) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], int, int, int, int]:
    used_ids_1, preds_1, refs_1, count_total_1, skipped_1 = load_preds_refs(json_path_1)
    used_ids_2, preds_2, refs_2, count_total_2, skipped_2 = load_preds_refs(json_path_2)
    return used_ids_1, used_ids_2, preds_1, preds_2, refs_1, refs_2, count_total_1, count_total_2, skipped_1, skipped_2

def rouge_for_one_round_results_json(
    json_path: str | Path,
    skip_errors: bool = True,
    include_per_item: bool = False,
    rouge_type: str = "rougeL",
    use_stemmer: bool = True,
    use_aggregator: bool = False,
) -> Dict[str, Any]:
    # Add progress hints to avoid hanging during download
    
    print("Loading ROUGE evaluation metric (first-time use requires downloading from HuggingFace, may take a few minutes)...")
    print("Tip: If download is slow, you can:")
    print("  1. Set HuggingFace mirror: export HF_ENDPOINT=https://hf-mirror.com")
    print("  2. Check network connection")
    try:
        rouge = evaluate.load("rouge")
    except Exception as e:
        print(f"Error loading ROUGE metric: {e}")
        print("Trying to use local cache...")
        try:
            rouge = evaluate.load("rouge", trust_remote_code=True)
        except Exception as e2:
            print(f"Local cache also failed: {e2}")
            print("Please check network connection or manually download ROUGE metric")
            raise
    print("✓ ROUGE metric loaded successfully")
    if rouge_type not in {"rouge1", "rouge2", "rougeL", "rougeLsum"}:
        raise ValueError("rouge_type only supports: rouge1/rouge2/rougeL/rougeLsum")

    used_ids, preds, refs, count_total, skipped = load_preds_refs(json_path, skip_errors=skip_errors)
    path = Path(json_path)

    per_item_scores = rouge.compute(
        predictions=preds,
        references=refs,
        use_stemmer=use_stemmer,
        use_aggregator=use_aggregator,
    )
    if use_aggregator:
        avg_score = per_item_scores[rouge_type]
    else:
        raise NotImplementedError("use_aggregator must be True")

    out2: Dict[str, Any] = {
        "file": str(path),
        "count_total": count_total,
        "count_used": len(preds),
        "count_skipped": skipped,
        "rouge_type": rouge_type,
        "use_stemmer": use_stemmer,
        "use_aggregator": use_aggregator,
        "avg_score": avg_score,
    }
    return out2


def bleu_for_one_round_results_json(
    json_path: str | Path,
    *,
    skip_errors: bool = True,
) -> Dict[str, Any]:
    """Use HuggingFace evaluate's BLEU (based on sacrebleu) for overall evaluation."""
    used_ids, preds, refs, count_total, skipped = _load_preds_refs(
        json_path, skip_errors=skip_errors
    )
    path = Path(json_path)

    if not preds:
        return {
            "file": str(path),
            "count_total": count_total,
            "count_used": 0,
            "count_skipped": skipped,
            "bleu": 0.0,
            "precisions": [],
            "brevity_penalty": 0.0,
            "length_ratio": 0.0,
            "translation_length": 0,
            "reference_length": 0,
        }

    # evaluate's BLEU doesn't support per-sample use_aggregator=False, returns overall score here.
    bleu_metric = evaluate.load("bleu")
    score = bleu_metric.compute(predictions=preds, references=refs)
    return {
        "file": str(path),
        "count_total": count_total,
        "count_used": len(preds),
        "count_skipped": skipped,
        **score,
    }


def bertscore_for_one_round_results_json(
    json_path: str | Path,
    *,
    skip_errors: bool = True,
    model_type: str = "microsoft/deberta-base-mnli",
    batch_size: Optional[int] = None,
    idf: bool = False,
    lang: Optional[str] = None,
    include_per_item: bool = False,
) -> Dict[str, Any]:
    """Use HuggingFace evaluate's BERTScore."""
    used_ids, preds, refs, count_total, skipped = _load_preds_refs(
        json_path, skip_errors=skip_errors
    )
    path = Path(json_path)

    if not preds:
        out: Dict[str, Any] = {
            "file": str(path),
            "count_total": count_total,
            "count_used": 0,
            "count_skipped": skipped,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
        }
        if include_per_item:
            out["per_item"] = []
        return out

    bertscore = evaluate.load("bertscore")
    scores = bertscore.compute(
        predictions=preds,
        references=refs,
        model_type=model_type,
        batch_size=batch_size,
        idf=idf,
        lang=lang,
    )
    precisions: List[float] = scores["precision"]
    recalls: List[float] = scores["recall"]
    f1s: List[float] = scores["f1"]

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)

    out2: Dict[str, Any] = {
        "file": str(path),
        "count_total": count_total,
        "count_used": len(preds),
        "count_skipped": skipped,
        "model_type": model_type,
        "idf": idf,
        "avg_precision": float(avg_p),
        "avg_recall": float(avg_r),
        "avg_f1": float(avg_f1),
    }
    if include_per_item:
        out2["per_item"] = [
            {
                "conversation_id": cid,
                "precision": p,
                "recall": r,
                "f1": f,
            }
            for cid, p, r, f in zip(used_ids, precisions, recalls, f1s)
        ]
    return out2

def llm_as_judge(config: Config, system_setting: SystemSetting, model_a_response: str, model_b_response: str, history_turns_str: str) -> str:
    big5 = Big5(**big5_mapping[system_setting.your_character_info.archetype])
    prompt = llm_judge_one_round_prompt.format(
        system_setting=system_setting,
        your_big5=big5,
        history_turns_str=history_turns_str,
        model_a_response=model_a_response,
        model_b_response=model_b_response
    )
    response = call_llm(prompt, config)
    winner = parse_bw_tags(response, "Winner").strip()
    return winner

def run_llm_as_judge_for_one_round_results_json(config: Config, results_json: Path, dataset_json: Path) -> str:
    if results_json is None or dataset_json is None:
        raise ValueError("results_json and dataset_json are required")
    results = json.load(open(results_json, "r", encoding="utf-8"))
    dataset = build_dataset(dataset_json)
    data_path = Path(dataset.file_path)
    results_source_path = Path(results["dataset_path"])
    assert results_source_path.stem == data_path.stem
    for result, conversation in zip(results["results"], dataset.conversations):
        assert result["conversation_id"] == conversation.conversation_id
    for result, conversation in tqdm(zip(results["results"], dataset.conversations), total=len(results["results"]), desc="LLM Judging"):
        system_value = conversation.conversations[0].value
        system_setting = SystemSetting(scenario=None, topic=None, your_character_info=None, other_character_info=None)
        system_setting.from_value(system_value)
        history_turns_str = []
        for turn in conversation.conversations[1:]:
            if turn.role == "gpt":
                value = parse_bw_tags(turn.value, "Response")
                history_turns_str.append(f"{system_setting.your_character_info.name}: {value}")
            elif turn.role == "human":
                history_turns_str.append(f"{system_setting.other_character_info.name}: {turn.value}")
            else:
                raise ValueError(f"Unknown role: {turn.role}")
        history_turns_str = "\n".join(history_turns_str[:-1])
        predicted_response = parse_bw_tags(result["predicted_response"], "Response").strip()
        true_response = parse_bw_tags(conversation.conversations[-1].value, "Response").strip()
        winner = llm_as_judge(config, system_setting, predicted_response, true_response, history_turns_str)
        result["winner"] = winner
    new_file_path = results_json.parent / (results_json.stem + "_llm_judge.json")
    with open(new_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results

def _main() -> None:
    parser = argparse.ArgumentParser(description="Compute metrics for one-round results JSON")
    parser.add_argument(
        "--results",
        type=str,
        required=True
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rouge",
        choices=["rouge", "bleu", "bertscore", "llm", "metrics"]
    )
    # parser.add_argument(
    #     "--rouge_type",
    #     type=str,
    #     default="rougeL",
    #     help="rouge1/rouge2/rougeL/rougeLsum",
    # )
    parser.add_argument("--include_per_item", action="store_true")
    parser.add_argument("--no_skip_errors", action="store_true", help="Do not skip items with error_message")
    parser.add_argument("--no_stemmer", action="store_true", help="Disable stemming in ROUGE computation")
    parser.add_argument("--use_aggregator", action="store_true", help="Use aggregator in ROUGE computation")
    parser.add_argument("--bertscore_model", type=str, default="microsoft/deberta-base-mnli")
    parser.add_argument("--bertscore_idf", action="store_true", help="Use IDF weighting for BERTScore")
    parser.add_argument("--bertscore_lang", type=str, default=None, help="Language code for BERTScore")
    parser.add_argument("--bertscore_batch_size", type=int, default=None, help="Batch size for BERTScore")
    parser.add_argument("--dataset", type=str, default=None)

    parser.add_argument("--rouge_type", type=str, default="rougeL", help="ROUGE type")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    args = parser.parse_args()

    llm_config = Config(
        api_key="AIzaSyAgOkfCTbV5P1s6tKPDWEtsTZPt1tjZWXs",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_name="gemini-2.5-pro",
        temperature=1,
        max_tokens=8192,
        top_p=0.8,
    )

    if args.metric == "llm":
        report = run_llm_as_judge_for_one_round_results_json(llm_config, args.results, args.dataset)
    elif args.metric == "rouge":
        report = rouge_for_one_round_results_json(
            args.results,
            skip_errors=not args.no_skip_errors,
            include_per_item=args.include_per_item,
            rouge_type=args.rouge_type,
            use_stemmer=not args.no_stemmer,
            use_aggregator=args.use_aggregator,
        )
    elif args.metric == "bleu":
        report = bleu_for_one_round_results_json(
            args.results,
            skip_errors=not args.no_skip_errors,
        )
    elif args.metric == "metrics":
        load_two_results_json(args.results, args.dataset)
        report = metrics_for_one_round_results_json(
            args.results,
            skip_errors=not args.no_skip_errors,
            use_stemmer=not args.no_stemmer,
            use_aggregator=args.use_aggregator,
        )
    else:  # bertscore
        report = bertscore_for_one_round_results_json(
            args.results,
            skip_errors=not args.no_skip_errors,
            model_type=args.bertscore_model,
            batch_size=args.bertscore_batch_size,
            idf=args.bertscore_idf,
            lang=args.bertscore_lang,
            include_per_item=args.include_per_item,
        )

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    _main()
