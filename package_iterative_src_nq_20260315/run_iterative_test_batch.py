#!/usr/bin/env python3
"""Batch runner: evaluate iterative pipeline on full test set."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# =========================
# 路径设置：确保脚本直接运行时可导入本地包
# =========================
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluate_em_f1 import EMF1Evaluator


DEFAULT_RUN_CONFIG = ROOT_DIR / "experiments" / "iterative_runner_config.json"


def _load_completed_indices(jsonl_path: Path) -> set[int]:
    """Load finished sample indices from an existing jsonl output file."""
    if not jsonl_path.exists():
        return set()

    completed: set[int] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                # Allow resuming after abrupt interruption that leaves a partial trailing line.
                continue
            idx = row.get("index")
            if isinstance(idx, int):
                completed.add(idx)
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run iterative pipeline on test set in batch (config-driven)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RUN_CONFIG,
        help="Unified run config JSON path",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print merged common + batch_test config and exit",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a dict: {config_path}")
    return data


def _strip_comment_keys(obj: dict) -> dict:
    """Drop _comment_* keys so runtime config contains only effective options."""
    return {k: v for k, v in obj.items() if not str(k).startswith("_comment_")}


def _gold_answers(sample: dict) -> list[str]:
    """Collect deduplicated gold answers from either `answers` or `answer` field."""
    vals: list[str] = []

    answers = sample.get("answers")
    if isinstance(answers, list):
        for x in answers:
            s = str(x).strip()
            if s:
                vals.append(s)

    single = str(sample.get("answer", "")).strip()
    if single:
        vals.append(single)

    deduped: list[str] = []
    seen = set()
    for x in vals:
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(x)
    return deduped


def _best_em_f1(evaluator: EMF1Evaluator, prediction: str, gold_answers: list[str]) -> tuple[float, float]:
    """Compute best EM/F1 against multiple references."""
    if not gold_answers:
        return 0.0, 0.0
    em = max(float(evaluator.exact_match_score(prediction, gt)) for gt in gold_answers)
    f1 = max(float(evaluator.f1_score(prediction, gt)) for gt in gold_answers)
    return em, f1


def main() -> None:
    args = parse_args()

    conf = _load_config(args.config)
    common = _strip_comment_keys(conf.get("common", {}))
    batch = _strip_comment_keys(conf.get("batch_test", {}))
    if not isinstance(common, dict) or not isinstance(batch, dict):
        raise ValueError("Config sections 'common' and 'batch_test' must be objects")
    runtime = dict(common)
    runtime.update(batch)

    if args.print_config:
        print(json.dumps(runtime, ensure_ascii=False, indent=2))
        return

    from iterative_workflow.elasticsearch_retriever import ElasticsearchRetriever
    from iterative_workflow.pipeline import IterativeRewriteRetrieverPipeline, PipelineConfig

    dataset = Path(str(runtime.get("dataset", "src/Dataset/hotpot_train_test_sample_200.json")))
    with dataset.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Invalid dataset: {args.dataset}")

    total_dataset_size = len(data)

    max_samples_cfg = runtime.get("max_samples", None)
    if max_samples_cfg is not None:
        data = data[: int(max_samples_cfg)]

    dataset_name = dataset.stem
    test_size = len(data)
    reranker_top_k_cfg = runtime.get("reranker_top_k", None)
    reranker_top_k = int(reranker_top_k_cfg) if reranker_top_k_cfg is not None else int(runtime.get("top_k", 10))

    output_jsonl_cfg = runtime.get("output_jsonl", None)
    summary_json_cfg = runtime.get("summary_json", None)
    output_jsonl = Path(str(output_jsonl_cfg)) if output_jsonl_cfg else (
        Path("outputs/iterative") / f"{dataset_name}-{test_size}-N{int(runtime.get('rewrite_n', 6))}.jsonl"
    )
    summary_json = Path(str(summary_json_cfg)) if summary_json_cfg else (
        Path("outputs/iterative") / f"{dataset_name}-{test_size}-N{int(runtime.get('rewrite_n', 6))}-summary.json"
    )

    rewrite_prompts = runtime.get("rewrite_prompts", ["hyde", "keyword_rewrite", "rafe_sft_rewrite"])
    if not isinstance(rewrite_prompts, list) or not rewrite_prompts:
        raise ValueError("rewrite_prompts must be a non-empty list in config")

    config = PipelineConfig(
        corpus_name=str(runtime.get("corpus_name", "hotpotpqa")),
        max_iterations=int(runtime.get("max_iterations", 4)),
        retrieval_top_k=int(runtime.get("top_k", 10)),
        retrieval_buffer_k=int(runtime.get("retrieval_buffer_k", 50)),
        retrieval_bm25_top_k=int(runtime.get("bm25_top_k", 30)),
        rewrite_prompt_names=[str(x) for x in rewrite_prompts],
        experiment_mode=str(runtime.get("experiment_mode", "baseline")),
        rewrite_n=int(runtime.get("rewrite_n", 6)),
        rewrite_max_chars=int(runtime.get("rewrite_max_chars", 320)),
        enable_rewrite_cache=bool(runtime.get("enable_rewrite_cache", False)),
        enable_rewrite_diversity_boost=bool(runtime.get("enable_rewrite_diversity_boost", False)),
        rewrite_min_unique=int(runtime.get("rewrite_min_unique", 0)),
        rewrite_diversity_temperature_step=float(runtime.get("rewrite_diversity_temperature_step", 0.15)),
        rewrite_diversity_max_temperature=float(runtime.get("rewrite_diversity_max_temperature", 1.3)),
        rewrite_diversity_max_rounds=int(runtime.get("rewrite_diversity_max_rounds", 4)),
        enable_reranker=bool(runtime.get("enable_reranker", False)),
        reranker_model_name=str(runtime.get("reranker_model", "BAAI/bge-reranker-base")),
        reranker_top_k=reranker_top_k,
        reranker_batch_size=int(runtime.get("reranker_batch_size", 16)),
        reranker_device=runtime.get("reranker_device", None),
        general_on_cpu=bool(runtime.get("general_on_cpu", False)),
        subquestion_prompt_mode=str(runtime.get("subquestion_prompt_mode", "builder")),
        control_prompt_mode=str(runtime.get("control_prompt_mode", "builder")),
        include_original_query_in_baseline_retrieval=bool(
            runtime.get("include_original_query_in_baseline_retrieval", True)
        ),
        enable_overlap_filter=bool(runtime.get("enable_overlap_filter", False)),
        min_token_overlap=float(runtime.get("min_token_overlap", 0.12)),
        random_seed=runtime.get("seed", None),
        enable_llm_cache=bool(runtime.get("enable_llm_cache", True)),
        llm_cache_path=str(runtime.get("llm_cache_path", "outputs/cache/general_llm_cache.json")),
    )

    retriever = ElasticsearchRetriever(
        host=str(runtime.get("es_host", "localhost")),
        port=int(runtime.get("es_port", 9200)),
        username=str(runtime.get("es_user", "elastic")),
        password=str(runtime.get("es_password", "246247")),
    )
    pipeline = IterativeRewriteRetrieverPipeline(config=config, retriever=retriever)

    evaluator = EMF1Evaluator()
    records = []

    completed_indices = _load_completed_indices(output_jsonl)
    if completed_indices:
        print(f"Resume mode: found {len(completed_indices)} completed samples in {output_jsonl}")

    writer_mode = "a" if output_jsonl.exists() else "w"

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open(writer_mode, encoding="utf-8") as out_f:
        for i, sample in enumerate(data):
            if i in completed_indices:
                continue

            question = str(sample.get("question", "")).strip()
            gold_answers = _gold_answers(sample)
            sample_id = str(sample.get("_id", i))

            prediction, retrieved_times, precision, recall = pipeline.run_interface(main_query=question)

            em, f1 = _best_em_f1(evaluator, prediction, gold_answers)

            record = {
                "index": i,
                "sample_id": sample_id,
                "dataset_name": dataset_name,
                "dataset_total_size": total_dataset_size,
                "dataset_eval_size": test_size,
                "question": question,
                "reference_answers": gold_answers,
                "reference_answer": gold_answers[0] if gold_answers else "",
                "final_prediction": prediction,
                "accuracy": em,
                "em": em,
                "f1": f1,
                "retrieved_times": retrieved_times,
                "precision": precision,
                "recall": recall,
                "experiment_report": {
                    "config_path": str(args.config),
                    "runtime_config": runtime,
                    "pipeline_config": asdict(config),
                    "dataset": {
                        "path": str(dataset),
                        "name": dataset_name,
                        "total_size": total_dataset_size,
                        "eval_size": test_size,
                    },
                },
            }
            records.append(record)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(data)}")

    all_records = []
    if output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    all_records.append(row)

    avg_em = sum(float(r.get("em", 0.0)) for r in all_records) / len(all_records) if all_records else 0.0
    avg_f1 = sum(float(r.get("f1", 0.0)) for r in all_records) / len(all_records) if all_records else 0.0
    avg_retrieved_times = (
        sum(float(r.get("retrieved_times", 0.0)) for r in all_records) / len(all_records) if all_records else 0.0
    )

    summary = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset),
        "dataset_total_size": total_dataset_size,
        "dataset_eval_size": test_size,
        "count": len(all_records),
        "avg_em": avg_em,
        "avg_accuracy": avg_em,
        "avg_f1": avg_f1,
        "avg_retrieved_times": avg_retrieved_times,
        "output_jsonl": str(output_jsonl),
        "n": int(runtime.get("rewrite_n", 6)),
        "runtime_config": runtime,
        "pipeline_config": asdict(config),
        "config": {
            "experiment_mode": str(runtime.get("experiment_mode", "baseline")),
            "rewrite_n": int(runtime.get("rewrite_n", 6)),
            "enable_reranker": bool(runtime.get("enable_reranker", False)),
            "retriever": {
                "corpus_name": str(runtime.get("corpus_name", "hotpotpqa")),
                "es_host": str(runtime.get("es_host", "localhost")),
                "es_port": int(runtime.get("es_port", 9200)),
                "top_k": int(runtime.get("top_k", 5)),
                "max_iterations": int(runtime.get("max_iterations", 4)),
            },
            "models": {
                "rewrite_model_role": "finetuned_rewrite",
                "general_model_role": "base_general",
                "reranker_model": str(runtime.get("reranker_model", "BAAI/bge-reranker-base")),
                "reranker_top_k": reranker_top_k,
                "reranker_batch_size": int(runtime.get("reranker_batch_size", 16)),
                "reranker_device": runtime.get("reranker_device", None),
            },
        },
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
