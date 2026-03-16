#!/usr/bin/env python3
"""Debug runner: pick one sample from test set and run iterative pipeline."""

from __future__ import annotations

import argparse
import json
import random
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one debug sample from test set (config-driven)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RUN_CONFIG,
        help="Unified run config JSON path",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print merged common + single_test config and exit",
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
    """Drop _comment_* keys so report config reflects effective runtime settings."""
    return {k: v for k, v in obj.items() if not str(k).startswith("_comment_")}


def main() -> None:
    args = parse_args()

    conf = _load_config(args.config)
    common = _strip_comment_keys(conf.get("common", {}))
    single = _strip_comment_keys(conf.get("single_test", {}))
    if not isinstance(common, dict) or not isinstance(single, dict):
        raise ValueError("Config sections 'common' and 'single_test' must be objects")
    runtime = dict(common)
    runtime.update(single)

    if args.print_config:
        print(json.dumps(runtime, ensure_ascii=False, indent=2))
        return

    from iterative_workflow.elasticsearch_retriever import ElasticsearchRetriever
    from iterative_workflow.pipeline import IterativeRewriteRetrieverPipeline, PipelineConfig

    dataset = Path(str(runtime.get("dataset", "src/Dataset/hotpot_train_test_sample_200.json")))
    with dataset.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    seed = int(runtime.get("seed", 42))
    random_sample = bool(runtime.get("random_sample", False))
    index = int(runtime.get("index", 0))

    if random_sample:
        rng = random.Random(seed)
        idx = rng.randrange(len(data))
    else:
        if index < 0 or index >= len(data):
            raise IndexError(f"index out of range: {index}, dataset size={len(data)}")
        idx = index

    sample = data[idx]
    question = str(sample.get("question", "")).strip()
    answer = str(sample.get("answer", "")).strip()
    sample_id = str(sample.get("_id", idx))

    rewrite_prompts = runtime.get("rewrite_prompts", ["hyde", "keyword_rewrite", "rafe_sft_rewrite"])
    if not isinstance(rewrite_prompts, list) or not rewrite_prompts:
        raise ValueError("rewrite_prompts must be a non-empty list in config")

    reranker_top_k_cfg = runtime.get("reranker_top_k", None)
    reranker_top_k = int(reranker_top_k_cfg) if reranker_top_k_cfg is not None else int(runtime.get("top_k", 10))

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
        random_seed=seed,
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

    # 只执行一次 run()，随后 run_interface 复用同一 trace，保证调试轨迹与最终结果完全一致
    trace = pipeline.run(main_query=question)
    prediction, retrieved_times, precision, recall = pipeline.run_interface(main_query=question, trace=trace)

    evaluator = EMF1Evaluator()
    em = evaluator.exact_match_score(prediction, answer)
    f1 = evaluator.f1_score(prediction, answer)

    step_reports = []
    for r in trace.get("rounds", []):
        step_reports.append(
            {
                "iteration": r.get("iteration"),
                "sub_question": r.get("sub_question"),
                "selection_strategy": r.get("selection_strategy"),
                "rewrite_count": r.get("rewrite_count"),
                "rewrite_candidates": [
                    {
                        "prompt_name": x.get("prompt_name"),
                        "normalized_rewrite": x.get("normalized_rewrite"),
                    }
                    for x in r.get("rewrites", [])
                ],
                "selected_rewrite": r.get("selected_rewrite"),
                "sub_answer": r.get("sub_answer", ""),
                "control_action": (r.get("control") or {}).get("action", ""),
            }
        )

    result = {
        "sample_index": idx,
        "sample_id": sample_id,
        "dataset_name": dataset.stem,
        "dataset_size": len(data),
        "question": question,
        "reference_answer": answer,
        "final_prediction": prediction,
        "accuracy": em,
        "em": em,
        "f1": f1,
        "retrieved_times": retrieved_times,
        "precision": precision,
        "recall": recall,
        "step_reports": step_reports,
        "config": {
            "experiment_mode": str(runtime.get("experiment_mode", "baseline")),
            "rewrite_n": int(runtime.get("rewrite_n", 6)),
            "enable_reranker": bool(runtime.get("enable_reranker", False)),
            "retriever": {
                "corpus_name": str(runtime.get("corpus_name", "hotpotpqa")),
                "es_host": str(runtime.get("es_host", "localhost")),
                "es_port": int(runtime.get("es_port", 9200)),
                "top_k": int(runtime.get("top_k", 5)),
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
        "experiment_report": {
            "config_path": str(args.config),
            "runtime_config": runtime,
            "pipeline_config": asdict(config),
            "dataset": {
                "path": str(dataset),
                "name": dataset.stem,
                "size": len(data),
            },
        },
    }

    result["trace"] = trace

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    output_json_cfg = runtime.get("output_json", None)
    output_path = Path(str(output_json_cfg)) if output_json_cfg else None
    if output_path is None:
        # 默认输出命名中包含样本索引与关键实验参数，便于批量比对
        output_path = Path("outputs/iterative") / (
            f"{dataset.stem}_debug_one_idx{idx}_N{int(runtime.get('rewrite_n', 6))}.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    print(f"Saved debug report to: {output_path}")


if __name__ == "__main__":
    main()
