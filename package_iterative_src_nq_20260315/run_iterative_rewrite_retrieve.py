#!/usr/bin/env python3
"""CLI: iterative sub-question generation + rewrite + retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# =========================
# 路径设置：确保脚本直接运行时可导入本地包
# =========================
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative query decomposition + rewrite + retrieve pipeline")
    parser.add_argument("--query", type=str, required=True, help="Main query")
    parser.add_argument("--corpus-name", type=str, default="hotpotpqa", help="Elasticsearch index name")
    parser.add_argument("--max-iterations", type=int, default=4, help="Max number of iterations")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k docs per iteration after merge")
    parser.add_argument(
        "--experiment-mode",
        choices=["baseline", "greedy", "best_of_n"],
        default="baseline",
        help="baseline: one rewrite per prompt; greedy: 1 temp=0 + N-1 temp=0.7; best_of_n: N sampled rewrites",
    )
    parser.add_argument("--rewrite-n", type=int, default=6, help="N for greedy/best_of_n candidate generation")
    parser.add_argument("--rewrite-max-chars", type=int, default=320, help="Max allowed rewrite length before filtering")
    parser.add_argument("--enable-rewrite-cache", action="store_true", help="Enable rewrite-stage cache")
    parser.add_argument(
        "--enable-rewrite-diversity-boost",
        action="store_true",
        help="Enable adaptive temperature increase to get at least M unique rewrites",
    )
    parser.add_argument("--rewrite-min-unique", type=int, default=0, help="M unique rewrites target (0 to disable)")
    parser.add_argument("--rewrite-diversity-temperature-step", type=float, default=0.15)
    parser.add_argument("--rewrite-diversity-max-temperature", type=float, default=1.3)
    parser.add_argument("--rewrite-diversity-max-rounds", type=int, default=4)
    parser.add_argument(
        "--rewrite-prompts",
        nargs="+",
        default=["hyde", "keyword_rewrite", "rafe_sft_rewrite"],
        help="Prompt templates for rewrite model",
    )
    parser.add_argument("--es-host", type=str, default="localhost", help="Elasticsearch host")
    parser.add_argument("--es-port", type=int, default=9200, help="Elasticsearch port")
    parser.add_argument("--es-user", type=str, default="elastic", help="Elasticsearch username")
    parser.add_argument("--es-password", type=str, default="246247", help="Elasticsearch password")
    parser.add_argument("--enable-reranker", action="store_true", help="Enable BGE reranker on deduplicated docs")
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="BAAI/bge-reranker-base",
        help="Reranker model name or local path",
    )
    parser.add_argument("--reranker-top-k", type=int, default=5, help="Top-k docs after reranking")
    parser.add_argument("--reranker-batch-size", type=int, default=16, help="Reranker batch size")
    parser.add_argument("--reranker-device", type=str, default=None, help="Reranker device: cuda/cpu")
    parser.add_argument(
        "--subquestion-prompt-file",
        type=Path,
        default=None,
        help="Optional file path for sub-question prompt template",
    )
    parser.add_argument(
        "--control-prompt-file",
        type=Path,
        default=None,
        help="Optional file path for iteration control prompt template",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output path")
    parser.add_argument(
        "--output-mode",
        choices=["structured", "interface"],
        default="structured",
        help="structured: return full intermediate trace; interface: return run_interface-style tuple",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 延迟导入：避免仅查看 --help 时触发不必要的模型/torch初始化
    from iterative_workflow.elasticsearch_retriever import ElasticsearchRetriever
    from iterative_workflow.iterative_prompts import ITERATION_CONTROL_PROMPT, ITERATIVE_SUBQUESTION_PROMPT
    from iterative_workflow.pipeline import IterativeRewriteRetrieverPipeline, PipelineConfig

    subquestion_prompt_template = None
    if args.subquestion_prompt_file is not None:
        subquestion_prompt_template = args.subquestion_prompt_file.read_text(encoding="utf-8")

    control_prompt_template = None
    if args.control_prompt_file is not None:
        control_prompt_template = args.control_prompt_file.read_text(encoding="utf-8")

    config = PipelineConfig(
        corpus_name=args.corpus_name,
        max_iterations=args.max_iterations,
        retrieval_top_k=args.top_k,
        rewrite_prompt_names=args.rewrite_prompts,
        experiment_mode=args.experiment_mode,
        rewrite_n=args.rewrite_n,
        rewrite_max_chars=args.rewrite_max_chars,
        enable_rewrite_cache=args.enable_rewrite_cache,
        enable_rewrite_diversity_boost=args.enable_rewrite_diversity_boost,
        rewrite_min_unique=args.rewrite_min_unique,
        rewrite_diversity_temperature_step=args.rewrite_diversity_temperature_step,
        rewrite_diversity_max_temperature=args.rewrite_diversity_max_temperature,
        rewrite_diversity_max_rounds=args.rewrite_diversity_max_rounds,
        subquestion_prompt_template=(
            subquestion_prompt_template
            if subquestion_prompt_template is not None
            else ITERATIVE_SUBQUESTION_PROMPT
        ),
        control_prompt_template=(
            control_prompt_template
            if control_prompt_template is not None
            else ITERATION_CONTROL_PROMPT
        ),
        enable_reranker=args.enable_reranker,
        reranker_model_name=args.reranker_model,
        reranker_top_k=args.reranker_top_k,
        reranker_batch_size=args.reranker_batch_size,
        reranker_device=args.reranker_device,
    )
    retriever = ElasticsearchRetriever(
        host=args.es_host,
        port=args.es_port,
        username=args.es_user,
        password=args.es_password,
    )
    pipeline = IterativeRewriteRetrieverPipeline(config=config, retriever=retriever)

    if args.output_mode == "interface":
        compressed_predict, retrieved_times, precision, recall = pipeline.run_interface(main_query=args.query)
        result = {
            "compressed_predict": compressed_predict,
            "retrieved_times": retrieved_times,
            "precision": precision,
            "recall": recall,
        }
    else:
        result = pipeline.run(main_query=args.query)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
