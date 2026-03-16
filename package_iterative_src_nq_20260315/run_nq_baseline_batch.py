#!/usr/bin/env python3
"""Batch runner for non-iterative NQ baseline: greedy rewrite -> retrieve -> answer -> compress."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluate_em_f1 import EMF1Evaluator
from iterative_workflow.elasticsearch_retriever import ElasticsearchRetriever
from iterative_workflow.pipeline import IterativeRewriteRetrieverPipeline, PipelineConfig
from iterative_workflow.prompt_templates import build_nq_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单跳基线批跑：重写 -> 检索 -> 生成答案 -> 压缩答案"
    )

    # =========================
    # 数据与评测范围配置
    # =========================
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("src/Dataset/biencoder_nq_train_sample_500.json"),
        help="输入数据集文件（JSON 列表）",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="可选：仅评测前 N 条样本，用于快速验证")

    # =========================
    # 检索配置
    # =========================
    parser.add_argument("--corpus-name", type=str, default="nq_train", help="Elasticsearch 索引名")
    parser.add_argument("--top-k", type=int, default=10, help="最终用于答案生成的文档数")
    parser.add_argument("--retrieval-buffer-k", type=int, default=50, help="检索阶段候选缓冲池大小")
    parser.add_argument("--bm25-top-k", type=int, default=30, help="每个查询的 BM25 初始召回条数")

    # =========================
    # 重写策略配置
    # =========================
    parser.add_argument(
        "--experiment-mode",
        type=str,
        default="greedy",
        choices=["greedy", "best_of_n"],
        help="greedy：每模板取1条；best_of_n：每模板采样N条后用reranker选最优",
    )
    parser.add_argument(
        "--rewrite-prompts",
        nargs="+",
        default=["hyde", "keyword_rewrite", "rafe_sft_rewrite"],
        help="启用的重写模板列表",
    )
    parser.add_argument("--rewrite-n", type=int, default=6, help="best_of_n 下每模板采样候选数量")
    parser.add_argument("--rewrite-max-chars", type=int, default=320, help="重写文本长度上限")

    # =========================
    # Elasticsearch 连接配置
    # =========================
    parser.add_argument("--es-host", type=str, default="localhost", help="Elasticsearch 主机地址")
    parser.add_argument("--es-port", type=int, default=9200, help="Elasticsearch 端口")
    parser.add_argument("--es-user", type=str, default="elastic", help="Elasticsearch 用户名")
    parser.add_argument("--es-password", type=str, default="246247", help="Elasticsearch 密码")

    # =========================
    # Reranker 配置
    # =========================
    parser.add_argument("--enable-reranker", action="store_true", help="是否启用 reranker 做检索融合重排")
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="/root/autodl-tmp/temp/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70",
        help="reranker 模型路径（本地目录）",
    )
    parser.add_argument("--reranker-top-k", type=int, default=10, help="重排后保留的文档数")
    parser.add_argument("--reranker-batch-size", type=int, default=16, help="reranker 推理批大小")
    parser.add_argument("--reranker-device", type=str, default=None, help="reranker 运行设备（如 cuda:0 / cpu）")

    # =========================
    # 运行时与缓存配置
    # =========================
    parser.add_argument("--general-on-cpu", action="store_true", help="是否强制 general 模型在 CPU 上运行")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（影响采样与可复现性）")
    parser.add_argument("--enable-llm-cache", action="store_true", help="是否启用 LLM 响应缓存")
    parser.add_argument("--llm-cache-path", type=str, default="outputs/cache/general_llm_cache.json", help="LLM 缓存文件路径")

    # =========================
    # 输出配置
    # 命名约定：数据集-模式-(非greedy时附N)-日期
    # =========================
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="逐样本输出文件（默认按命名约定自动生成）",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="汇总输出文件（默认按命名约定自动生成）",
    )
    parser.add_argument("--fresh", action="store_true", help="Overwrite old outputs")
    return parser.parse_args()


def _default_output_paths(dataset: Path, mode: str, rewrite_n: int) -> tuple[Path, Path]:
    """构造默认输出路径：数据集-模式-(非greedy时附N)-日期。"""
    date_tag = datetime.now().strftime("%Y%m%d")
    dataset_tag = dataset.stem
    mode_tag = str(mode).lower()

    parts = [dataset_tag, mode_tag]
    if mode_tag != "greedy":
        parts.append(f"n{int(rewrite_n)}")
    parts.append(date_tag)
    stem = "-".join(parts)

    out_dir = Path("outputs/nq_baseline")
    return (
        out_dir / f"{stem}.jsonl",
        out_dir / f"{stem}_summary.json",
    )


def _load_completed_indices(jsonl_path: Path) -> set[int]:
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
                continue
            idx = row.get("index")
            if isinstance(idx, int):
                completed.add(idx)
    return completed


def _gold_answers(sample: Dict[str, Any]) -> List[str]:
    vals: List[str] = []
    answers = sample.get("answers")
    if isinstance(answers, list):
        for x in answers:
            s = str(x).strip()
            if s:
                vals.append(s)

    single = str(sample.get("answer", "")).strip()
    if single:
        vals.append(single)

    deduped: List[str] = []
    seen = set()
    for x in vals:
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(x)
    return deduped


def _best_em_f1(evaluator: EMF1Evaluator, prediction: str, gold_answers: List[str]) -> tuple[float, float]:
    if not gold_answers:
        return 0.0, 0.0
    em = max(float(evaluator.exact_match_score(prediction, gt)) for gt in gold_answers)
    f1 = max(float(evaluator.f1_score(prediction, gt)) for gt in gold_answers)
    return em, f1


def _build_reference_from_docs(docs: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, d in enumerate(docs, start=1):
        title = str(d.get("title", "")).strip()
        para = str(d.get("paragraph_text", "")).strip()
        if not para:
            continue
        if title:
            lines.append(f"Doc {idx} Title: {title}\nDoc {idx} Text: {para}")
        else:
            lines.append(f"Doc {idx} Text: {para}")
    return "\n\n".join(lines).strip()


def main() -> None:
    args = parse_args()

    # 未显式指定时，按“数据集-模式-(非greedy带N)-日期”自动命名输出文件。
    if args.output_jsonl is None or args.summary_json is None:
        default_jsonl, default_summary = _default_output_paths(
            dataset=args.dataset,
            mode=str(args.experiment_mode),
            rewrite_n=int(args.rewrite_n),
        )
        if args.output_jsonl is None:
            args.output_jsonl = default_jsonl
        if args.summary_json is None:
            args.summary_json = default_summary

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    with args.dataset.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list")

    total_dataset_size = len(data)
    if args.max_samples is not None:
        data = data[: int(args.max_samples)]

    config = PipelineConfig(
        corpus_name=args.corpus_name,
        max_iterations=1,
        retrieval_top_k=int(args.top_k),
        retrieval_buffer_k=int(args.retrieval_buffer_k),
        retrieval_bm25_top_k=int(args.bm25_top_k),
        rewrite_prompt_names=[str(x) for x in args.rewrite_prompts],
        experiment_mode=args.experiment_mode,
        rewrite_n=int(args.rewrite_n),
        rewrite_max_chars=int(args.rewrite_max_chars),
        enable_rewrite_cache=False,
        enable_rewrite_diversity_boost=False,
        rewrite_min_unique=0,
        rewrite_diversity_temperature_step=0.15,
        rewrite_diversity_max_temperature=1.3,
        rewrite_diversity_max_rounds=4,
        enable_reranker=bool(args.enable_reranker),
        reranker_model_name=str(args.reranker_model),
        reranker_top_k=int(args.reranker_top_k),
        reranker_batch_size=int(args.reranker_batch_size),
        reranker_device=args.reranker_device,
        general_on_cpu=bool(args.general_on_cpu),
        include_original_query_in_baseline_retrieval=True,
        enable_overlap_filter=False,
        random_seed=int(args.seed),
        enable_llm_cache=bool(args.enable_llm_cache),
        llm_cache_path=str(args.llm_cache_path),
    )

    retriever = ElasticsearchRetriever(
        host=str(args.es_host),
        port=int(args.es_port),
        username=str(args.es_user),
        password=str(args.es_password),
    )
    pipeline = IterativeRewriteRetrieverPipeline(config=config, retriever=retriever)
    evaluator = EMF1Evaluator()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    if args.fresh and args.output_jsonl.exists():
        args.output_jsonl.unlink()

    completed_indices = _load_completed_indices(args.output_jsonl)
    if completed_indices:
        print(f"Resume mode: found {len(completed_indices)} completed samples in {args.output_jsonl}")

    records_count = 0
    output_mode = "a" if args.output_jsonl.exists() else "w"
    with args.output_jsonl.open(output_mode, encoding="utf-8") as out_f:
        for i, sample in enumerate(data):
            if i in completed_indices:
                continue

            question = str(sample.get("question", "")).strip()
            if not question:
                continue

            gold_answers = _gold_answers(sample)

            rewrites = pipeline._rewrite_candidates(question)

            if args.experiment_mode == "best_of_n":
                # Keep selection policy aligned with pipeline best_of_n branch: choose best per prompt.
                by_prompt: Dict[str, List[Dict[str, str]]] = {}
                for cand in rewrites:
                    name = cand.get("prompt_name", "")
                    by_prompt.setdefault(name, []).append(cand)

                selected_rewrites: List[Dict[str, str]] = []
                selected_docs_all: List[Dict[str, Any]] = []
                selection_details: List[Dict[str, Any]] = []
                round_rerank_cache: Dict[str, float] = {}

                ordered_names = list(config.rewrite_prompt_names or [])
                for name in by_prompt.keys():
                    if name not in ordered_names:
                        ordered_names.append(name)

                for prompt_name in ordered_names:
                    group = by_prompt.get(prompt_name, [])
                    if not group:
                        continue
                    best_one, best_docs, details = pipeline._select_best_rewrite(
                        original_query=question,
                        candidates=group,
                        score_cache=round_rerank_cache,
                    )
                    selected_rewrites.append(best_one)
                    selected_docs_all.extend(best_docs)
                    selection_details.extend(details)

                # Reuse selected docs then run one more rerank+topk pass to keep final shape stable.
                dedup_docs: List[Dict[str, Any]] = []
                seen_doc_keys = set()
                for d in selected_docs_all:
                    key = str(d.get("paragraph_text", "")).strip().lower()
                    if not key or key in seen_doc_keys:
                        continue
                    seen_doc_keys.add(key)
                    dedup_docs.append(d)

                dedup_docs = pipeline._rerank_with_cache(
                    original_query=question,
                    docs=dedup_docs,
                    score_cache=round_rerank_cache,
                )
                top_k = config.reranker_top_k if config.reranker_top_k > 0 else config.retrieval_top_k
                retrieved_docs = dedup_docs[:top_k]
                retrieval_debug = {
                    "mode": "best_of_n",
                    "selection_strategy": "selected_best_per_prompt_by_reranker",
                    "selected_rewrites": selected_rewrites,
                    "selection_details": selection_details,
                    "selected_doc_pool_size": len(selected_docs_all),
                    "selected_doc_pool_dedup_size": len(dedup_docs),
                }
            else:
                selected_map: Dict[str, Dict[str, str]] = {}
                for cand in rewrites:
                    name = cand.get("prompt_name", "")
                    if name not in selected_map:
                        selected_map[name] = cand
                selected_rewrites = list(selected_map.values())
                retrieved_docs, retrieval_debug = pipeline._retrieve(question, selected_rewrites)

            reference = _build_reference_from_docs(retrieved_docs)

            answer_prompt = build_nq_prompt(question=question, reference=reference)
            raw_answer = pipeline._sample_one(
                pipeline.general_llm,
                prompt=answer_prompt,
                temperature=0.0,
                max_tokens=128,
                top_p=0.9,
                use_cache=True,
                cache_scope="nq_answer",
            )
            assert isinstance(raw_answer, str)

            compressed_prediction = pipeline._compress_final_answer(question, raw_answer)
            em, f1 = _best_em_f1(evaluator, compressed_prediction, gold_answers)

            record = {
                "index": i,
                "dataset_name": args.dataset.stem,
                "question": question,
                "reference_answers": gold_answers,
                "raw_prediction": raw_answer,
                "final_prediction": compressed_prediction,
                "em": em,
                "f1": f1,
                "retrieved_times": 1,
                "rewrite_count": len(rewrites),
                "selected_rewrites": selected_rewrites,
                "retrieval_debug": retrieval_debug,
                "retrieved_docs": retrieved_docs,
                "experiment_report": {
                    "mode": "non_iterative_nq_baseline",
                    "config": asdict(config),
                    "dataset": {
                        "path": str(args.dataset),
                        "total_size": total_dataset_size,
                        "eval_size": len(data),
                    },
                },
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_count += 1

            if records_count % 10 == 0:
                print(f"Processed {records_count} new samples")

    all_records: List[Dict[str, Any]] = []
    with args.output_jsonl.open("r", encoding="utf-8") as f:
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

    summary = {
        "mode": "non_iterative_nq_baseline",
        "dataset_name": args.dataset.stem,
        "dataset_path": str(args.dataset),
        "dataset_total_size": total_dataset_size,
        "dataset_eval_size": len(data),
        "count": len(all_records),
        "avg_em": avg_em,
        "avg_f1": avg_f1,
        "output_jsonl": str(args.output_jsonl),
        "runtime_args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "pipeline_config": asdict(config),
    }

    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
