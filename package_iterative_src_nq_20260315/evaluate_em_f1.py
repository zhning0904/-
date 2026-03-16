#!/usr/bin/env python3
"""
EM/F1 evaluation utility.

该脚本复用了 temp/wikimultihop_evaluate.py 中的核心逻辑：
1. 归一化规则（小写、去标点、去冠词、压缩空白）
2. EM 计算
3. F1 计算

同时提供两个能力：
- 单样本 question/answer 的 EM/F1
- 批量样本 EM/F1 统计
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


class EMF1Evaluator:
    """EM/F1 评估器（还原原脚本核心计算逻辑）。"""

    # =====================================================
    # 1) 文本归一化
    # =====================================================
    def normalize_answer(self, s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            return "".join(ch for ch in text if ch not in set(string.punctuation))

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

    def normalize_answer_for_musique(self, s: str) -> str:
        """Musique 版本：把连字符当空格再去标点。"""

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            text = re.sub(r"[-–—]", " ", text)
            return "".join(ch for ch in text if ch not in set(string.punctuation))

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

    # =====================================================
    # 2) 单样本指标
    # =====================================================
    def exact_match_score(self, prediction: str, ground_truth: str, use_musique_norm: bool = False) -> int:
        """计算单条 EM。"""
        if use_musique_norm:
            pred = self.normalize_answer_for_musique(prediction)
            gt = self.normalize_answer_for_musique(ground_truth)
        else:
            pred = self.normalize_answer(prediction)
            gt = self.normalize_answer(ground_truth)
        return int(pred == gt)

    def f1_score(self, prediction: str, ground_truth: str, use_musique_norm: bool = False) -> float:
        """计算单条 F1。"""
        if use_musique_norm:
            prediction_tokens = self.normalize_answer_for_musique(prediction).split()
            ground_truth_tokens = self.normalize_answer_for_musique(ground_truth).split()
        else:
            prediction_tokens = self.normalize_answer(prediction).split()
            ground_truth_tokens = self.normalize_answer(ground_truth).split()

        common = set(prediction_tokens) & set(ground_truth_tokens)
        num_same = sum(min(prediction_tokens.count(w), ground_truth_tokens.count(w)) for w in common)

        if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            return float(int(prediction_tokens == ground_truth_tokens))
        if num_same == 0:
            return 0.0

        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        return float(2 * precision * recall / (precision + recall))

    def evaluate_single(
        self,
        question: str,
        answer: str,
        prediction: str,
        use_musique_norm: bool = False,
    ) -> Dict[str, Any]:
        """评估单条样本。"""
        em = self.exact_match_score(prediction, answer, use_musique_norm=use_musique_norm)
        f1 = self.f1_score(prediction, answer, use_musique_norm=use_musique_norm)
        return {
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "em": em,
            "f1": f1,
        }

    # =====================================================
    # 3) 批量评估
    # =====================================================
    def evaluate_batch(self, samples: Iterable[Dict[str, Any]], use_musique_norm: bool = False) -> Dict[str, Any]:
        """批量评估，返回逐条结果和平均分。"""
        results: List[Dict[str, Any]] = []
        total_em = 0.0
        total_f1 = 0.0

        for idx, item in enumerate(samples):
            question = str(item.get("question", item.get("query", "")))
            answer = str(item.get("answer", item.get("gold", "")))
            prediction = str(item.get("prediction", item.get("predict", item.get("output", ""))))

            single = self.evaluate_single(
                question=question,
                answer=answer,
                prediction=prediction,
                use_musique_norm=use_musique_norm,
            )
            single["index"] = idx
            results.append(single)

            total_em += float(single["em"])
            total_f1 += float(single["f1"])

        n = len(results)
        avg_em = total_em / n if n else 0.0
        avg_f1 = total_f1 / n if n else 0.0

        return {
            "count": n,
            "avg_em": avg_em,
            "avg_f1": avg_f1,
            "results": results,
        }


def load_samples(input_path: Path) -> List[Dict[str, Any]]:
    """支持 JSON(list) 或 JSONL 两种格式。"""
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # 先尝试 JSON list
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    # 回退 JSONL
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            out.append(obj)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EM/F1 for single or batch QA samples")

    mode = parser.add_subparsers(dest="mode", required=True)

    # 单条模式
    p_single = mode.add_parser("single", help="Evaluate one question-answer pair")
    p_single.add_argument("--question", type=str, default="", help="Question text")
    p_single.add_argument("--answer", type=str, required=True, help="Ground-truth answer")
    p_single.add_argument("--prediction", type=str, required=True, help="Model prediction")
    p_single.add_argument("--musique-norm", action="store_true", help="Use musique normalization")

    # 批量模式
    p_batch = mode.add_parser("batch", help="Evaluate a JSON/JSONL file")
    p_batch.add_argument("--input", type=Path, required=True, help="Input JSON/JSONL path")
    p_batch.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    p_batch.add_argument("--musique-norm", action="store_true", help="Use musique normalization")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = EMF1Evaluator()

    if args.mode == "single":
        result = evaluator.evaluate_single(
            question=args.question,
            answer=args.answer,
            prediction=args.prediction,
            use_musique_norm=args.musique_norm,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mode == "batch":
        samples = load_samples(args.input)
        report = evaluator.evaluate_batch(samples, use_musique_norm=args.musique_norm)

        # 控制台输出简报
        summary = {
            "count": report["count"],
            "avg_em": report["avg_em"],
            "avg_f1": report["avg_f1"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved detailed report to: {args.output}")


if __name__ == "__main__":
    main()
