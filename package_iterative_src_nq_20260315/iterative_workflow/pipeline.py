#!/usr/bin/env python3
"""Iterative sub-question -> rewrite -> retrieve workflow."""

from __future__ import annotations

import json
import hashlib
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

# =========================
# 路径设置：把项目的 src 加入 sys.path，方便 import 自己的模块
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.chat_llm import _load_model_roles, create_general_llm, create_rewrite_llm

from iterative_workflow.elasticsearch_retriever import ElasticsearchRetriever
from iterative_workflow.iterative_prompts import ITERATION_CONTROL_PROMPT, ITERATIVE_SUBQUESTION_PROMPT
from iterative_workflow.prompt_templates import (
    build_answer_prompt,
    build_compression_prompt,
    build_final_answer_prompt,
    build_iteration_control_prompt,
    build_subquestion_prompt,
)
try:
    from iterative_workflow.retrieval_service import RetrievalService  # type: ignore[reportMissingImports]
    from iterative_workflow.rewrite_service import RewriteService  # type: ignore[reportMissingImports]
except ImportError:
    from retrieval_service import RetrievalService  # type: ignore[reportMissingImports]
    from rewrite_service import RewriteService  # type: ignore[reportMissingImports]


@dataclass
class PipelineConfig:
    """迭代流程配置（字段均可通过 runner config 覆盖）。"""

    corpus_name: str = "hotpotpqa"  # Elasticsearch 索引名
    max_iterations: int = 4  # 每个样本最多迭代轮数
    retrieval_top_k: int = 10  # 每轮最终保留的文档数
    retrieval_buffer_k: int = 50  # 召回缓冲池大小（用于去重/融合）
    retrieval_bm25_top_k: int = 30  # BM25 初始召回条数
    rewrite_prompt_names: List[str] | None = None  # 启用的重写模板列表

    # 采样温度：subquestion/control 默认走确定性，rewrite 允许随机性
    subquestion_temperature: float = 0.0
    rewrite_temperature: float = 0.7
    control_temperature: float = 0.0

    # 各阶段最大输出 token
    max_tokens_subquestion: int = 96
    max_tokens_rewrite: int = 64
    max_tokens_control: int = 80
    max_tokens_answer: int = 120

    # 兜底模板（通常由 prompt builder 生成）
    subquestion_prompt_template: str = ITERATIVE_SUBQUESTION_PROMPT
    control_prompt_template: str = ITERATION_CONTROL_PROMPT

    # reranker 配置
    enable_reranker: bool = False
    reranker_model_name: str = "BAAI/bge-reranker-base"
    reranker_top_k: int = 10
    reranker_batch_size: int = 16
    reranker_device: str | None = None  # 例如 "cuda:0" / "cpu"，None=自动

    experiment_mode: str = "baseline"  # baseline | greedy | best_of_n
    rewrite_n: int = 6  # best_of_n 下每模板采样候选数
    rewrite_max_chars: int = 320  # rewrite 最大长度阈值（放宽过滤）
    enable_rewrite_cache: bool = False  # 是否启用 rewrite 阶段缓存（默认关闭）

    # 可选多样性增强：当 best_of_n 发生高重复时逐步升温，尽量拿到至少 M 条不重复改写
    enable_rewrite_diversity_boost: bool = False
    rewrite_min_unique: int = 0  # M，0 表示不启用；建议满足 0 < M < rewrite_n
    rewrite_diversity_temperature_step: float = 0.15
    rewrite_diversity_max_temperature: float = 1.3
    rewrite_diversity_max_rounds: int = 4

    # 资源与模板模式
    general_on_cpu: bool = False  # True 时 general 模型放 CPU，降低显存占用
    subquestion_prompt_mode: str = "builder"  # builder | raw_template
    control_prompt_mode: str = "builder"  # builder | raw_template

    # baseline 是否包含原始子问题检索
    include_original_query_in_baseline_retrieval: bool = True

    # rewrite 词重合过滤（默认关闭：不过滤低 overlap）
    enable_overlap_filter: bool = False
    min_token_overlap: float = 0.12

    # 可复现与缓存
    random_seed: int | None = None  # 全局随机种子
    enable_llm_cache: bool = True  # 启用 prompt-response 缓存
    llm_cache_path: str = "outputs/cache/general_llm_cache.json"  # 缓存文件路径


class IterativeRewriteRetrieverPipeline:
    """
    端到端流程：
    1) 从主问题生成下一步子问题
    2) 用 rewrite 专用模型按多模板改写子问题
    3) 用改写查询执行检索
    4) 用控制器判断继续/停止
    """

    def __init__(self, config: PipelineConfig, retriever: ElasticsearchRetriever):
        self.config = config
        self.retriever = retriever

        self._llm_cache: Dict[str, Dict[str, Any]] = {}
        self._llm_cache_file = ROOT_DIR / self.config.llm_cache_path
        if self.config.enable_llm_cache:
            self._load_llm_cache()

        if self.config.random_seed is not None:
            self._set_global_seed(self.config.random_seed)

        # rewrite 模型：已微调 adapter，专用于 query rewrite
        self.rewrite_llm = create_rewrite_llm()

        roles = _load_model_roles()
        rewrite_role = roles.get("rewrite", {}) if isinstance(roles, dict) else {}
        general_role = roles.get("general", {}) if isinstance(roles, dict) else {}
        # 判断重写模型和general模型是不是完全是一个模型
        same_model_roles = (
            str(rewrite_role.get("base_model_path", "")) == str(general_role.get("base_model_path", ""))
            and not rewrite_role.get("adapter_path")
            and not general_role.get("adapter_path")
            and rewrite_role.get("system_prompt") == general_role.get("system_prompt")
        )

        # general 模型：未微调基座，专用于子问题规划/控制/回答摘要
        # 可选放到 CPU，避免和 rewrite 模型同时占用大量显存导致 OOM。
        # 若 rewrite/general 实际是同一个底座且都无 adapter，复用同一实例，避免重复加载导致 OOM。
        if same_model_roles:
            self.general_llm = self.rewrite_llm
        elif self.config.general_on_cpu:
            self.general_llm = create_general_llm(device_map="cpu", torch_dtype=torch.float32)
        else:
            self.general_llm = create_general_llm()

        if self.config.rewrite_prompt_names is None:
            self.config.rewrite_prompt_names = [
                "hyde",
                "keyword_rewrite",
                "rafe_sft_rewrite",
            ]

        # 功能模块拆分：重写生成与检索融合/重排序交由独立服务管理。
        self.rewrite_service = RewriteService(
            config=self.config,
            rewrite_llm=self.rewrite_llm,
            sample_one=self._sample_one,
            sample_many=self._sample_many,
        )
        self.retrieval_service = RetrievalService(
            config=self.config,
            retriever=self.retriever,
            root_dir=ROOT_DIR,
        )
        if self.config.enable_reranker:
            self.retrieval_service.preload_reranker()

        # 记录最近一轮 rewrite 过滤信息，便于 trace/debug 排查。
        self._last_rewrite_filtering_debug: Dict[str, Any] = {
            "source_query": "",
            "kept": [],
            "dropped": [],
        }

    def _set_global_seed(self, seed: int) -> None:
        """统一设置随机种子，覆盖 Python / NumPy / Torch。"""
        random.seed(seed)
        if np is not None:
            np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_llm_cache(self) -> None:
        """加载持久化 LLM 缓存。"""
        path = self._llm_cache_file
        if not path.exists():
            self._llm_cache = {}
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._llm_cache = {}
            return

        entries = data.get("entries", {}) if isinstance(data, dict) else {}
        if isinstance(entries, dict):
            self._llm_cache = {str(k): dict(v) for k, v in entries.items() if isinstance(v, dict)}
        else:
            self._llm_cache = {}

    def _save_llm_cache(self) -> None:
        """持久化 LLM 缓存。"""
        path = self._llm_cache_file
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": self._llm_cache,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _make_llm_cache_key(
        self,
        *,
        model_role: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        cache_scope: str,
    ) -> str:
        """生成稳定缓存键。"""
        key_payload = {
            "model_role": model_role,
            "prompt": prompt,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
            "cache_scope": cache_scope,
        }
        serialized = json.dumps(key_payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ============================================================
    # 内部基础工具
    # ============================================================
    def _sample_one(
        self,
        llm,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.95,
        return_raw: bool = False,
        use_cache: bool = False,
        cache_scope: str = "",
    ) -> str | Dict[str, str]:
        """统一单条采样调用，支持返回 raw/stripped 两种文本。"""
        cache_key = ""
        if use_cache and self.config.enable_llm_cache:
            model_role = "general" if llm is self.general_llm else "rewrite"
            cache_key = self._make_llm_cache_key(
                model_role=model_role,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                cache_scope=cache_scope,
            )
            hit = self._llm_cache.get(cache_key)
            if hit is not None:
                if return_raw:
                    return {
                        "raw_text": hit.get("raw_text", ""),
                        "stripped_text": hit.get("stripped_text", ""),
                    }
                return hit.get("stripped_text", "")

        outputs = llm.sample(
            prompt=prompt,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        raw_text = outputs[0] if outputs else ""
        stripped = raw_text.strip()

        if use_cache and self.config.enable_llm_cache and cache_key:
            self._llm_cache[cache_key] = {
                "raw_text": raw_text,
                "stripped_text": stripped,
            }
            self._save_llm_cache()

        if return_raw:
            return {
                "raw_text": raw_text,
                "stripped_text": stripped,
            }
        return stripped

    def _sample_many(
        self,
        llm,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.95,
        use_cache: bool = False,
        cache_scope: str = "",
    ) -> List[str]:
        """统一多条采样调用，支持按相同参数复用缓存。"""
        sample_n = max(0, int(n))
        if sample_n <= 0:
            return []

        cache_key = ""
        if use_cache and self.config.enable_llm_cache:
            model_role = "general" if llm is self.general_llm else "rewrite"
            cache_key = self._make_llm_cache_key(
                model_role=model_role,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                cache_scope=f"{cache_scope}:n={sample_n}",
            )
            hit = self._llm_cache.get(cache_key)
            if hit is not None and isinstance(hit.get("outputs"), list):
                return [str(x).strip() for x in hit.get("outputs", [])]

        outputs = llm.sample(
            prompt=prompt,
            n=sample_n,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        stripped_outputs = [str(text or "").strip() for text in (outputs or [])]

        if use_cache and self.config.enable_llm_cache and cache_key:
            self._llm_cache[cache_key] = {
                "outputs": stripped_outputs,
            }
            self._save_llm_cache()

        return stripped_outputs

    def _history_json(self, rounds: List[Dict[str, Any]]) -> str:
        """将中间轮次压缩为 prompt 可读的 JSON。"""
        compact = []
        for item in rounds:
            compact.append(
                {
                    "iteration": item.get("iteration"),
                    "sub_question": item.get("sub_question"),
                    "sub_answer": item.get("sub_answer", ""),
                }
            )
        return json.dumps(compact, ensure_ascii=False)

    def _clean_output_text(self, text: str, *, keep_newlines: bool = True) -> str:
        """
        通用模型输出清洗：
        1) 去掉 <think>...</think> 思维链片段
        2) 去掉 markdown code fence 外壳
        3) 去掉常见 assistant/answer 前缀标签
        4) 去掉外围引号/反引号并清理空行

        该函数只做“通用降噪”，不做任务特定语义判断。
        """
        raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not raw:
            return ""

        # 删除 <think>...</think>（兼容属性和大小写）
        cleaned = re.sub(r"<think(?:\\s+[^>]*)?>[\\s\\S]*?</think>", " ", raw, flags=re.IGNORECASE)

        # 如果存在 fenced code block，优先提取其内容并拼接
        fenced = re.findall(r"```(?:[a-zA-Z0-9_+-]+)?\\s*([\\s\\S]*?)```", cleaned)
        if fenced:
            cleaned = "\n".join(part.strip() for part in fenced if part.strip())

        lines = []
        for line in cleaned.split("\n"):
            s = line.strip()
            if not s:
                continue
            # 常见输出前缀，避免污染下游解析
            s = re.sub(
                r"^(assistant|model|response|output|answer|final answer)\\s*[:：-]\\s*",
                "",
                s,
                flags=re.IGNORECASE,
            )
            s = s.strip().strip("`")
            if s:
                lines.append(s)

        if not lines:
            return ""

        out = "\n".join(lines)
        out = out.strip().strip('"\'` ')
        if not keep_newlines:
            out = re.sub(r"\\s+", " ", out)
        return out.strip()

    def _extract_json_block(self, text: str) -> str | None:
        """
        从文本中提取最可能的 JSON 对象：
        1) 优先 json 代码块
        2) 再尝试首个花括号平衡对象
        """
        raw = (text or "")
        if not raw:
            return None

        for block in re.findall(r"```(?:json)?\\s*([\\s\\S]*?)```", raw, flags=re.IGNORECASE):
            b = block.strip()
            if b.startswith("{") and b.endswith("}"):
                return b

        s = raw
        start = s.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(s)):
                ch = s[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = s[start: idx + 1].strip()
                        if candidate:
                            return candidate
                        break

            start = s.find("{", start + 1)
        return None

    def _safe_parse_control(self, text: str) -> Dict[str, str]:
        """尽量稳健地解析控制器输出；不启用 STOP 早停，统一回退为 CONTINUE。"""
        raw = text or ""
        cleaned = self._clean_output_text(raw, keep_newlines=True)

        parse_candidates: List[str] = []
        if cleaned:
            parse_candidates.append(cleaned)
        extracted = self._extract_json_block(raw)
        if extracted and extracted not in parse_candidates:
            parse_candidates.append(extracted)

        for candidate in parse_candidates:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(obj, dict):
                action = str(obj.get("action", "CONTINUE")).strip().upper()
                reason = str(obj.get("reason", "")).strip()
                if action != "CONTINUE":
                    action = "CONTINUE"
                return {"action": action, "reason": reason}

        # 非 JSON 回退：统一 CONTINUE（不启用早停）。
        fallback_text = cleaned or raw.strip()
        return {"action": "CONTINUE", "reason": fallback_text}

    def _normalize_rewrite_text(self, text: str, fallback_query: str) -> str:
        """
        rewrite 清洗：
        1) 统一通用降噪（think/code-fence/标签/引号）
        2) 去除项目符号与常见解释性前缀
        3) 仅保留单行检索 query
        4) 为空时回退到原始 query
        """
        raw = self._clean_output_text(text, keep_newlines=True)
        if not raw:
            return fallback_query

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        cleaned = ""
        for line in lines:
            line = re.sub(r"^[-*•]+\\s*", "", line)
            line = re.sub(r"^\\d+[\\.)]\\s*", "", line)
            line = line.strip()
            if not line:
                continue

            low = line.lower()
            if low.startswith(("here is", "explanation", "because", "i rewrote", "this query")):
                continue
            cleaned = line
            break

        if not cleaned:
            cleaned = lines[0] if lines else ""

        prefixes = [
            "optimized search query:",
            "search query:",
            "rewritten query:",
            "rewritten search query:",
            "final query:",
            "rewrite:",
            "query:",
        ]
        low = cleaned.lower()
        for prefix in prefixes:
            if low.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                low = cleaned.lower()
                break

        cleaned = re.sub(r"\\s+", " ", cleaned).strip('"\'` ')
        return cleaned if cleaned else fallback_query

    def _format_evidence_for_prompt(
        self,
        docs: List[Dict[str, Any]],
        *,
        top_k: int = 5,
        max_chars_per_doc: int = 520,
        max_total_chars: int = 2400,
    ) -> str:
        """
        将检索文档压缩为提示词证据块。

        设计动机：
        - 给子问题生成器显式展示“已掌握证据”，减少重复追问。
        - 给控制器展示“当前证据覆盖面”，降低因只看历史 QA 而误停/误继续。
        - 统一总字符预算，避免证据过长挤占模型决策空间。
        """
        if not docs:
            return "No retrieved evidence yet."

        lines: List[str] = []
        used = 0
        kept = 0
        for idx, d in enumerate(docs[: max(1, top_k)], start=1):
            title = str(d.get("title", "")).strip() or "Untitled"
            para = str(d.get("paragraph_text", "")).replace("\r", " ").replace("\n", " ").strip()
            if not para:
                continue
            para = re.sub(r"\\s+", " ", para)
            snippet = para[:max_chars_per_doc].rstrip()
            if len(para) > max_chars_per_doc:
                snippet += " ..."

            row = f"[{idx}] {title}: {snippet}"
            row_len = len(row)
            if used + row_len > max_total_chars:
                break
            lines.append(row)
            used += row_len
            kept += 1

        if not lines:
            return "No retrieved evidence yet."
        return f"Evidence snippets ({kept} docs):\\n" + "\\n".join(lines)

    def _canonicalize_query_text(self, text: str) -> str:
        """委托给 RewriteService 做 query 归一化。"""
        return self.rewrite_service._canonicalize_query_text(text)

    def _tokenize_for_overlap(self, text: str) -> List[str]:
        """委托给 RewriteService 做 overlap 分词。"""
        return self.rewrite_service._tokenize_for_overlap(text)

    def _evaluate_rewrite_quality(self, rewrite: str, original_query: str) -> tuple[bool, Dict[str, Any]]:
        """委托给 RewriteService 做 rewrite 质量过滤。"""
        return self.rewrite_service._evaluate_rewrite_quality(rewrite, original_query)

    def _collect_evidence_docs_from_rounds(self, rounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从历史轮次聚合证据并按段落去重。"""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for item in rounds:
            for d in item.get("retrieved_docs", []) or []:
                key = str(d.get("paragraph_text", "")).strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(d)
        return merged

    def _build_qa_history(self, rounds: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """提取已完成轮次中的 (sub_question, sub_answer) 历史。"""
        return [
            (item.get("sub_question", ""), item.get("sub_answer", ""))
            for item in rounds
            if item.get("sub_question", "")
        ]

    def _build_iteration_evidence_summary(
        self,
        rounds: List[Dict[str, Any]],
        *,
        max_chars_per_doc: int,
        max_total_chars: int,
    ) -> str:
        """为“子问题生成/控制器”统一构造证据摘要字符串。"""
        evidence_docs = self._collect_evidence_docs_from_rounds(rounds)
        return self._format_evidence_for_prompt(
            evidence_docs,
            top_k=max(self.config.retrieval_top_k, 5),
            max_chars_per_doc=max_chars_per_doc,
            max_total_chars=max_total_chars,
        )

    def _normalize_subquestion_candidates(self, cleaned_text: str) -> List[str]:
        """把模型输出标准化为候选子问题列表（去标签、去编号、去引号）。"""
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]

        candidates: List[str] = []
        for line in lines:
            item = re.sub(r"^[-*•]+\\s*", "", line)
            item = re.sub(r"^\\d+[\\.)]\\s*", "", item)
            item = re.sub(
                r"^(sub[- ]?question|next question|question|follow-up question)\\s*[:：-]\\s*",
                "",
                item,
                flags=re.IGNORECASE,
            )
            item = item.strip().strip('"\'` ')
            if item:
                candidates.append(item)
        return candidates

    def _pick_subquestion(self, candidates: List[str]) -> str:
        """从候选列表中选择一个简洁子问题。"""
        # 优先选择疑问句，若没有则取首条有效候选。
        for item in candidates:
            if "?" in item or "？" in item:
                return re.sub(r"\\s+", " ", item).strip('"\'` ')

        if candidates:
            return re.sub(r"\\s+", " ", candidates[0]).strip('"\'` ')
        return ""

    def _synthesize_sub_answer(self, sub_question: str, docs: List[Dict[str, Any]]) -> str:
        """用 general 模型基于检索证据给出简短子答案。"""
        doc_texts = [d.get("paragraph_text", "") for d in docs if d.get("paragraph_text", "")]
        prompt = build_answer_prompt(sub_query=sub_question, docs=doc_texts)
        return self._sample_one(
            self.general_llm,
            prompt=prompt,
            temperature=0.0,
            max_tokens=self.config.max_tokens_answer,
            top_p=0.9,
            use_cache=True,
            cache_scope="sub_answer",
        )

    # ============================================================
    # 核心流程步骤
    # ============================================================
    def _generate_next_subquestion(self, main_query: str, rounds: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """
        基于主问题 + QA历史 + 检索证据，生成下一条子问题。

        返回：
        - sub_question: 标准化后的子问题
        - prompt: 本轮实际使用的 prompt
        - raw_text: 模型原始输出（调试追踪用）
        """
        # 关键变量：qa_history 是已完成轮次的问答对，用于避免重复追问。
        qa_history = self._build_qa_history(rounds)

        if self.config.subquestion_prompt_mode == "builder":
            prompt = build_subquestion_prompt(
                main_query=main_query,
                history=qa_history,
            )
        elif self.config.subquestion_prompt_mode == "raw_template":
            prompt = self.config.subquestion_prompt_template.format(
                main_query=main_query,
                history_json=self._history_json(rounds),
            )
        else:
            raise ValueError(f"Unsupported subquestion_prompt_mode: {self.config.subquestion_prompt_mode}")

        sampled = self._sample_one(
            self.general_llm,
            prompt=prompt,
            temperature=self.config.subquestion_temperature,
            max_tokens=self.config.max_tokens_subquestion,
            top_p=0.95,
            return_raw=True,
        )
        assert isinstance(sampled, dict)

        raw_text = sampled["raw_text"]
        cleaned = self._clean_output_text(raw_text, keep_newlines=True)
        candidates = self._normalize_subquestion_candidates(cleaned)
        sub_question = self._pick_subquestion(candidates)
        return sub_question, prompt, raw_text

    def _rewrite_candidates(self, sub_question: str) -> List[Dict[str, str]]:
        """委托给 RewriteService 生成并过滤重写候选。"""
        candidates = self.rewrite_service.rewrite_candidates(sub_question)
        self._last_rewrite_filtering_debug = dict(self.rewrite_service.last_filtering_debug)
        return candidates

    def _merge_dedup_docs(self, docs_a: List[Dict[str, Any]], docs_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """委托给 RetrievalService 合并去重。"""
        return self.retrieval_service._merge_dedup_docs(docs_a, docs_b)

    def _select_best_rewrite(
        self,
        original_query: str,
        candidates: List[Dict[str, str]],
        score_cache: Dict[str, float] | None = None,
    ) -> tuple[Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """委托给 RetrievalService 进行 best-of-n 重写竞争。"""
        return self.retrieval_service.select_best_rewrite(
            original_query=original_query,
            candidates=candidates,
            score_cache=score_cache,
        )

    def _ensure_reranker(self):
        """委托给 RetrievalService 初始化/返回 reranker。"""
        return self.retrieval_service._ensure_reranker()

    def _rerank_with_cache(
        self,
        *,
        original_query: str,
        docs: List[Dict[str, Any]],
        score_cache: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """委托给 RetrievalService 执行带缓存重排序。"""
        return self.retrieval_service.rerank_with_cache(
            original_query=original_query,
            docs=docs,
            score_cache=score_cache,
        )

    def _retrieve(self, original_query: str, rewrites: List[Dict[str, str]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """委托给 RetrievalService 做检索融合与 rerank 排序。"""
        return self.retrieval_service.retrieve_fused(original_query, rewrites)

    def _control_iteration(self, main_query: str, rounds: List[Dict[str, Any]], iteration: int) -> Tuple[Dict[str, str], str, str]:
        """基于当前历史与证据做 CONTINUE / STOP 决策。"""
        # 关键变量：qa_history 是控制器判断“是否还缺关键信息”的历史输入。
        qa_history = self._build_qa_history(rounds)
        # 关键变量：evidence_summary 让控制器显式看到检索证据覆盖面。
        evidence_summary = self._build_iteration_evidence_summary(
            rounds,
            max_chars_per_doc=560,
            max_total_chars=2600,
        )

        if self.config.control_prompt_mode == "builder":
            prompt = build_iteration_control_prompt(
                main_query=main_query,
                history=qa_history,
                evidence_summary=evidence_summary,
                iteration=iteration,
                max_iterations=self.config.max_iterations,
            )
        elif self.config.control_prompt_mode == "raw_template":
            prompt = self.config.control_prompt_template.format(
                main_query=main_query,
                iteration=iteration,
                max_iterations=self.config.max_iterations,
                history_json=self._history_json(rounds),
                evidence_summary=evidence_summary,
            )
        else:
            raise ValueError(f"Unsupported control_prompt_mode: {self.config.control_prompt_mode}")

        sampled = self._sample_one(
            self.general_llm,
            prompt=prompt,
            temperature=self.config.control_temperature,
            max_tokens=self.config.max_tokens_control,
            top_p=0.9,
            return_raw=True,
        )
        assert isinstance(sampled, dict)
        raw_text = sampled["raw_text"]
        parsed = self._safe_parse_control(raw_text)
        return parsed, prompt, raw_text

    # ============================================================
    # 对外主入口
    # ============================================================
    def run(self, main_query: str) -> Dict[str, Any]:
        """执行完整迭代流程并返回结构化中间结果（便于调试与追踪）。"""
        rounds: List[Dict[str, Any]] = []

        for i in range(1, self.config.max_iterations + 1):
            # 每轮先生成子问题，再进行改写/检索/控制决策
            sub_question, subquestion_prompt, subquestion_raw_output = self._generate_next_subquestion(main_query, rounds)
            sub_question = sub_question.strip()

            rewrites = self._rewrite_candidates(sub_question)
            rewrite_filtering_debug = dict(self._last_rewrite_filtering_debug)
            # 每轮共享同一打分缓存：同一文档针对同一 sub_question 只打分一次。
            round_rerank_cache: Dict[str, float] = {}

            if self.config.experiment_mode == "best_of_n":
                # 每种模板类型各选 1 条最优 rewrite，再统一检索。
                by_prompt: Dict[str, List[Dict[str, str]]] = {}
                for cand in rewrites:
                    name = cand.get("prompt_name", "")
                    by_prompt.setdefault(name, []).append(cand)

                selected_rewrites: List[Dict[str, str]] = []
                selected_docs_all: List[Dict[str, Any]] = []
                rewrite_details = []

                ordered_names = list(self.config.rewrite_prompt_names or [])
                for name in by_prompt.keys():
                    if name not in ordered_names:
                        ordered_names.append(name)

                for prompt_name in ordered_names:
                    group = by_prompt.get(prompt_name, [])
                    if not group:
                        continue
                    best_one, _best_docs, details = self._select_best_rewrite(
                        original_query=sub_question,
                        candidates=group,
                        score_cache=round_rerank_cache,
                    )
                    selected_rewrites.append(best_one)
                    selected_docs_all.extend(_best_docs)
                    rewrite_details.extend(details)

                selected_rewrite = selected_rewrites
                selection_strategy = "selected_best_per_prompt_by_reranker"
                # best_of_n 已在选优阶段拿到每个模板最优 rewrite 的高分文档，直接复用，避免重复检索。
                dedup_docs: List[Dict[str, Any]] = []
                seen_doc_keys = set()
                for d in selected_docs_all:
                    key = str(d.get("paragraph_text", "")).strip().lower()
                    if not key or key in seen_doc_keys:
                        continue
                    seen_doc_keys.add(key)
                    dedup_docs.append(d)

                dedup_docs = self._rerank_with_cache(
                    original_query=sub_question,
                    docs=dedup_docs,
                    score_cache=round_rerank_cache,
                )
                top_k = self.config.reranker_top_k if self.config.reranker_top_k > 0 else self.config.retrieval_top_k
                retrieved_docs = dedup_docs[:top_k]

                retrieval_debug = {
                    "reranker_query": sub_question,
                    "reranker_enabled": True,
                    "merge_strategy": "best_per_prompt_reuse_selection_docs",
                    "selected_prompt_count": len(selected_rewrites),
                    "selected_doc_pool_size": len(selected_docs_all),
                    "selected_doc_pool_dedup_size": len(dedup_docs),
                    "retrieval_by_query": [],
                }
                retrieval_debug["selected_prompt_count"] = len(selected_rewrites)
                retrieval_debug["merge_strategy"] = "best_per_prompt_reuse_selection_docs"
            elif self.config.experiment_mode == "greedy":
                # greedy 不做竞争：每种模板类型取首条有效 rewrite。
                selected_map: Dict[str, Dict[str, str]] = {}
                for cand in rewrites:
                    name = cand.get("prompt_name", "")
                    if name not in selected_map:
                        selected_map[name] = cand

                selected_rewrites = list(selected_map.values())
                selected_rewrite = selected_rewrites
                selection_strategy = "greedy_one_per_prompt_no_competition"
                retrieved_docs, retrieval_debug = self._retrieve(sub_question, selected_rewrites)
                rewrite_details = retrieval_debug.get("retrieval_by_query", [])
            else:
                selected_rewrite = None
                selection_strategy = "controlled_merge_rewrites"
                retrieved_docs, retrieval_debug = self._retrieve(sub_question, rewrites)
                rewrite_details = retrieval_debug.get("retrieval_by_query", [])

            # 关键变量：sub_answer 是当前轮证据压缩后的子结论，后续控制器会使用。
            sub_answer = self._synthesize_sub_answer(sub_question, retrieved_docs)

            round_item = {
                "iteration": i,
                "sub_question": sub_question,
                "subquestion_prompt": subquestion_prompt,
                "subquestion_raw_output": subquestion_raw_output,
                "rewrites": rewrites,
                "rewrite_filtering": rewrite_filtering_debug,
                "selection_strategy": selection_strategy,
                "selected_rewrite": selected_rewrite,
                "rewrite_count": len(rewrites),
                "rewrite_prompt_names": self.config.rewrite_prompt_names,
                "rewrite_selection_details": rewrite_details,
                "retrieval_debug": retrieval_debug,
                "retrieved_docs": retrieved_docs,
                "sub_answer": sub_answer,
            }
            rounds.append(round_item)

            control, control_prompt, control_raw_output = self._control_iteration(main_query, rounds, i)
            round_item["control_prompt"] = control_prompt
            round_item["control_raw_output"] = control_raw_output
            round_item["control"] = control

        return {
            "main_query": main_query,
            "config": {
                "corpus_name": self.config.corpus_name,
                "max_iterations": self.config.max_iterations,
                "retrieval_top_k": self.config.retrieval_top_k,
                "retrieval_buffer_k": self.config.retrieval_buffer_k,
                "retrieval_bm25_top_k": self.config.retrieval_bm25_top_k,
                "rewrite_prompt_names": self.config.rewrite_prompt_names,
                "experiment_mode": self.config.experiment_mode,
                "rewrite_n": self.config.rewrite_n,
                "include_original_query_in_baseline_retrieval": self.config.include_original_query_in_baseline_retrieval,
                "random_seed": self.config.random_seed,
                "prompt_modes": {
                    "subquestion": self.config.subquestion_prompt_mode,
                    "control": self.config.control_prompt_mode,
                    "rewrite": "named_templates",
                },
            },
            "rounds": rounds,
        }

    def _build_final_answer(self, main_query: str, rounds: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> str:
        """基于 QA 历史 + 检索文档生成最终答案（统一模板来源）。"""
        qa_history = [
            (item.get("sub_question", ""), item.get("sub_answer", ""))
            for item in rounds
            if item.get("sub_question", "")
        ]
        doc_texts = [d.get("paragraph_text", "") for d in documents if d.get("paragraph_text", "")]

        prompt = build_final_answer_prompt(
            query=main_query,
            qa_history=qa_history,
            documents=doc_texts,
        )

        return self._sample_one(
            self.general_llm,
            prompt=prompt,
            temperature=0.0,
            max_tokens=256,
            top_p=0.9,
            use_cache=True,
            cache_scope="final_answer",
        )

    def _compress_final_answer(self, main_query: str, answer: str) -> str:
        """压缩最终答案（统一模板来源）。"""
        prompt = build_compression_prompt(question=main_query, predict=answer)
        return self._sample_one(
            self.general_llm,
            prompt=prompt,
            temperature=0.0,
            max_tokens=128,
            top_p=0.9,
            use_cache=True,
            cache_scope="compress_answer",
        )

    def run_interface(self, main_query: str, trace: Dict[str, Any] | None = None) -> tuple[str, int, float, float]:
        """
        对齐 run_rag_chain.run_interface 风格输出：
        - compressed_predict
        - retrieved_times
        - precision
        - recall

        注：precision/recall 需要黄金文档才可计算，此处默认返回 0.0。
        若传入 trace，将直接复用该轨迹，避免重复执行 pipeline。
        """
        if trace is None:
            trace = self.run(main_query)
        rounds = trace.get("rounds", [])

        retrieved_times = len(rounds)

        # 汇总每轮检索文档并去重，作为最终答案阶段的统一证据池
        merged_docs: List[Dict[str, Any]] = []
        seen = set()
        for item in rounds:
            for d in item.get("retrieved_docs", []):
                key = d.get("paragraph_text", "").strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged_docs.append(d)

        final_answer = self._build_final_answer(main_query, rounds, merged_docs)
        compressed_predict = self._compress_final_answer(main_query, final_answer)

        precision = 0.0
        recall = 0.0
        return compressed_predict, retrieved_times, precision, recall
