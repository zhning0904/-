from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from iterative_workflow.reranker import BGEReranker


# This module encapsulates retrieval-time logic that used to live in pipeline:
# - query-group retrieval and fusion
# - reranker lifecycle (including offline fallback)
# - per-round reranker score cache reuse
# - best-of-n rewrite competition by reranker top-k sum

class RetrievalService:
    """Retrieval and ranking service used by pipeline orchestration.

    Design goal: keep pipeline focused on control flow while this class owns
    all retrieval-side policies and constraints.
    """

    def __init__(self, *, config: Any, retriever: Any, root_dir: Path):
        # config: runtime knobs (top_k, reranker model, etc.)
        # retriever: backend adapter with retrieve_paragraphs(...)
        # root_dir: workspace root used for offline model snapshot fallback
        self.config = config
        self.retriever = retriever
        self.root_dir = root_dir
        self.reranker: BGEReranker | None = None

    def preload_reranker(self) -> None:
        """Optionally warm up reranker at startup.

        This preserves previous behavior when caller wants failures to happen
        early rather than during first retrieval.
        """
        self._ensure_reranker()

    def _merge_dedup_docs(self, docs_a: List[Dict[str, Any]], docs_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge two doc lists and deduplicate by normalized paragraph text."""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for d in docs_a + docs_b:
            key = d.get("paragraph_text", "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(d)
        return merged

    def _ensure_reranker(self) -> BGEReranker:
        """Return a reranker instance with offline-safe fallback.

        Fallback order when HF download is unavailable:
        1) temp/models--BAAI--bge-reranker-base/refs/main -> snapshots/<id>
        2) first available snapshot under snapshots/*
        """
        if self.reranker is None:
            try:
                self.reranker = BGEReranker(
                    model_name=self.config.reranker_model_name,
                    device=self.config.reranker_device,
                    batch_size=self.config.reranker_batch_size,
                )
            except Exception:
                # Offline path: use local HF cache snapshot in workspace.
                cache_root = self.root_dir / "temp" / "models--BAAI--bge-reranker-base"
                snapshot_path: Path | None = None
                if cache_root.exists():
                    ref_main = cache_root / "refs" / "main"
                    if ref_main.exists():
                        snap_id = ref_main.read_text(encoding="utf-8").strip()
                        cand = cache_root / "snapshots" / snap_id
                        if cand.exists():
                            snapshot_path = cand
                    if snapshot_path is None:
                        snaps = sorted((cache_root / "snapshots").glob("*"))
                        if snaps:
                            snapshot_path = snaps[0]

                if snapshot_path is None:
                    raise RuntimeError(
                        "Reranker is required for fusion ranking, but model init failed and no local snapshot was found."
                    )

                self.reranker = BGEReranker(
                    model_name=str(snapshot_path),
                    device=self.config.reranker_device,
                    batch_size=self.config.reranker_batch_size,
                )
        return self.reranker

    def rerank_with_cache(
        self,
        *,
        original_query: str,
        docs: List[Dict[str, Any]],
        score_cache: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Rerank docs by original query with per-round cache reuse.

        Cache key is normalized paragraph_text. For the same sub-question round,
        repeated documents will be scored once and then reused across
        baseline/greedy/best_of_n paths.
        """
        if not docs:
            return []

        reranker = self._ensure_reranker()
        uncached_docs: List[Dict[str, Any]] = []
        uncached_passages: List[str] = []

        for d in docs:
            key = str(d.get("paragraph_text", "")).strip().lower()
            if not key:
                continue
            if key not in score_cache:
                uncached_docs.append(d)
                uncached_passages.append(str(d.get("paragraph_text", "")))

        # Only score passages not seen in this round cache.
        if uncached_passages:
            new_scores = reranker.compute_scores(original_query, uncached_passages)
            for d, s in zip(uncached_docs, new_scores):
                key = str(d.get("paragraph_text", "")).strip().lower()
                score_cache[key] = float(s)

        reranked: List[Dict[str, Any]] = []
        for d in docs:
            key = str(d.get("paragraph_text", "")).strip().lower()
            if not key:
                continue
            enriched = dict(d)
            enriched["reranker_score"] = float(score_cache.get(key, float("-inf")))
            reranked.append(enriched)

        reranked.sort(key=lambda x: x.get("reranker_score", float("-inf")), reverse=True)
        return reranked

    def retrieve_fused(self, original_query: str, rewrites: List[Dict[str, str]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run retrieval fusion for one round and return reranked top docs.

        Semantics:
        - original_query is the current round sub-question
        - final ranking always uses original_query as reranker query
        - rewrites only expand recall candidates
        """
        merged: List[Dict[str, Any]] = []
        retrieval_by_query: List[Dict[str, Any]] = []

        query_items: List[Dict[str, str]] = []
        if self.config.include_original_query_in_baseline_retrieval:
            query_items.append({"prompt_name": "__original__", "query_text": original_query})

        for item in rewrites:
            query_items.append(
                {
                    "prompt_name": item.get("prompt_name", ""),
                    "query_text": item.get("normalized_rewrite", item.get("rewrite", "")),
                }
            )

        for item in query_items:
            query_text = item["query_text"]
            docs = self.retriever.retrieve_paragraphs(
                corpus_name=self.config.corpus_name,
                query_text=query_text,
                max_hits_count=self.config.retrieval_bm25_top_k,
                max_buffer_count=self.config.retrieval_buffer_k,
            )
            retrieval_by_query.append(
                {
                    "prompt_name": item.get("prompt_name", ""),
                    "query_text": query_text,
                    "docs": docs,
                }
            )

        query_count = max(1, len(retrieval_by_query))
        per_query_keep = max(1, self.config.retrieval_bm25_top_k)

        seen = set()
        for group in retrieval_by_query:
            query_text = group.get("query_text", "")
            prompt_name = group.get("prompt_name", "")
            docs = group.get("docs", [])
            for rank, d in enumerate(docs[:per_query_keep], start=1):
                key = d.get("paragraph_text", "").strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)

                # blended_score is diagnostics only; final order is reranker_score.
                base_score = float(d.get("score", 0.0))
                rank_bonus = 1.0 / float(rank)
                source_bonus = 0.1 if prompt_name == "__original__" else 0.0

                merged.append(
                    {
                        "query_used": query_text,
                        "prompt_name": prompt_name,
                        "title": d.get("title", ""),
                        "paragraph_text": d.get("paragraph_text", ""),
                        "score": base_score,
                        "blended_score": base_score + rank_bonus + source_bonus,
                        "url": d.get("url", ""),
                    }
                )

        if not merged:
            return [], {
                "retrieval_by_query": retrieval_by_query,
                "reranker_query": original_query,
                "reranker_enabled": True,
            }

        reranked = self.rerank_with_cache(
            original_query=original_query,
            docs=merged,
            score_cache={},
        )
        top_k = self.config.reranker_top_k if self.config.reranker_top_k > 0 else self.config.retrieval_top_k
        top_docs = reranked[:top_k]
        return top_docs, {
            "retrieval_by_query": retrieval_by_query,
            "reranker_query": original_query,
            "reranker_enabled": True,
            "merge_strategy": "per_query_top_then_dedup_then_rerank",
            "per_query_keep": per_query_keep,
            "query_count": query_count,
        }

    def select_best_rewrite(
        self,
        *,
        original_query: str,
        candidates: List[Dict[str, str]],
        score_cache: Dict[str, float] | None = None,
    ) -> tuple[Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Pick the best rewrite candidate using reranker top-k sum.

        For each rewrite candidate:
        1) retrieve original_query docs
        2) retrieve rewrite docs
        3) merge + dedup
        4) rerank with original_query and compute top-k sum
        """
        if not candidates:
            fallback = {"prompt_name": "fallback", "rewrite": original_query}
            return fallback, [], []

        cache = score_cache if score_cache is not None else {}

        # Shared retrieval for original_query: reused by all candidates.
        docs_q = self.retriever.retrieve_paragraphs(
            corpus_name=self.config.corpus_name,
            query_text=original_query,
            max_hits_count=self.config.retrieval_bm25_top_k,
            max_buffer_count=self.config.retrieval_buffer_k,
        )

        best = None
        best_docs: List[Dict[str, Any]] = []
        best_score = float("-inf")
        details: List[Dict[str, Any]] = []

        for cand in candidates:
            rw = cand.get("normalized_rewrite", cand["rewrite"])
            docs_rw = self.retriever.retrieve_paragraphs(
                corpus_name=self.config.corpus_name,
                query_text=rw,
                max_hits_count=self.config.retrieval_bm25_top_k,
                max_buffer_count=self.config.retrieval_buffer_k,
            )
            merged = self._merge_dedup_docs(docs_q, docs_rw)

            if not merged:
                details.append(
                    {
                        "prompt_name": cand.get("prompt_name", ""),
                        "rewrite": rw,
                        "score": float("-inf"),
                        "top_k_sum": float("-inf"),
                        "docs_original": docs_q,
                        "docs_rewrite": docs_rw,
                        "merged_count": 0,
                    }
                )
                continue

            enriched = self.rerank_with_cache(
                original_query=original_query,
                docs=merged,
                score_cache=cache,
            )
            top_k = self.config.reranker_top_k if self.config.reranker_top_k > 0 else self.config.retrieval_top_k
            top_docs = enriched[:top_k]
            top_sum = float(sum(x.get("reranker_score", 0.0) for x in top_docs))

            details.append(
                {
                    "rewrite": rw,
                    "prompt_name": cand.get("prompt_name", ""),
                    "quality": cand.get("quality", {}),
                    "score": top_sum,
                    "top_k_sum": top_sum,
                    "docs_original": docs_q,
                    "docs_rewrite": docs_rw,
                    "merged_count": len(merged),
                    "top_docs": top_docs,
                }
            )

            if top_sum > best_score:
                best_score = top_sum
                best = cand
                best_docs = top_docs

        if best is None:
            best = candidates[0]
        return best, best_docs, details
