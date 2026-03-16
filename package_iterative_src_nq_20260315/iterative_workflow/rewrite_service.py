from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

from prompts import prompt_manager


# This module owns rewrite-side behavior:
# - rendering rewrite prompts
# - sampling strategy by experiment mode
# - normalization and quality filtering
# - structured debug traces for kept/dropped candidates

class RewriteService:
    """Generate and filter rewrite candidates for one sub-question round."""

    def __init__(
        self,
        *,
        config: Any,
        rewrite_llm: Any,
        sample_one: Callable[..., Any],
        sample_many: Callable[..., List[str]],
    ):
        # sample_one/sample_many are injected from pipeline so caching behavior
        # remains centralized and consistent with other LLM calls.
        self.config = config
        self.rewrite_llm = rewrite_llm
        self._sample_one = sample_one
        self._sample_many = sample_many

        self.last_filtering_debug: Dict[str, Any] = {
            "source_query": "",
            "kept": [],
            "dropped": [],
        }

    def _canonicalize_query_text(self, text: str) -> str:
        """Normalize query text for stable per-template dedup keys."""
        lowered = (text or "").lower()
        lowered = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _tokenize_for_overlap(self, text: str) -> List[str]:
        """Tokenization used by lexical overlap checks."""
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", (text or "").lower())

    def _evaluate_rewrite_quality(self, rewrite: str, original_query: str) -> tuple[bool, Dict[str, Any]]:
        """Filter noisy rewrites before retrieval.

        Intuition:
        - reject empty/overly long/explanatory outputs
        - optionally enforce lexical-overlap floor when enabled
        """
        rw = (rewrite or "").strip()
        original = (original_query or "").strip()

        info: Dict[str, Any] = {
            "rewrite": rw,
            "reason": "",
            "char_len": len(rw),
            "token_overlap": 0.0,
        }

        if not rw:
            info["reason"] = "empty"
            return False, info

        max_chars = int(getattr(self.config, "rewrite_max_chars", 320))
        if len(rw) > max_chars:
            info["reason"] = "too_long"
            return False, info

        explanation_patterns = [
            r"\bhere is\b",
            r"\bi rewrote\b",
            r"\bthis query\b",
            r"\bbecause\b",
            r"\bthe rewritten query\b",
            r"\bto find\b.*\byou can\b",
        ]
        low = rw.lower()
        if any(re.search(p, low) for p in explanation_patterns):
            info["reason"] = "explanatory_style"
            return False, info

        orig_tokens = set(self._tokenize_for_overlap(original))
        rw_tokens = set(self._tokenize_for_overlap(rw))
        if orig_tokens and rw_tokens:
            overlap = len(orig_tokens & rw_tokens) / float(max(1, len(orig_tokens)))
        else:
            overlap = 0.0
        info["token_overlap"] = round(overlap, 4)

        # overlap filter is intentionally optional and disabled by default.
        if self.config.enable_overlap_filter and overlap < float(self.config.min_token_overlap):
            info["reason"] = "low_lexical_overlap"
            return False, info

        info["reason"] = "kept"
        return True, info

    def _generate_bestofn_outputs(self, *, prompt_name: str, prompt: str, sub_question: str, n: int) -> List[str]:
        """Generate best-of-n outputs with optional diversity boosting.

        When diversity mode is enabled, this method increases temperature
        step-by-step and keeps sampling until at least M unique normalized
        rewrites are observed (or retry budget is exhausted).
        """
        use_rewrite_cache = bool(
            getattr(self.config, "enable_llm_cache", True)
            and getattr(self.config, "enable_rewrite_cache", False)
        )

        outputs = [
            self._sample_one(
                self.rewrite_llm,
                prompt=prompt,
                temperature=0.0,
                max_tokens=self.config.max_tokens_rewrite,
                top_p=1.0,
                use_cache=use_rewrite_cache,
                cache_scope=f"rewrite_single:{prompt_name}",
            )
        ]

        stochastic_target = max(0, n - 1)
        if stochastic_target <= 0:
            return outputs

        enable_diversity = bool(getattr(self.config, "enable_rewrite_diversity_boost", False))
        min_unique = int(getattr(self.config, "rewrite_min_unique", 0))
        if not enable_diversity or min_unique <= 0:
            outputs.extend(
                self._sample_many(
                    self.rewrite_llm,
                    prompt=prompt,
                    n=stochastic_target,
                    temperature=float(getattr(self.config, "rewrite_temperature", 0.7)),
                    max_tokens=self.config.max_tokens_rewrite,
                    top_p=0.95,
                    use_cache=use_rewrite_cache,
                    cache_scope=f"rewrite_batch:{prompt_name}",
                )
            )
            return outputs

        # target_unique follows user constraint M < N; we clamp to [1, n-1].
        target_unique = max(1, min(min_unique, max(1, n - 1)))
        current_temp = float(getattr(self.config, "rewrite_temperature", 0.7))
        temp_step = float(getattr(self.config, "rewrite_diversity_temperature_step", 0.15))
        max_temp = float(getattr(self.config, "rewrite_diversity_max_temperature", 1.3))
        max_rounds = max(0, int(getattr(self.config, "rewrite_diversity_max_rounds", 4)))

        sampled: List[str] = []
        rounds = 0
        while len(sampled) < stochastic_target and rounds <= max_rounds:
            need = stochastic_target - len(sampled)
            sampled.extend(
                self._sample_many(
                    self.rewrite_llm,
                    prompt=prompt,
                    n=need,
                    temperature=current_temp,
                    max_tokens=self.config.max_tokens_rewrite,
                    top_p=0.95,
                    use_cache=use_rewrite_cache,
                    cache_scope=f"rewrite_batch:{prompt_name}:t={current_temp:.2f}:r={rounds}",
                )
            )

            # Count unique normalized rewrites over sampled-only candidates.
            unique_norm = set()
            for text in sampled:
                normalized = self._normalize_rewrite_text(str(text), fallback_query=sub_question)
                stable_norm = self._canonicalize_query_text(normalized)
                if stable_norm:
                    unique_norm.add(stable_norm)

            if len(unique_norm) >= target_unique:
                break

            if current_temp >= max_temp:
                break

            current_temp = min(max_temp, current_temp + temp_step)
            rounds += 1

        outputs.extend(sampled[:stochastic_target])
        return outputs

    def _normalize_rewrite_text(self, text: str, fallback_query: str) -> str:
        """Normalize model output into a single retrieval-ready query string."""
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        cleaned = ""

        # Keep first informative line and drop explanation-style preambles.
        for line in lines:
            line = re.sub(r"^[-*•]+\s*", "", line)
            line = re.sub(r"^\d+[\.)]\s*", "", line)
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
                break

        cleaned = re.sub(r"\s+", " ", cleaned).strip('"\'` ')
        return cleaned if cleaned else fallback_query

    def rewrite_candidates(self, sub_question: str) -> List[Dict[str, str]]:
        """Generate rewrite candidates according to experiment mode.

        Mode policy:
        - baseline/greedy: 1 deterministic sample per template
        - best_of_n: 1 deterministic + (n-1) stochastic samples per template
        """
        mode = self.config.experiment_mode
        n = max(1, int(self.config.rewrite_n))

        candidates: List[Dict[str, str]] = []
        seen_by_prompt: Dict[str, set[str]] = {}
        kept_debug: List[Dict[str, Any]] = []
        dropped_debug: List[Dict[str, Any]] = []

        for prompt_name in self.config.rewrite_prompt_names or []:
            prompt = prompt_manager.render(prompt_name, query=sub_question)
            use_rewrite_cache = bool(
                getattr(self.config, "enable_llm_cache", True)
                and getattr(self.config, "enable_rewrite_cache", False)
            )

            if mode in {"baseline", "greedy"}:
                outputs = [
                    self._sample_one(
                        self.rewrite_llm,
                        prompt=prompt,
                        temperature=0.0,
                        max_tokens=self.config.max_tokens_rewrite,
                        top_p=1.0,
                        use_cache=use_rewrite_cache,
                        cache_scope=f"rewrite_single:{prompt_name}",
                    )
                ]
            elif mode == "best_of_n":
                outputs = self._generate_bestofn_outputs(
                    prompt_name=prompt_name,
                    prompt=prompt,
                    sub_question=sub_question,
                    n=n,
                )
            else:
                raise ValueError(f"Unsupported experiment_mode: {mode}")

            for text in outputs:
                normalized = self._normalize_rewrite_text(str(text), fallback_query=sub_question)
                stable_norm = self._canonicalize_query_text(normalized)
                if not stable_norm:
                    dropped_debug.append(
                        {
                            "prompt_name": prompt_name,
                            "raw_rewrite": str(text).strip(),
                            "normalized_rewrite": normalized,
                            "drop_reason": "empty_after_normalization",
                        }
                    )
                    continue

                keep, quality = self._evaluate_rewrite_quality(normalized, sub_question)
                if not keep:
                    dropped_debug.append(
                        {
                            "prompt_name": prompt_name,
                            "raw_rewrite": str(text).strip(),
                            "normalized_rewrite": normalized,
                            "drop_reason": quality.get("reason", "quality_filter"),
                            "quality": quality,
                        }
                    )
                    continue

                # Dedup only within each prompt template, not across templates.
                prompt_seen = seen_by_prompt.setdefault(prompt_name, set())
                if stable_norm in prompt_seen:
                    dropped_debug.append(
                        {
                            "prompt_name": prompt_name,
                            "raw_rewrite": str(text).strip(),
                            "normalized_rewrite": normalized,
                            "drop_reason": "duplicate_within_prompt",
                            "stable_norm": stable_norm,
                        }
                    )
                    continue

                prompt_seen.add(stable_norm)
                candidates.append(
                    {
                        "prompt_name": prompt_name,
                        "prompt_text": prompt,
                        "rewrite": str(text).strip(),
                        "normalized_rewrite": normalized,
                        "stable_norm": stable_norm,
                        "quality": quality,
                    }
                )
                kept_debug.append(
                    {
                        "prompt_name": prompt_name,
                        "raw_rewrite": str(text).strip(),
                        "normalized_rewrite": normalized,
                        "quality": quality,
                    }
                )

        # Expose full filtering trace for downstream debug/reporting.
        self.last_filtering_debug = {
            "source_query": sub_question,
            "mode": mode,
            "requested_n": n,
            "kept": kept_debug,
            "dropped": dropped_debug,
        }
        return candidates
