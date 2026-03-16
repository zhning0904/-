from __future__ import annotations

from typing import List, Optional

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - optional dependency
    CrossEncoder = None


class BGEReranker:
    """
    轻量 BGE Cross-Encoder 重排序器。

    用途：
    - 输入 query + 文档列表
    - 输出每个文档与 query 的相关性分数
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        if CrossEncoder is None:
            raise ImportError(
                "Missing dependency 'sentence_transformers'. Install it to enable reranking."
            )

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device)

    # ---------------------------------------------------------
    #                核心推理函数
    # ---------------------------------------------------------
    def compute_scores(self, query: str, passages: List[str]) -> List[float]:
        """批量计算 query 与 passages 的相关性分数。"""
        if not passages:
            return []

        pairs = [(query, p) for p in passages]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # 统一转换为 Python float，便于后续 JSON 序列化
        return [float(x) for x in scores]
