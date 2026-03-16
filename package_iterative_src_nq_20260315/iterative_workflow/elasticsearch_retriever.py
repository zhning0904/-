from collections import OrderedDict
from typing import Dict, List

try:
    from elasticsearch import Elasticsearch
except ImportError:  # pragma: no cover - optional dependency in editor diagnostics
    Elasticsearch = None


class ElasticsearchRetriever:
    """
    轻量 Elasticsearch 检索器。

    说明：
    - bool/must 类似 AND
    - bool/should 类似 OR
    - 这里使用 multi_match 在 title + paragraph_text 上联合检索
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        username: str = "elastic",
        password: str = "246247",
        timeout: int = 30,
    ):
        if Elasticsearch is None:
            raise ImportError(
                "Missing dependency 'elasticsearch'. Please install it before running retrieval."
            )

        # 兼容 ES Python Client v8/v7 的初始化参数
        try:
            self._es = Elasticsearch(
                [{"host": host, "port": port, "scheme": "http"}],
                basic_auth=(username, password),
                request_timeout=timeout,
            )
        except TypeError:
            self._es = Elasticsearch(
                [{"host": host, "port": port}],
                http_auth=(username, password),
                timeout=timeout,
            )

    def retrieve_paragraphs(
        self,
        corpus_name: str,
        query_text: str,
        max_hits_count: int = 10,
        max_buffer_count: int = 100,
    ) -> List[Dict]:
        """
        从指定索引中检索与 query_text 最相关的段落。

        Args:
            corpus_name: Elasticsearch 索引名
            query_text: 查询文本
            max_hits_count: 最终返回条数
            max_buffer_count: 初始候选条数

        Returns:
            标准化检索结果列表
        """

        query = {
            # 先取较大的候选池，后续在代码侧去重再截断
            "size": max_buffer_count,
            "_source": ["title", "paragraph_text", "url"],
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["title^2", "paragraph_text"],
                    "type": "best_fields",
                }
            },
        }

        result = self._es.search(index=corpus_name, body=query)
        hits = result.get("hits", {}).get("hits", [])
        if not hits:
            return []

        # 去重：避免同段落重复进入候选池
        text2retrieval = OrderedDict()
        for item in hits:
            # 用段落文本做 key，保持“首次出现优先”并去掉大小写差异
            text = item.get("_source", {}).get("paragraph_text", "").strip().lower()
            if text:
                text2retrieval[text] = item

        retrieval = list(text2retrieval.values())
        retrieval = sorted(retrieval, key=lambda e: e.get("_score", 0.0), reverse=True)
        retrieval = retrieval[:max_hits_count]

        formatted: List[Dict] = []
        for item in retrieval:
            src = item.get("_source", {})
            formatted.append(
                {
                    "title": src.get("title", ""),
                    "paragraph_text": src.get("paragraph_text", ""),
                    "url": src.get("url", ""),
                    "score": item.get("_score", 0.0),
                    "corpus_name": corpus_name,
                }
            )

        return formatted
