"""
Prompt templates for query rewriting in RAG systems.

This file contains:
1. HyDE prompt (NeurIPS 2022)
2. Keyword-centric query rewrite prompt (Cloudflare AI Search official system prompt)

All prompts are sourced from peer-reviewed papers or official engineering documentation.
"""


# =====================================================
# 1. HyDE: Hypothetical Document Embeddings
# Source:
#   Gao et al., "HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels"
#   NeurIPS 2022
#   https://arxiv.org/abs/2212.10496
# =====================================================

HYDE_PROMPT = """generate a hypothetical passage that directly answers this question.

Question: {query}

Passage:
"""


# =====================================================
# 2. Keyword-Centric Query Rewrite (Concise Rewrite)
# Source:
#   Cloudflare AI Search - Official System Prompt
#   https://developers.cloudflare.com/ai-search/configuration/system-prompt/
# =====================================================

KEYWORD_REWRITE_PROMPT = """You are a search query optimizer for retrieval-augmented generation systems.
Your task is to reformulate user queries into more effective search terms.

Given a user's search query, you must:
1. Identify the core concepts and intent
2. Add relevant synonyms and related terms
3. Remove irrelevant filler words
4. Structure the query to emphasize key terms
5. Include technical or domain-specific terminology if applicable

Provide only the optimized search query without any explanations, greetings, or additional commentary.

User query: {query}
Optimized search query:
"""


RAFE_SFT_REWRITE_PROMPT = """You are an expert in query optimizer for retrieval-augmented generation systems.

Rewrite the following user query into a retrieval-optimized query.
The rewritten query should:
1. Preserve the original intent
2. Be clear and unambiguous
3. Improve retrievability in a knowledge base
4. Avoid adding new information

User query: {query}
Rewrited query:
"""