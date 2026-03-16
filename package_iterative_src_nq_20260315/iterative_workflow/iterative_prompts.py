"""
Prompt templates for iterative sub-question generation and iteration control.

Design goals:
1. Generate one actionable sub-question each round
2. Use retrieval evidence and QA history to decide continue/stop
3. Keep outputs machine-parseable (strict JSON where needed)
"""


# =====================================================
# 1. 子问题生成 Prompt
# =====================================================

ITERATIVE_SUBQUESTION_PROMPT = """You are a multi-hop reasoning planner.
Given the main question and previous progress, propose exactly ONE next sub-question.

Main question:
{main_query}

History (JSON list of solved sub-questions and brief answers):
{history_json}

Rules:
1. Return one concise sub-question that helps solve the main question.
2. Use history to avoid repeating already-answered sub-questions.
3. Return plain text only, no markdown.

Next sub-question:
"""


# =====================================================
# 2. 迭代控制 Prompt（要求严格 JSON）
# =====================================================

ITERATION_CONTROL_PROMPT = """You are an iteration controller for a retrieval-augmented QA pipeline.
Decide whether another iteration is needed.

Main question:
{main_query}

Current iteration: {iteration}
Max iterations: {max_iterations}

History (JSON):
{history_json}

Retrieved evidence summary:
{evidence_summary}

Rules:
1. Decide with both history and retrieved evidence summary, not history alone.
2. Output CONTINUE with a concise reason for current progress.
4. Output STRICT JSON only (no markdown, no extra text):
{{"action":"CONTINUE","reason":"..."}}
"""
