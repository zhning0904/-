# src/prompts/prompt_manager.py
import string

# 导入你原文件里的模板常量（原文件不需要改）
from prompts.rewrite_prompt import (
    HYDE_PROMPT,
    KEYWORD_REWRITE_PROMPT,
    RAFE_SFT_REWRITE_PROMPT,
)

# 用名字做一个轻量注册表
PROMPTS = {
    "hyde": HYDE_PROMPT,
    "keyword_rewrite": KEYWORD_REWRITE_PROMPT,
    "rafe_sft_rewrite": RAFE_SFT_REWRITE_PROMPT,
}


def _placeholders(tpl: str) -> set[str]:
    """提取模板里 {xxx} 占位符名字"""
    fmt = string.Formatter()
    names = set()
    for _, field, __, ___ in fmt.parse(tpl):
        if field:
            names.add(field)
    return names


def render(name: str, **kwargs) -> str:
    """
    用法：render("hyde", query="xxx")
    """
    if name not in PROMPTS:
        raise KeyError(f"Unknown prompt name: {name}. Available: {list(PROMPTS.keys())}")

    tpl = PROMPTS[name]
    need = _placeholders(tpl)
    missing = need - set(kwargs.keys())
    if missing:
        raise ValueError(f"Missing variables for '{name}': {sorted(missing)}")

    return tpl.format(**kwargs)


def list_prompts() -> list[str]:
    return list(PROMPTS.keys())
