# Iterative Rewrite + Retrieve Workflow

## 功能说明

该模块实现一个独立迭代流程（不改动原有代码）：

1. 输入原始问题 `q`
2. 生成下一步子问题（迭代 prompt）
3. 对子问题执行查询重写（使用 rewrite 专用微调模型，支持 3 种模板）
4. 对重写后的查询进行 Elasticsearch 检索并去重
5. （可选）使用 BGE reranker 对“原始问题 + 去重文档”重新打分，取 top-k
5. 基于迭代控制 prompt 判断是否继续

## 文件结构

- `run_iterative_rewrite_retrieve.py`：CLI 入口
- `iterative_workflow/pipeline.py`：核心流程
- `iterative_workflow/elasticsearch_retriever.py`：检索器
- `iterative_workflow/iterative_prompts.py`：默认 prompt
- `iterative_workflow/prompts/subquestion_prompt.txt`：可编辑子问题 prompt
- `iterative_workflow/prompts/control_prompt.txt`：可编辑控制 prompt

## 快速运行

```bash
python run_iterative_rewrite_retrieve.py \
  --query "Who is the mother of the director of Polish-Russian War film?" \
  --corpus-name 2wikimultihopqa \
  --max-iterations 4 \
  --top-k 5 \
  --rewrite-prompts hyde keyword_rewrite rafe_sft_rewrite
```

## 使用你提供的迭代 Prompt

```bash
python run_iterative_rewrite_retrieve.py \
  --query "..." \
  --corpus-name hotpotpqa \
  --subquestion-prompt-file iterative_workflow/prompts/subquestion_prompt.txt \
  --control-prompt-file iterative_workflow/prompts/control_prompt.txt

## 启用 Reranker（原始查询打分）

```bash
python run_iterative_rewrite_retrieve.py \
  --query "..." \
  --corpus-name hotpotpqa \
  --enable-reranker \
  --reranker-model BAAI/bge-reranker-base \
  --reranker-top-k 5
```

说明：`retrieved_docs` 中会新增 `reranker_score` 字段，并按该分数降序返回 top-k。
```

## 输出

默认输出 JSON 到控制台；也可写文件：

```bash
python run_iterative_rewrite_retrieve.py \
  --query "..." \
  --output-json outputs/iterative/result.json
```
