# Retrieval-Augmented Generation

## 检索增强生成

RAG 是 Retrieval-Augmented Generation 的缩写。检索增强生成会先从知识库选择相关证据，再让语言模型基于证据生成回答，从而减少脱离资料的猜测。

## Retrieval Pipeline

Retrieval selects relevant context before generation. A good retriever returns useful evidence with source metadata so that answers can be checked against the original document.

## Minimal Example

```python
def retrieve(query: str) -> list[str]:
    # A real implementation ranks indexed knowledge chunks.
    return ["relevant context"]
```
