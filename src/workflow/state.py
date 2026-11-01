from __future__ import annotations

from typing import Optional, TypedDict
from langchain_core.documents import Document


class SourceMetadata(TypedDict):
    doc_id: str
    source: str
    chunk_id: str
    heading: str


class RAGState(TypedDict):
    original_question: str
    top_k: int
    rewritten_query: str
    query_type: str
    retrieved_documents: list[Document]
    relevant_documents: list[Document]
    retry_count: int
    max_retries: int
    answer: str
    sources: list[SourceMetadata]
    generation_supported: bool
    error: Optional[str]
