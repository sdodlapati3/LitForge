"""
Service layer for LitForge.

Services provide the core business logic, orchestrating clients and stores.
"""

from litforge.services.discovery import DiscoveryService
from litforge.services.retrieval import RetrievalService
from litforge.services.citation import CitationService
from litforge.services.knowledge import KnowledgeService
from litforge.services.qa import QAService, QAResponse, ChatResponse, ComparisonResult
from litforge.services.scoring import EnsembleScorer, ScoredPaper, score_and_verify, verify_with_llm
from litforge.services.rag_search import (
    rag_search_pipeline, 
    scout_search, 
    expand_with_rag, 
    multi_api_search,
    deduplicate_papers,
    ScoutResult,
)
from litforge.services.hybrid_search import (
    HybridSearchService,
    HybridResult,
    hybrid_search,
)

__all__ = [
    "DiscoveryService",
    "RetrievalService",
    "CitationService",
    "KnowledgeService",
    "QAService",
    "QAResponse",
    "ChatResponse",
    "ComparisonResult",
    "EnsembleScorer",
    "ScoredPaper",
    "score_and_verify",
    "verify_with_llm",
    # RAG search
    "rag_search_pipeline",
    "scout_search",
    "expand_with_rag",
    "multi_api_search",
    "deduplicate_papers",
    "ScoutResult",
    # Hybrid search
    "HybridSearchService",
    "HybridResult",
    "hybrid_search",
]
