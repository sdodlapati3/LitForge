"""
Core data models for LitForge.
"""

from litforge.models.publication import (
    Publication,
    Author,
    Citation,
    Affiliation,
    PublicationType,
    AccessType,
)
from litforge.models.search import SearchQuery, SearchResult, SearchFilter
from litforge.models.network import CitationNetwork, NetworkNode, NetworkEdge, Cluster

__all__ = [
    # Publication models
    "Publication",
    "Author",
    "Citation",
    "Affiliation",
    "PublicationType",
    "AccessType",
    
    # Search models
    "SearchQuery",
    "SearchResult",
    "SearchFilter",
    
    # Network models
    "CitationNetwork",
    "NetworkNode",
    "NetworkEdge",
    "Cluster",
]
