"""
🔥 LitForge - Forging Knowledge from Literature

An open-source Python library for unified scientific literature search,
retrieval, and knowledge synthesis.

SIMPLE API (recommended):
    >>> import litforge
    >>> 
    >>> # Search for papers - that's it!
    >>> papers = litforge.search("CRISPR gene editing")
    >>> 
    >>> # Look up by DOI
    >>> paper = litforge.lookup("10.1038/nature14539")
    >>> 
    >>> # Get citations
    >>> cites = litforge.citations("10.1126/science.1225829")

ADVANCED API:
    >>> from litforge import Forge
    >>> forge = Forge()
    >>> papers = forge.search("CRISPR mechanisms", limit=50)
    >>> forge.index(papers)
    >>> answer = forge.ask("What are the main CRISPR mechanisms?")

AGENT INTEGRATION:
    >>> from litforge import get_tools
    >>> tools = get_tools()  # Returns callable tools for any agent system
"""

# Simple API (recommended for most users)
from litforge.api import (
    search,
    lookup,
    citations,
    references,
    get_tools,
    Paper,
    LitForgeClient,
)

# Advanced API
from litforge.core.forge import Forge
from litforge.models.publication import Publication, Author, Citation
from litforge.models.search import SearchQuery, SearchResult
from litforge.models.network import CitationNetwork

# Services (for advanced usage)
from litforge.services.discovery import DiscoveryService
from litforge.services.retrieval import RetrievalService
from litforge.services.citation import CitationService
from litforge.services.knowledge import KnowledgeService
from litforge.services.qa import QAService

__version__ = "0.1.0"
__author__ = "Sribharath Kainkaryam"
__email__ = "sdodlapati3@gmail.com"

__all__ = [
    # Simple API (recommended)
    "search",
    "lookup",
    "citations",
    "references",
    "get_tools",
    "Paper",
    "LitForgeClient",
    
    # Advanced API
    "Forge",
    
    # Models
    "Publication",
    "Author", 
    "Citation",
    "SearchQuery",
    "SearchResult",
    "CitationNetwork",
    
    # Services
    "DiscoveryService",
    "RetrievalService",
    "CitationService",
    "KnowledgeService",
    "QAService",
]
