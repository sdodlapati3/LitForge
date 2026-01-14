"""
🔥 LitForge - Forging Knowledge from Literature

An open-source Python library for unified scientific literature search,
retrieval, and knowledge synthesis.

Example:
    >>> from litforge import Forge
    >>> forge = Forge()
    >>> papers = forge.search("CRISPR mechanisms", limit=50)
    >>> forge.index(papers)
    >>> answer = forge.ask("What are the main CRISPR mechanisms?")
"""

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
    # Main interface
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
