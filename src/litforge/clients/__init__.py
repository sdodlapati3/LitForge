"""API clients for various literature data sources."""

from litforge.clients.openalex import OpenAlexClient
from litforge.clients.semantic_scholar import SemanticScholarClient
from litforge.clients.pubmed import PubMedClient
from litforge.clients.arxiv import ArxivClient
from litforge.clients.crossref import CrossrefClient
from litforge.clients.unpaywall import UnpaywallClient

__all__ = [
    "OpenAlexClient",
    "SemanticScholarClient",
    "PubMedClient",
    "ArxivClient",
    "CrossrefClient",
    "UnpaywallClient",
]
