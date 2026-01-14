"""
Semantic Scholar API client.

Good for citations, recommendations, and AI-related papers.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from litforge.clients.base import BaseClient
from litforge.models import (
    Author,
    Publication,
    PublicationType,
    SearchFilter,
)

logger = logging.getLogger(__name__)


class SemanticScholarClient(BaseClient):
    """
    Client for the Semantic Scholar API.
    
    Semantic Scholar provides excellent citation data and recommendations.
    Free tier: 100 requests/5 minutes without API key.
    
    API Docs: https://api.semanticscholar.org/
    """
    
    def __init__(self, api_key: str | None = None):
        """
        Initialize the Semantic Scholar client.
        
        Args:
            api_key: API key for higher rate limits
        """
        super().__init__(
            base_url="https://api.semanticscholar.org",
            rate_limit=1.0 if not api_key else 10.0,
        )
        self.api_key = api_key
    
    def _get_headers(self) -> dict[str, str] | None:
        """Get headers with API key if available."""
        if self.api_key:
            return {"x-api-key": self.api_key}
        return None
    
    def search(
        self,
        query: str,
        limit: int = 25,
        filters: SearchFilter | None = None,
    ) -> list[Publication]:
        """
        Search for papers.
        
        Args:
            query: Search query
            limit: Maximum results
            filters: Optional filters
            
        Returns:
            List of publications
        """
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "paperId,externalIds,title,abstract,year,authors,"
                     "venue,publicationDate,citationCount,openAccessPdf,"
                     "fieldsOfStudy",
        }
        
        if filters:
            if filters.year_from:
                params["year"] = f"{filters.year_from}-"
            if filters.year_to:
                if "year" in params:
                    params["year"] = params["year"][:-1] + f"-{filters.year_to}"
                else:
                    params["year"] = f"-{filters.year_to}"
            if filters.open_access_only:
                params["openAccessPdf"] = ""
        
        response = self._get(
            "/graph/v1/paper/search",
            params=params,
            headers=self._get_headers(),
        )
        
        papers = response.get("data", [])
        return [self._parse_paper(p) for p in papers if p]
    
    def get_paper(
        self,
        paper_id: str | None = None,
        doi: str | None = None,
    ) -> Publication | None:
        """
        Get a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            doi: DOI
            
        Returns:
            Publication or None
        """
        if paper_id:
            lookup_id = paper_id
        elif doi:
            lookup_id = f"DOI:{doi}"
        else:
            return None
        
        params = {
            "fields": "paperId,externalIds,title,abstract,year,authors,"
                     "venue,publicationDate,citationCount,openAccessPdf,"
                     "fieldsOfStudy,references,citations",
        }
        
        try:
            response = self._get(
                f"/graph/v1/paper/{lookup_id}",
                params=params,
                headers=self._get_headers(),
            )
            return self._parse_paper(response)
        except Exception as e:
            logger.debug(f"Semantic Scholar lookup failed: {e}")
            return None
    
    def get_citations(
        self,
        paper_id: str,
        limit: int = 50,
    ) -> list[Publication]:
        """Get papers that cite this paper."""
        params = {
            "fields": "paperId,externalIds,title,abstract,year,authors,"
                     "venue,citationCount",
            "limit": min(limit, 1000),
        }
        
        response = self._get(
            f"/graph/v1/paper/{paper_id}/citations",
            params=params,
            headers=self._get_headers(),
        )
        
        citations = response.get("data", [])
        return [
            self._parse_paper(c.get("citingPaper", {}))
            for c in citations
            if c.get("citingPaper")
        ]
    
    def get_references(
        self,
        paper_id: str,
        limit: int = 50,
    ) -> list[Publication]:
        """Get papers referenced by this paper."""
        params = {
            "fields": "paperId,externalIds,title,abstract,year,authors,"
                     "venue,citationCount",
            "limit": min(limit, 1000),
        }
        
        response = self._get(
            f"/graph/v1/paper/{paper_id}/references",
            params=params,
            headers=self._get_headers(),
        )
        
        references = response.get("data", [])
        return [
            self._parse_paper(r.get("citedPaper", {}))
            for r in references
            if r.get("citedPaper")
        ]
    
    def get_recommendations(
        self,
        paper_ids: list[str],
        limit: int = 20,
    ) -> list[Publication]:
        """
        Get paper recommendations based on positive examples.
        
        Args:
            paper_ids: List of paper IDs or DOIs as positive examples
            limit: Maximum recommendations
            
        Returns:
            List of recommended papers
        """
        # Build request body
        positive_ids = []
        for pid in paper_ids:
            if pid.startswith("10."):  # Looks like a DOI
                positive_ids.append(f"DOI:{pid}")
            else:
                positive_ids.append(pid)
        
        response = self._post(
            "/recommendations/v1/papers",
            json_data={
                "positivePaperIds": positive_ids[:5],  # API limit
                "negativePaperIds": [],
            },
            headers=self._get_headers(),
        )
        
        papers = response.get("recommendedPapers", [])[:limit]
        return [self._parse_paper(p) for p in papers if p]
    
    def _parse_paper(self, data: dict[str, Any]) -> Publication:
        """Parse a Semantic Scholar paper into a Publication."""
        # Parse authors
        authors = []
        for author in data.get("authors", []):
            authors.append(Author(
                name=author.get("name", "Unknown"),
                semantic_scholar_id=author.get("authorId"),
            ))
        
        # Get external IDs
        ext_ids = data.get("externalIds", {}) or {}
        
        # Parse date
        pub_date = None
        date_str = data.get("publicationDate")
        if date_str:
            try:
                pub_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                pass
        
        # Get PDF URL
        pdf_url = None
        oa_pdf = data.get("openAccessPdf")
        if oa_pdf:
            pdf_url = oa_pdf.get("url")
        
        # Generate internal ID from Semantic Scholar paper ID
        ss_id = data.get("paperId", "")
        internal_id = f"s2:{ss_id}" if ss_id else f"s2:{hash(data.get('title', ''))}"
        
        return Publication(
            id=internal_id,
            title=data.get("title", "Untitled"),
            authors=authors,
            abstract=data.get("abstract"),
            doi=ext_ids.get("DOI"),
            year=data.get("year"),
            publication_date=pub_date,
            venue=data.get("venue"),
            publication_type=PublicationType.ARTICLE,
            keywords=data.get("fieldsOfStudy") or [],
            citation_count=data.get("citationCount", 0),
            semantic_scholar_id=ss_id,
            pmid=ext_ids.get("PubMed"),
            pmcid=ext_ids.get("PubMedCentral"),
            arxiv_id=ext_ids.get("ArXiv"),
            pdf_url=pdf_url,
            is_open_access=pdf_url is not None,
            sources=["semantic_scholar"],
        )
