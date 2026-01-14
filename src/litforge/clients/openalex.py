"""
OpenAlex API client.

OpenAlex is the primary data source - comprehensive, free, and well-maintained.
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


class OpenAlexClient(BaseClient):
    """
    Client for the OpenAlex API.
    
    OpenAlex provides comprehensive bibliographic data for over 200M works.
    It's free, open, and doesn't require authentication (but email is polite).
    
    API Docs: https://docs.openalex.org/
    """
    
    def __init__(self, email: str | None = None):
        """
        Initialize the OpenAlex client.
        
        Args:
            email: Email for polite pool (higher rate limit)
        """
        super().__init__(
            base_url="https://api.openalex.org",
            rate_limit=10.0 if email else 1.0,  # Polite pool = 10/sec
        )
        self.email = email
    
    def _get_params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Add email to params if provided."""
        params = params or {}
        if self.email:
            params["mailto"] = self.email
        return params
    
    def search(
        self,
        query: str,
        limit: int = 25,
        filters: SearchFilter | None = None,
    ) -> list[Publication]:
        """
        Search for works in OpenAlex.
        
        Args:
            query: Search query (searches title, abstract, fulltext)
            limit: Maximum results
            filters: Optional filters
            
        Returns:
            List of publications
        """
        params = self._get_params({
            "search": query,
            "per_page": min(limit, 200),
        })
        
        # Build filter string
        filter_parts = []
        
        if filters:
            if filters.year_from:
                filter_parts.append(f"publication_year:>{filters.year_from - 1}")
            if filters.year_to:
                filter_parts.append(f"publication_year:<{filters.year_to + 1}")
            if filters.publication_types:
                type_map = {
                    "article": "article",
                    "review": "review",
                    "preprint": "preprint",
                    "book_chapter": "book-chapter",
                    "book": "book",
                }
                types = [type_map.get(t, t) for t in filters.publication_types]
                filter_parts.append(f"type:{'|'.join(types)}")
            if filters.open_access_only:
                filter_parts.append("is_oa:true")
            if filters.has_full_text:
                filter_parts.append("has_fulltext:true")
        
        if filter_parts:
            params["filter"] = ",".join(filter_parts)
        
        response = self._get("/works", params=params)
        
        works = response.get("results", [])
        return [self._parse_work(w) for w in works]
    
    def get_paper(
        self,
        openalex_id: str | None = None,
        doi: str | None = None,
    ) -> Publication | None:
        """
        Get a specific paper.
        
        Args:
            openalex_id: OpenAlex ID (e.g., "W2741809807")
            doi: DOI
            
        Returns:
            Publication or None
        """
        if openalex_id:
            endpoint = f"/works/{openalex_id}"
        elif doi:
            # Normalize DOI
            doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
            endpoint = f"/works/https://doi.org/{doi_clean}"
        else:
            return None
        
        try:
            response = self._get(endpoint, params=self._get_params())
            return self._parse_work(response)
        except Exception as e:
            logger.debug(f"OpenAlex lookup failed: {e}")
            return None
    
    def get_citations(
        self,
        openalex_id: str,
        limit: int = 50,
    ) -> list[Publication]:
        """Get works that cite this paper."""
        params = self._get_params({
            "filter": f"cites:{openalex_id}",
            "per_page": min(limit, 200),
        })
        
        response = self._get("/works", params=params)
        works = response.get("results", [])
        return [self._parse_work(w) for w in works]
    
    def get_references(
        self,
        openalex_id: str,
        limit: int = 50,
    ) -> list[Publication]:
        """Get works referenced by this paper."""
        params = self._get_params({
            "filter": f"cited_by:{openalex_id}",
            "per_page": min(limit, 200),
        })
        
        response = self._get("/works", params=params)
        works = response.get("results", [])
        return [self._parse_work(w) for w in works]
    
    def _parse_work(self, data: dict[str, Any]) -> Publication:
        """Parse an OpenAlex work into a Publication."""
        # Parse authors
        authors = []
        for authorship in data.get("authorships", []):
            author_data = authorship.get("author", {})
            if author_data:
                authors.append(Author(
                    name=author_data.get("display_name", "Unknown"),
                    openalex_id=author_data.get("id"),
                    orcid=author_data.get("orcid"),
                ))
        
        # Parse publication type
        type_map = {
            "article": PublicationType.ARTICLE,
            "review": PublicationType.REVIEW,
            "preprint": PublicationType.PREPRINT,
            "book-chapter": PublicationType.BOOK_CHAPTER,
            "book": PublicationType.BOOK,
            "proceedings-article": PublicationType.CONFERENCE,
        }
        pub_type = type_map.get(data.get("type", ""), PublicationType.ARTICLE)
        
        # Parse date
        pub_date = None
        date_str = data.get("publication_date")
        if date_str:
            try:
                pub_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                pass
        
        # Get primary location info
        primary_loc = data.get("primary_location", {}) or {}
        source = primary_loc.get("source", {}) or {}
        
        # Get best OA URL
        oa_url = None
        pdf_url = None
        best_oa = data.get("best_oa_location", {})
        if best_oa:
            oa_url = best_oa.get("landing_page_url")
            pdf_url = best_oa.get("pdf_url")
        
        # Generate internal ID from OpenAlex ID
        openalex_id = data.get("id", "").replace("https://openalex.org/", "")
        internal_id = f"openalex:{openalex_id}" if openalex_id else f"openalex:{hash(data.get('title', ''))}"
        
        # Extract keyword strings from concepts
        keywords = []
        if data.get("concepts"):
            keywords = [c.get("display_name", "") for c in data.get("concepts", [])[:10] if isinstance(c, dict) and c.get("display_name")]
        
        return Publication(
            id=internal_id,
            title=data.get("title", "Untitled"),
            authors=authors,
            abstract=data.get("abstract"),
            doi=data.get("doi", "").replace("https://doi.org/", "") if data.get("doi") else None,
            year=data.get("publication_year"),
            publication_date=pub_date,
            venue=source.get("display_name"),
            volume=data.get("biblio", {}).get("volume"),
            issue=data.get("biblio", {}).get("issue"),
            pages=data.get("biblio", {}).get("first_page"),
            publication_type=pub_type,
            keywords=keywords,
            citation_count=data.get("cited_by_count", 0),
            openalex_id=openalex_id,
            url=oa_url or data.get("doi"),
            pdf_url=pdf_url,
            is_open_access=data.get("open_access", {}).get("is_oa", False),
            sources=["openalex"],
        )
