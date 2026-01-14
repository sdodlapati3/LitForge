"""
Crossref API client.

Comprehensive metadata for DOI-registered content.
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


class CrossrefClient(BaseClient):
    """
    Client for the Crossref API.
    
    Crossref provides metadata for over 130M DOI-registered works.
    Free, polite pool available with email.
    
    API Docs: https://api.crossref.org/
    """
    
    def __init__(self, email: str | None = None):
        """
        Initialize the Crossref client.
        
        Args:
            email: Email for polite pool (better rate limits)
        """
        super().__init__(
            base_url="https://api.crossref.org",
            rate_limit=10.0 if email else 1.0,
        )
        self.email = email
    
    def _get_headers(self) -> dict[str, str] | None:
        """Get headers with email if provided."""
        if self.email:
            return {"User-Agent": f"LitForge/1.0 (mailto:{self.email})"}
        return {"User-Agent": "LitForge/1.0"}
    
    def search(
        self,
        query: str,
        limit: int = 25,
        filters: SearchFilter | None = None,
    ) -> list[Publication]:
        """
        Search Crossref.
        
        Args:
            query: Search query
            limit: Maximum results
            filters: Optional filters
            
        Returns:
            List of publications
        """
        params: dict[str, Any] = {
            "query": query,
            "rows": min(limit, 1000),
        }
        
        # Build filter
        filter_parts = []
        
        if filters:
            if filters.year_from:
                filter_parts.append(f"from-pub-date:{filters.year_from}")
            if filters.year_to:
                filter_parts.append(f"until-pub-date:{filters.year_to}")
            if filters.types:
                type_map = {
                    PublicationType.ARTICLE: "journal-article",
                    PublicationType.BOOK: "book",
                    PublicationType.BOOK_CHAPTER: "book-chapter",
                    PublicationType.CONFERENCE: "proceedings-article",
                }
                for t in filters.types:
                    if t in type_map:
                        filter_parts.append(f"type:{type_map[t]}")
        
        if filter_parts:
            params["filter"] = ",".join(filter_parts)
        
        response = self._get(
            "/works",
            params=params,
            headers=self._get_headers(),
        )
        
        items = response.get("message", {}).get("items", [])
        return [self._parse_work(w) for w in items if w]
    
    def get_paper(self, doi: str) -> Publication | None:
        """
        Get a specific paper by DOI.
        
        Args:
            doi: DOI
            
        Returns:
            Publication or None
        """
        # Normalize DOI
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        
        try:
            response = self._get(
                f"/works/{doi}",
                headers=self._get_headers(),
            )
            work = response.get("message", {})
            return self._parse_work(work) if work else None
        except Exception as e:
            logger.debug(f"Crossref lookup failed: {e}")
            return None
    
    def _parse_work(self, data: dict[str, Any]) -> Publication:
        """Parse a Crossref work into a Publication."""
        # Get title
        titles = data.get("title", [])
        title = titles[0] if titles else "Untitled"
        
        # Parse authors
        authors = []
        for auth in data.get("author", []):
            given = auth.get("given", "")
            family = auth.get("family", "")
            name = f"{given} {family}".strip() or "Unknown"
            authors.append(Author(
                name=name,
                orcid=auth.get("ORCID"),
            ))
        
        # Get abstract
        abstract = data.get("abstract")
        if abstract:
            # Remove JATS tags
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract)
        
        # Parse date
        year = None
        pub_date = None
        
        # Try published-print, then published-online, then created
        for date_field in ["published-print", "published-online", "created"]:
            date_parts = data.get(date_field, {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                parts = date_parts[0]
                year = parts[0] if len(parts) > 0 else None
                if len(parts) >= 3:
                    try:
                        pub_date = datetime(parts[0], parts[1], parts[2]).date()
                    except ValueError:
                        pass
                if year:
                    break
        
        # Get publication type
        type_map = {
            "journal-article": PublicationType.ARTICLE,
            "book": PublicationType.BOOK,
            "book-chapter": PublicationType.BOOK_CHAPTER,
            "proceedings-article": PublicationType.CONFERENCE,
            "posted-content": PublicationType.PREPRINT,
        }
        pub_type = type_map.get(data.get("type", ""), PublicationType.ARTICLE)
        
        # Get venue
        container = data.get("container-title", [])
        venue = container[0] if container else None
        
        # Get DOI
        doi = data.get("DOI")
        
        return Publication(
            title=title,
            authors=authors,
            abstract=abstract,
            doi=doi,
            year=year,
            publication_date=pub_date,
            venue=venue,
            volume=data.get("volume"),
            issue=data.get("issue"),
            pages=data.get("page"),
            publication_type=pub_type,
            citation_count=data.get("is-referenced-by-count", 0),
            url=f"https://doi.org/{doi}" if doi else None,
            sources=["crossref"],
        )
