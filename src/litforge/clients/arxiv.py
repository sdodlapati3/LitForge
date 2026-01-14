"""
arXiv API client.

Essential for preprints in physics, math, CS, and related fields.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from litforge.clients.base import BaseClient
from litforge.models import (
    Author,
    Publication,
    PublicationType,
    SearchFilter,
)

logger = logging.getLogger(__name__)


class ArxivClient(BaseClient):
    """
    Client for the arXiv API.
    
    arXiv is the primary source for preprints in many scientific fields.
    Free, no authentication needed.
    
    API Docs: https://info.arxiv.org/help/api/index.html
    """
    
    # XML namespaces
    ATOM_NS = "{http://www.w3.org/2005/Atom}"
    ARXIV_NS = "{http://arxiv.org/schemas/atom}"
    
    def __init__(self):
        """Initialize the arXiv client."""
        super().__init__(
            base_url="http://export.arxiv.org/api",
            rate_limit=1.0,  # Be polite to arXiv
        )
    
    def search(
        self,
        query: str,
        limit: int = 25,
        filters: SearchFilter | None = None,
    ) -> list[Publication]:
        """
        Search arXiv.
        
        Args:
            query: Search query
            limit: Maximum results
            filters: Optional filters (year filtering not well supported)
            
        Returns:
            List of publications
        """
        # Build query - arXiv uses a simple search syntax
        search_query = f"all:{query}"
        
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(limit, 2000),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        
        # arXiv API returns XML
        self._rate_limit_wait()
        response = self.client.get(f"/query?{urlencode(params)}")
        response.raise_for_status()
        
        return self._parse_feed(response.text, filters)
    
    def get_paper(self, arxiv_id: str) -> Publication | None:
        """
        Get a specific paper by arXiv ID.
        
        Args:
            arxiv_id: arXiv ID (e.g., "2103.00020" or "hep-ph/0001001")
            
        Returns:
            Publication or None
        """
        # Clean up ID
        arxiv_id = arxiv_id.replace("arXiv:", "").replace("arxiv:", "")
        
        params = {
            "id_list": arxiv_id,
        }
        
        self._rate_limit_wait()
        response = self.client.get(f"/query?{urlencode(params)}")
        response.raise_for_status()
        
        papers = self._parse_feed(response.text)
        return papers[0] if papers else None
    
    def _parse_feed(
        self,
        xml_text: str,
        filters: SearchFilter | None = None,
    ) -> list[Publication]:
        """Parse arXiv Atom feed."""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for entry in root.findall(f"{self.ATOM_NS}entry"):
                paper = self._parse_entry(entry)
                if paper:
                    # Apply filters
                    if filters and not self._matches_filters(paper, filters):
                        continue
                    papers.append(paper)
                    
        except ET.ParseError as e:
            logger.error(f"Error parsing arXiv XML: {e}")
        
        return papers
    
    def _parse_entry(self, entry: ET.Element) -> Publication | None:
        """Parse a single arXiv entry."""
        try:
            # Get ID
            id_elem = entry.find(f"{self.ATOM_NS}id")
            if id_elem is None or id_elem.text is None:
                return None
            
            # Extract arXiv ID from URL
            arxiv_id = id_elem.text.split("/abs/")[-1]
            # Remove version suffix if present
            base_id = arxiv_id.rsplit("v", 1)[0] if "v" in arxiv_id else arxiv_id
            
            # Get title
            title_elem = entry.find(f"{self.ATOM_NS}title")
            title = title_elem.text if title_elem is not None else "Untitled"
            # Clean up title (remove newlines)
            title = " ".join(title.split())
            
            # Get abstract
            abstract = None
            summary_elem = entry.find(f"{self.ATOM_NS}summary")
            if summary_elem is not None and summary_elem.text:
                abstract = " ".join(summary_elem.text.split())
            
            # Get authors
            authors = []
            for author_elem in entry.findall(f"{self.ATOM_NS}author"):
                name_elem = author_elem.find(f"{self.ATOM_NS}name")
                if name_elem is not None and name_elem.text:
                    authors.append(Author(name=name_elem.text))
            
            # Get publication date
            pub_date = None
            year = None
            published_elem = entry.find(f"{self.ATOM_NS}published")
            if published_elem is not None and published_elem.text:
                try:
                    pub_date = datetime.fromisoformat(
                        published_elem.text.replace("Z", "+00:00")
                    ).date()
                    year = pub_date.year
                except ValueError:
                    pass
            
            # Get categories
            categories = []
            for cat_elem in entry.findall(f"{self.ATOM_NS}category"):
                term = cat_elem.get("term")
                if term:
                    categories.append(term)
            
            # Get DOI if available
            doi = None
            doi_elem = entry.find(f"{self.ARXIV_NS}doi")
            if doi_elem is not None and doi_elem.text:
                doi = doi_elem.text
            
            # Build URLs
            pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"
            abs_url = f"https://arxiv.org/abs/{base_id}"
            
            return Publication(
                title=title,
                authors=authors,
                abstract=abstract,
                doi=doi,
                year=year,
                publication_date=pub_date,
                publication_type=PublicationType.PREPRINT,
                keywords=categories if categories else None,
                arxiv_id=base_id,
                url=abs_url,
                pdf_url=pdf_url,
                is_open_access=True,  # All arXiv papers are OA
                sources=["arxiv"],
            )
            
        except Exception as e:
            logger.error(f"Error parsing arXiv entry: {e}")
            return None
    
    def _matches_filters(
        self,
        paper: Publication,
        filters: SearchFilter,
    ) -> bool:
        """Check if paper matches filters."""
        if filters.year_from and paper.year and paper.year < filters.year_from:
            return False
        if filters.year_to and paper.year and paper.year > filters.year_to:
            return False
        return True
