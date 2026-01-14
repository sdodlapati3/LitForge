"""
PubMed API client via NCBI E-utilities.

Essential for biomedical and life sciences literature.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
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


class PubMedClient(BaseClient):
    """
    Client for PubMed/NCBI E-utilities API.
    
    PubMed is essential for biomedical research with 35M+ citations.
    Free with email, API key recommended for higher limits.
    
    API Docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
    """
    
    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the PubMed client.
        
        Args:
            email: Email (required for E-utilities)
            api_key: API key for higher rate limits (10/sec vs 3/sec)
        """
        super().__init__(
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            rate_limit=10.0 if api_key else 3.0,
        )
        self.email = email
        self.api_key = api_key
    
    def _get_params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Add common params."""
        params = params or {}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params
    
    def search(
        self,
        query: str,
        limit: int = 25,
        filters: SearchFilter | None = None,
    ) -> list[Publication]:
        """
        Search PubMed.
        
        Args:
            query: Search query (PubMed syntax supported)
            limit: Maximum results
            filters: Optional filters
            
        Returns:
            List of publications
        """
        # Build search query
        search_terms = [query]
        
        if filters:
            if filters.year_from and filters.year_to:
                search_terms.append(f"{filters.year_from}:{filters.year_to}[dp]")
            elif filters.year_from:
                search_terms.append(f"{filters.year_from}:3000[dp]")
            elif filters.year_to:
                search_terms.append(f"1800:{filters.year_to}[dp]")
            
            if filters.publication_types:
                type_map = {
                    "review": "review[pt]",
                }
                for t in filters.publication_types:
                    if t in type_map:
                        search_terms.append(type_map[t])
            
            if filters.open_access_only:
                search_terms.append("free full text[sb]")
        
        # Search for IDs
        params = self._get_params({
            "db": "pubmed",
            "term": " AND ".join(search_terms),
            "retmax": min(limit, 10000),
            "retmode": "json",
        })
        
        response = self._get("/esearch.fcgi", params=params)
        
        id_list = response.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            return []
        
        # Fetch details
        return self._fetch_papers(id_list[:limit])
    
    def get_paper(
        self,
        pmid: str | None = None,
        doi: str | None = None,
    ) -> Publication | None:
        """
        Get a specific paper.
        
        Args:
            pmid: PubMed ID
            doi: DOI (will search for it)
            
        Returns:
            Publication or None
        """
        if pmid:
            papers = self._fetch_papers([pmid])
            return papers[0] if papers else None
        
        if doi:
            # Search by DOI
            params = self._get_params({
                "db": "pubmed",
                "term": f"{doi}[doi]",
                "retmode": "json",
            })
            response = self._get("/esearch.fcgi", params=params)
            id_list = response.get("esearchresult", {}).get("idlist", [])
            
            if id_list:
                papers = self._fetch_papers([id_list[0]])
                return papers[0] if papers else None
        
        return None
    
    def _fetch_papers(self, pmids: list[str]) -> list[Publication]:
        """Fetch paper details by PMIDs."""
        if not pmids:
            return []
        
        params = self._get_params({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        })
        
        # Get raw XML
        self._rate_limit_wait()
        response = self.client.get("/efetch.fcgi", params=params)
        response.raise_for_status()
        
        return self._parse_xml(response.text)
    
    def _parse_xml(self, xml_text: str) -> list[Publication]:
        """Parse PubMed XML response."""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
                    
        except ET.ParseError as e:
            logger.error(f"Error parsing PubMed XML: {e}")
        
        return papers
    
    def _parse_article(self, article: ET.Element) -> Publication | None:
        """Parse a single PubMed article."""
        try:
            medline = article.find("MedlineCitation")
            if medline is None:
                return None
            
            art = medline.find("Article")
            if art is None:
                return None
            
            # Get PMID
            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            # Get title
            title_elem = art.find("ArticleTitle")
            title = title_elem.text if title_elem is not None else "Untitled"
            
            # Get abstract
            abstract = None
            abstract_elem = art.find("Abstract/AbstractText")
            if abstract_elem is not None:
                abstract = abstract_elem.text
            
            # Get authors
            authors = []
            author_list = art.find("AuthorList")
            if author_list is not None:
                for auth in author_list.findall("Author"):
                    last = auth.find("LastName")
                    first = auth.find("ForeName")
                    if last is not None:
                        name = last.text or ""
                        if first is not None and first.text:
                            name = f"{first.text} {name}"
                        authors.append(Author(name=name))
            
            # Get publication date
            year = None
            pub_date = None
            
            # Try PubDate
            journal = art.find("Journal")
            if journal is not None:
                pub_date_elem = journal.find("JournalIssue/PubDate")
                if pub_date_elem is not None:
                    year_elem = pub_date_elem.find("Year")
                    if year_elem is not None:
                        year = int(year_elem.text)
                    
                    month_elem = pub_date_elem.find("Month")
                    day_elem = pub_date_elem.find("Day")
                    
                    if year_elem is not None and month_elem is not None:
                        try:
                            month = self._parse_month(month_elem.text)
                            day = int(day_elem.text) if day_elem is not None else 1
                            pub_date = datetime(year, month, day).date()
                        except (ValueError, TypeError):
                            pass
            
            # Get journal
            venue = None
            journal_elem = art.find("Journal/Title")
            if journal_elem is not None:
                venue = journal_elem.text
            
            # Get DOI
            doi = None
            for id_elem in article.findall(".//ArticleId"):
                if id_elem.get("IdType") == "doi":
                    doi = id_elem.text
                    break
            
            # Get PMC ID
            pmcid = None
            for id_elem in article.findall(".//ArticleId"):
                if id_elem.get("IdType") == "pmc":
                    pmcid = id_elem.text
                    break
            
            # Get keywords/MeSH terms
            keywords = []
            for mesh in medline.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    keywords.append(mesh.text)
            
            # Generate internal ID from PMID
            internal_id = f"pubmed:{pmid}" if pmid else f"pubmed:{hash(title)}"
            
            return Publication(
                id=internal_id,
                title=title,
                authors=authors,
                abstract=abstract,
                doi=doi,
                year=year,
                publication_date=pub_date,
                venue=venue,
                publication_type=PublicationType.ARTICLE,
                keywords=keywords[:20] if keywords else None,
                pmid=pmid,
                pmcid=pmcid,
                is_open_access=pmcid is not None,
                sources=["pubmed"],
            )
            
        except Exception as e:
            logger.error(f"Error parsing PubMed article: {e}")
            return None
    
    def _parse_month(self, month_str: str | None) -> int:
        """Parse month string to integer."""
        if not month_str:
            return 1
        
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        
        try:
            return int(month_str)
        except ValueError:
            return month_map.get(month_str.lower()[:3], 1)
