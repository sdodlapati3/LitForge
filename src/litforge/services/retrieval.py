"""
Retrieval Service - Full-text document retrieval.

Handles downloading and extracting text from PDFs.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from litforge.config import LitForgeConfig, get_config
from litforge.models import Publication

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service for retrieving full-text documents.
    
    Supports downloading from multiple sources (publisher, Unpaywall, preprint)
    and extracting text from PDFs with section parsing.
    """
    
    def __init__(self, config: LitForgeConfig | None = None):
        """
        Initialize the retrieval service.
        
        Args:
            config: LitForge configuration
        """
        self.config = config or get_config()
        self._cache_dir = Path(self.config.cache.directory) / "pdfs"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._unpaywall_client: Any | None = None
        self._pdf_extractor: Any | None = None
    
    @property
    def unpaywall_client(self) -> Any:
        """Lazy-load Unpaywall client."""
        if self._unpaywall_client is None:
            from litforge.clients.unpaywall import UnpaywallClient
            self._unpaywall_client = UnpaywallClient(
                email=self.config.sources.unpaywall_email
            )
        return self._unpaywall_client
    
    @property
    def pdf_extractor(self) -> Any:
        """Lazy-load PDF extractor."""
        if self._pdf_extractor is None:
            from litforge.processing import PDFExtractor
            self._pdf_extractor = PDFExtractor()
        return self._pdf_extractor
    
    def retrieve(
        self,
        publication: Publication | str,
        *,
        download_pdf: bool = True,
        extract_text: bool = True,
    ) -> Publication:
        """
        Retrieve full text for a publication.
        
        Tries multiple sources in order of preference:
        1. Local cache
        2. Open access PDF via Unpaywall
        3. PubMed Central
        4. arXiv
        
        Args:
            publication: Publication to retrieve or DOI/identifier string
            download_pdf: Whether to download PDF
            extract_text: Whether to extract text from PDF
            
        Returns:
            Publication with full_text populated if available
        """
        # Handle string input (DOI or ID)
        if isinstance(publication, str):
            # Try to look up the publication
            from litforge.services.discovery import DiscoveryService
            discovery = DiscoveryService(self.config)
            pub = discovery.lookup(doi=publication)
            if not pub:
                raise ValueError(f"Could not find publication: {publication}")
            publication = pub
        
        # Check cache first
        cached = self._get_cached(publication)
        if cached:
            logger.debug(f"Found cached text for {publication.doi or publication.title}")
            publication.full_text = cached
            return publication
        
        if not download_pdf:
            return publication
        
        # Try to get PDF URL
        pdf_url = self._find_pdf_url(publication)
        
        if not pdf_url:
            logger.warning(f"No PDF URL found for {publication.doi or publication.title}")
            return publication
        
        # Download and extract
        try:
            pdf_path = self._download_pdf(pdf_url, publication)
            if pdf_path and extract_text:
                text = self._extract_text(pdf_path)
                if text:
                    self._cache_text(publication, text)
                    publication.full_text = text
        except Exception as e:
            logger.error(f"Error retrieving {publication.doi}: {e}")
        
        return publication
    
    def retrieve_batch(
        self,
        publications: list[Publication],
        *,
        max_concurrent: int = 5,
        progress: bool = True,
    ) -> list[Publication]:
        """
        Retrieve full text for multiple publications.
        
        Args:
            publications: Publications to retrieve
            max_concurrent: Maximum concurrent downloads
            progress: Show progress bar (if tqdm installed)
            
        Returns:
            List of publications with full_text populated where available
        """
        results: list[Publication] = []
        
        # Try to use tqdm for progress if requested
        iterator = publications
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(publications, desc="Retrieving papers")
            except ImportError:
                pass
        
        for pub in iterator:
            results.append(self.retrieve(pub))
        
        return results
    
    def _find_pdf_url(self, publication: Publication) -> str | None:
        """Find a PDF URL for the publication."""
        # Check if publication has a PDF URL
        if publication.pdf_url:
            return publication.pdf_url
        
        # Try Unpaywall
        if publication.doi:
            try:
                oa_info = self.unpaywall_client.get_open_access(publication.doi)
                if oa_info and oa_info.get("pdf_url"):
                    return oa_info["pdf_url"]
            except Exception as e:
                logger.debug(f"Unpaywall lookup failed: {e}")
        
        # Try arXiv
        if publication.arxiv_id:
            return f"https://arxiv.org/pdf/{publication.arxiv_id}.pdf"
        
        # Try PubMed Central
        if publication.pmcid:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{publication.pmcid}/pdf/"
        
        return None
    
    def _download_pdf(self, url: str, publication: Publication) -> Path | None:
        """Download PDF to cache directory."""
        import httpx
        
        # Generate cache filename
        cache_key = self._get_cache_key(publication)
        pdf_path = self._cache_dir / f"{cache_key}.pdf"
        
        if pdf_path.exists():
            return pdf_path
        
        try:
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                response = client.get(url, headers={
                    "User-Agent": "LitForge/1.0 (mailto:litforge@example.com)"
                })
                response.raise_for_status()
                
                content_type = response.headers.get("content-type", "")
                if "application/pdf" in content_type or url.endswith(".pdf"):
                    pdf_path.write_bytes(response.content)
                    logger.info(f"Downloaded PDF: {pdf_path}")
                    return pdf_path
                else:
                    logger.warning(f"Response is not PDF (content-type: {content_type}): {url}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            return None
    
    def _extract_text(self, pdf_path: Path) -> str | None:
        """Extract text from a PDF file using the PDF extractor."""
        try:
            text = self.pdf_extractor.extract_text_only(pdf_path)
            return text if text and text.strip() else None
        except ImportError:
            logger.error("PyMuPDF not installed. Install with: pip install pymupdf")
            return None
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None
    
    def _extract_with_sections(self, pdf_path: Path) -> tuple[str | None, dict[str, str] | None]:
        """
        Extract text with section parsing from a PDF file.
        
        Returns:
            Tuple of (full_text, sections_dict)
        """
        try:
            doc = self.pdf_extractor.extract(pdf_path)
            return doc.full_text, doc.sections
        except ImportError:
            logger.error("PyMuPDF not installed. Install with: pip install pymupdf")
            return None, None
        except Exception as e:
            logger.error(f"Error extracting sections from {pdf_path}: {e}")
            return None, None
    
    def retrieve_with_sections(
        self,
        publication: Publication | str,
    ) -> Publication:
        """
        Retrieve full text with section parsing.
        
        Args:
            publication: Publication to retrieve, DOI, or arXiv ID string
            
        Returns:
            Publication with full_text and sections populated
        """
        # Handle string input
        if isinstance(publication, str):
            from litforge.services.discovery import DiscoveryService
            discovery = DiscoveryService(self.config)
            
            identifier = publication.strip()
            pub = None
            
            # Try to detect identifier type
            if identifier.startswith("10."):
                # Looks like a DOI
                pub = discovery.lookup(doi=identifier)
            elif identifier.replace(".", "").replace("v", "").isdigit() or "/" in identifier:
                # Looks like an arXiv ID (e.g., 1706.03762, 2301.07041, cs/0001001)
                pub = discovery.lookup(arxiv_id=identifier)
            else:
                # Try DOI first, then arXiv
                pub = discovery.lookup(doi=identifier)
                if not pub:
                    pub = discovery.lookup(arxiv_id=identifier)
            
            if not pub:
                raise ValueError(f"Could not find publication: {publication}")
            publication = pub
        
        # Check cache first
        cached = self._get_cached(publication)
        if cached:
            publication.full_text = cached
            # Try to parse sections from cached text
            from litforge.processing import SectionParser
            parser = SectionParser()
            publication.sections = parser.parse(cached)
            return publication
        
        # Find and download PDF
        pdf_url = self._find_pdf_url(publication)
        if not pdf_url:
            logger.warning(f"No PDF URL found for {publication.doi or publication.title}")
            return publication
        
        try:
            pdf_path = self._download_pdf(pdf_url, publication)
            if pdf_path:
                full_text, sections = self._extract_with_sections(pdf_path)
                if full_text:
                    self._cache_text(publication, full_text)
                    publication.full_text = full_text
                    publication.sections = sections
        except Exception as e:
            logger.error(f"Error retrieving with sections {publication.doi}: {e}")
        
        return publication
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" +", " ", text)
        
        # Remove common artifacts
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)  # Page numbers
        
        return text.strip()
    
    def _get_cache_key(self, publication: Publication) -> str:
        """Generate a cache key for a publication."""
        if publication.doi:
            # Use DOI hash
            return hashlib.md5(publication.doi.lower().encode()).hexdigest()
        elif publication.openalex_id:
            return f"openalex_{publication.openalex_id}"
        elif publication.arxiv_id:
            return f"arxiv_{publication.arxiv_id.replace('/', '_')}"
        else:
            # Use title hash
            return hashlib.md5(publication.title.encode()).hexdigest()
    
    def _get_cached(self, publication: Publication) -> str | None:
        """Get cached text for a publication."""
        cache_key = self._get_cache_key(publication)
        text_path = self._cache_dir / f"{cache_key}.txt"
        
        if text_path.exists():
            return text_path.read_text()
        
        return None
    
    def _cache_text(self, publication: Publication, text: str) -> None:
        """Cache extracted text for a publication."""
        cache_key = self._get_cache_key(publication)
        text_path = self._cache_dir / f"{cache_key}.txt"
        text_path.write_text(text)
