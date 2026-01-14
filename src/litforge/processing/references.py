"""
Reference Extractor for Scientific Papers.

Extracts and parses bibliographic references from paper text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParsedReference:
    """A parsed bibliographic reference."""
    
    raw_text: str
    authors: list[str] | None = None
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    volume: str | None = None
    pages: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    
    @property
    def is_parsed(self) -> bool:
        """Whether we successfully parsed this reference."""
        return self.title is not None or self.authors is not None


class ReferenceExtractor:
    """
    Extracts bibliographic references from scientific papers.
    
    Handles various citation styles:
    - Numbered references [1], [2], etc.
    - Author-year citations (Smith et al., 2020)
    - Full bibliography entries
    
    Example:
        >>> extractor = ReferenceExtractor()
        >>> refs = extractor.extract(paper_text)
        >>> for ref in refs:
        ...     print(ref.title, ref.year)
    """
    
    def __init__(
        self,
        parse_details: bool = True,
        include_dois: bool = True,
    ):
        """
        Initialize the reference extractor.
        
        Args:
            parse_details: Attempt to parse reference details (authors, title, etc.)
            include_dois: Extract DOIs from references when parsing
        """
        self.parse_details = parse_details
        self.include_dois = include_dois
    
    def extract(self, text: str) -> list[str]:
        """
        Extract raw reference strings from paper text.
        
        Args:
            text: Full paper text
            
        Returns:
            List of reference strings
        """
        # Find the references section
        ref_section = self._find_references_section(text)
        
        if not ref_section:
            return []
        
        # Split into individual references
        references = self._split_references(ref_section)
        
        return references
    
    def extract_and_parse(self, text: str) -> list[ParsedReference]:
        """
        Extract and parse references with structured information.
        
        Args:
            text: Full paper text
            
        Returns:
            List of ParsedReference objects
        """
        raw_refs = self.extract(text)
        
        parsed_refs = []
        for raw in raw_refs:
            parsed = self._parse_reference(raw)
            parsed_refs.append(parsed)
        
        return parsed_refs
    
    def extract_dois(self, text: str) -> list[str]:
        """
        Extract all DOIs from the text.
        
        Args:
            text: Paper text
            
        Returns:
            List of DOI strings
        """
        # DOI pattern: 10.xxxx/xxxxx
        doi_pattern = r"10\.\d{4,}/[^\s\]\)>\"']+"
        
        dois = re.findall(doi_pattern, text)
        
        # Clean up DOIs (remove trailing punctuation)
        cleaned = []
        for doi in dois:
            doi = doi.rstrip(".,;:")
            if doi not in cleaned:
                cleaned.append(doi)
        
        return cleaned
    
    def extract_arxiv_ids(self, text: str) -> list[str]:
        """
        Extract arXiv IDs from the text.
        
        Args:
            text: Paper text
            
        Returns:
            List of arXiv ID strings
        """
        # arXiv patterns:
        # - New format: 2103.00020
        # - Old format: hep-ph/0001001
        patterns = [
            r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"arXiv:([a-z\-]+/\d{7}(?:v\d+)?)",
            r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"arxiv\.org/abs/([a-z\-]+/\d{7}(?:v\d+)?)",
        ]
        
        arxiv_ids = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match not in arxiv_ids:
                    arxiv_ids.append(match)
        
        return arxiv_ids
    
    def _find_references_section(self, text: str) -> str | None:
        """Find and extract the references section."""
        # Common reference section headers
        patterns = [
            r"\n\s*references?\s*\n",
            r"\n\s*bibliography\s*\n",
            r"\n\s*literature\s+cited\s*\n",
            r"\n\s*works?\s+cited\s*\n",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract from the header to end of document
                # (or until supplementary/appendix section)
                start = match.end()
                
                # Find end (supplementary, appendix, or end of text)
                end_patterns = [
                    r"\n\s*(?:supplement|appendix|supporting\s+information)",
                ]
                
                end_pos = len(text)
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, text[start:], re.IGNORECASE)
                    if end_match:
                        end_pos = start + end_match.start()
                        break
                
                return text[start:end_pos].strip()
        
        return None
    
    def _split_references(self, ref_text: str) -> list[str]:
        """Split references section into individual references."""
        references = []
        
        # Try numbered references first [1], [2], etc.
        numbered = re.split(r"\n\s*\[(\d+)\]\s*", ref_text)
        if len(numbered) > 2:
            # Skip first empty element and get pairs of (number, content)
            for i in range(1, len(numbered), 2):
                if i + 1 < len(numbered):
                    ref = numbered[i + 1].strip()
                    if ref and len(ref) > 20:
                        references.append(ref)
            if references:
                return references
        
        # Try numbered without brackets: 1., 2., etc.
        numbered = re.split(r"\n\s*(\d+)\.\s+", ref_text)
        if len(numbered) > 2:
            for i in range(1, len(numbered), 2):
                if i + 1 < len(numbered):
                    ref = numbered[i + 1].strip()
                    if ref and len(ref) > 20:
                        references.append(ref)
            if references:
                return references
        
        # Fall back to paragraph splitting
        paragraphs = re.split(r"\n\n+", ref_text)
        for para in paragraphs:
            para = para.strip()
            # A reference should be at least ~50 chars
            if len(para) > 50:
                # Clean up
                para = re.sub(r"\s+", " ", para)
                references.append(para)
        
        return references
    
    def _parse_reference(self, ref_text: str) -> ParsedReference:
        """Parse a single reference string into structured data."""
        result = ParsedReference(raw_text=ref_text)
        
        if not self.parse_details:
            return result
        
        # Extract DOI
        if self.extract_dois:
            doi_match = re.search(r"10\.\d{4,}/[^\s\]\)>\"']+", ref_text)
            if doi_match:
                result.doi = doi_match.group().rstrip(".,;:")
        
        # Extract arXiv ID
        arxiv_match = re.search(r"arXiv:(\d{4}\.\d{4,5})", ref_text, re.IGNORECASE)
        if arxiv_match:
            result.arxiv_id = arxiv_match.group(1)
        
        # Extract year (4-digit number between 1900-2100)
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", ref_text)
        if year_match:
            result.year = int(year_match.group(1))
        
        # Extract URL
        url_match = re.search(r"https?://[^\s\]\)>\"']+", ref_text)
        if url_match:
            result.url = url_match.group().rstrip(".,;:")
        
        # Try to extract title (usually in quotes or after author list)
        title_patterns = [
            r'"([^"]{20,})"',  # Quoted title
            r"'([^']{20,})'",  # Single-quoted title
            r"\.\s*([A-Z][^.]{20,}[.?!])\s*(?:[A-Z]|In:|$)",  # Sentence after period
        ]
        
        for pattern in title_patterns:
            title_match = re.search(pattern, ref_text)
            if title_match:
                result.title = title_match.group(1).strip()
                break
        
        # Try to extract authors (names at the beginning)
        # This is a simplified heuristic
        author_match = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?)*(?:\s*(?:and|&)\s*[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?)?)", ref_text)
        if author_match:
            author_str = author_match.group(1)
            # Split authors
            authors = re.split(r",\s*|\s+and\s+|\s*&\s*", author_str)
            result.authors = [a.strip() for a in authors if a.strip()]
        
        # Try to extract volume and pages
        vol_pages_match = re.search(r"(\d+)\s*[:\(]\s*(\d+(?:-\d+)?)", ref_text)
        if vol_pages_match:
            result.volume = vol_pages_match.group(1)
            result.pages = vol_pages_match.group(2)
        
        return result


def extract_inline_citations(text: str) -> list[str]:
    """
    Extract inline citations from paper body.
    
    Finds citations like [1], [1,2,3], (Smith et al., 2020), etc.
    
    Args:
        text: Paper body text
        
    Returns:
        List of citation strings
    """
    citations = []
    
    # Numbered citations: [1], [1,2], [1-5]
    numbered = re.findall(r"\[(\d+(?:[,\-–]\s*\d+)*)\]", text)
    citations.extend(numbered)
    
    # Author-year citations: (Smith et al., 2020)
    author_year = re.findall(
        r"\(([A-Z][a-z]+(?:\s+et\s+al\.?)?(?:\s*,\s*\d{4})?(?:\s*;\s*[A-Z][a-z]+(?:\s+et\s+al\.?)?(?:\s*,\s*\d{4})?)*)\)",
        text
    )
    citations.extend(author_year)
    
    return citations
