"""
Publication-related data models.

These models represent scientific publications, authors, and citations
in a normalized, source-agnostic format.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class PublicationType(str, Enum):
    """Types of publications."""
    ARTICLE = "article"
    REVIEW = "review"
    PREPRINT = "preprint"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    CONFERENCE = "conference"
    THESIS = "thesis"
    DATASET = "dataset"
    OTHER = "other"


class AccessType(str, Enum):
    """Open access status."""
    OPEN = "open"
    CLOSED = "closed"
    HYBRID = "hybrid"
    BRONZE = "bronze"
    GREEN = "green"
    GOLD = "gold"
    UNKNOWN = "unknown"


class Affiliation(BaseModel):
    """Author affiliation/institution."""
    
    name: str = Field(..., description="Institution name")
    ror_id: str | None = Field(default=None, description="ROR identifier")
    country: str | None = Field(default=None, description="Country code (ISO 3166-1)")
    city: str | None = Field(default=None, description="City name")
    
    def __str__(self) -> str:
        parts = [self.name]
        if self.city:
            parts.append(self.city)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)


class Author(BaseModel):
    """Author of a publication."""
    
    name: str = Field(..., description="Full author name")
    given_name: str | None = Field(default=None, description="First/given name")
    family_name: str | None = Field(default=None, description="Last/family name")
    
    # Identifiers
    orcid: str | None = Field(default=None, description="ORCID identifier")
    openalex_id: str | None = Field(default=None, description="OpenAlex author ID")
    semantic_scholar_id: str | None = Field(default=None, description="Semantic Scholar ID")
    
    # Affiliations
    affiliations: list[Affiliation] = Field(default_factory=list)
    
    # Position in author list
    position: int | None = Field(default=None, description="Position in author list (1-indexed)")
    is_corresponding: bool = Field(default=False, description="Is corresponding author")
    
    def __str__(self) -> str:
        return self.name
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Display name with initials if available."""
        if self.given_name and self.family_name:
            initial = self.given_name[0] if self.given_name else ""
            return f"{self.family_name}, {initial}."
        return self.name


class Citation(BaseModel):
    """Citation relationship between publications."""
    
    citing_id: str = Field(..., description="ID of citing publication")
    cited_id: str = Field(..., description="ID of cited publication")
    
    # Context (if available)
    context: str | None = Field(default=None, description="Citation context/sentence")
    section: str | None = Field(default=None, description="Section where citation appears")
    intent: str | None = Field(default=None, description="Citation intent (background, method, result)")
    
    # Source metadata
    source: str | None = Field(default=None, description="Data source for this citation")


class Publication(BaseModel):
    """
    A scientific publication.
    
    This is the core model for representing papers, articles, preprints,
    and other scholarly works in a normalized format.
    """
    
    # Primary identifiers
    id: str = Field(..., description="Internal LitForge ID")
    doi: str | None = Field(default=None, description="Digital Object Identifier")
    pmid: str | None = Field(default=None, description="PubMed ID")
    pmcid: str | None = Field(default=None, description="PubMed Central ID")
    arxiv_id: str | None = Field(default=None, description="arXiv ID")
    
    # Source-specific IDs
    openalex_id: str | None = Field(default=None, description="OpenAlex work ID")
    semantic_scholar_id: str | None = Field(default=None, description="Semantic Scholar paper ID")
    
    # Core metadata
    title: str = Field(..., description="Publication title")
    abstract: str | None = Field(default=None, description="Abstract text")
    
    # Authors
    authors: list[Author] = Field(default_factory=list)
    
    # Publication info
    publication_date: date | None = Field(default=None, description="Publication date")
    year: int | None = Field(default=None, description="Publication year")
    
    # Venue
    venue: str | None = Field(default=None, description="Journal/conference name")
    venue_short: str | None = Field(default=None, description="Abbreviated venue name")
    volume: str | None = Field(default=None)
    issue: str | None = Field(default=None)
    pages: str | None = Field(default=None)
    
    # Type and access
    publication_type: PublicationType = Field(default=PublicationType.ARTICLE)
    access_type: AccessType = Field(default=AccessType.UNKNOWN)
    
    # URLs
    url: str | None = Field(default=None, description="Primary URL")
    pdf_url: str | None = Field(default=None, description="Direct PDF URL")
    landing_page_url: str | None = Field(default=None, description="Publisher landing page")
    
    # Metrics
    citation_count: int = Field(default=0, description="Number of citations")
    influential_citation_count: int | None = Field(default=None, description="Influential citations (S2)")
    reference_count: int | None = Field(default=None, description="Number of references")
    
    # Concepts and topics
    concepts: list[str] = Field(default_factory=list, description="Topic concepts")
    keywords: list[str] = Field(default_factory=list, description="Author keywords")
    mesh_terms: list[str] = Field(default_factory=list, description="MeSH terms")
    fields_of_study: list[str] = Field(default_factory=list, description="Fields of study")
    
    # Full text
    full_text: str | None = Field(default=None, description="Full text content", exclude=True)
    sections: dict[str, str] | None = Field(default=None, description="Extracted sections", exclude=True)
    
    # Embedding (for vector search)
    embedding: list[float] | None = Field(default=None, description="Vector embedding", exclude=True)
    
    # Source tracking
    sources: list[str] = Field(default_factory=list, description="Data sources")
    retrieved_at: datetime | None = Field(default=None, description="When data was retrieved")
    
    # Raw data from sources (for debugging)
    raw_data: dict[str, Any] | None = Field(default=None, exclude=True)
    
    def __str__(self) -> str:
        author_str = self.authors[0].family_name if self.authors else "Unknown"
        if len(self.authors) > 1:
            author_str += " et al."
        year_str = str(self.year) if self.year else "n.d."
        return f"{author_str} ({year_str}). {self.title}"
    
    def __repr__(self) -> str:
        return f"Publication(id={self.id!r}, doi={self.doi!r}, title={self.title[:50]!r}...)"
    
    @computed_field
    @property
    def first_author(self) -> Author | None:
        """Get first author."""
        return self.authors[0] if self.authors else None
    
    @computed_field
    @property
    def author_names(self) -> list[str]:
        """Get list of author names."""
        return [a.name for a in self.authors]
    
    @computed_field
    @property
    def has_full_text(self) -> bool:
        """Check if full text is available."""
        return bool(self.full_text)
    
    @computed_field
    @property
    def is_open_access(self) -> bool:
        """Check if publication is open access."""
        return self.access_type in (
            AccessType.OPEN, AccessType.GOLD, AccessType.GREEN, 
            AccessType.BRONZE, AccessType.HYBRID
        )
    
    @computed_field
    @property
    def citation_string(self) -> str:
        """Generate a citation string."""
        parts = []
        
        # Authors
        if self.authors:
            if len(self.authors) == 1:
                parts.append(self.authors[0].family_name or self.authors[0].name)
            elif len(self.authors) == 2:
                parts.append(
                    f"{self.authors[0].family_name or self.authors[0].name} & "
                    f"{self.authors[1].family_name or self.authors[1].name}"
                )
            else:
                parts.append(f"{self.authors[0].family_name or self.authors[0].name} et al.")
        
        # Year
        if self.year:
            parts.append(f"({self.year})")
        
        # Title
        parts.append(self.title)
        
        # Venue
        if self.venue:
            parts.append(self.venue)
        
        # DOI
        if self.doi:
            parts.append(f"https://doi.org/{self.doi}")
        
        return ". ".join(parts)
    
    def merge_from(self, other: "Publication") -> None:
        """
        Merge data from another Publication instance.
        
        Useful for combining data from multiple sources.
        """
        # Merge identifiers (keep non-None values)
        if not self.doi and other.doi:
            self.doi = other.doi
        if not self.pmid and other.pmid:
            self.pmid = other.pmid
        if not self.pmcid and other.pmcid:
            self.pmcid = other.pmcid
        if not self.arxiv_id and other.arxiv_id:
            self.arxiv_id = other.arxiv_id
        if not self.openalex_id and other.openalex_id:
            self.openalex_id = other.openalex_id
        if not self.semantic_scholar_id and other.semantic_scholar_id:
            self.semantic_scholar_id = other.semantic_scholar_id
        
        # Merge content
        if not self.abstract and other.abstract:
            self.abstract = other.abstract
        if not self.full_text and other.full_text:
            self.full_text = other.full_text
        
        # Merge URLs
        if not self.pdf_url and other.pdf_url:
            self.pdf_url = other.pdf_url
        
        # Merge metrics (take higher values)
        self.citation_count = max(self.citation_count, other.citation_count)
        if other.influential_citation_count:
            if self.influential_citation_count:
                self.influential_citation_count = max(
                    self.influential_citation_count,
                    other.influential_citation_count
                )
            else:
                self.influential_citation_count = other.influential_citation_count
        
        # Merge lists (deduplicate)
        self.concepts = list(set(self.concepts) | set(other.concepts))
        self.keywords = list(set(self.keywords) | set(other.keywords))
        self.mesh_terms = list(set(self.mesh_terms) | set(other.mesh_terms))
        self.sources = list(set(self.sources) | set(other.sources))
    
    def to_bibtex(self) -> str:
        """Generate BibTeX entry."""
        # Generate key
        if self.authors:
            first_author = self.authors[0].family_name or self.authors[0].name.split()[-1]
        else:
            first_author = "unknown"
        year = self.year or "nd"
        key = f"{first_author.lower()}{year}"
        
        # Entry type
        type_map = {
            PublicationType.ARTICLE: "article",
            PublicationType.BOOK: "book",
            PublicationType.BOOK_CHAPTER: "incollection",
            PublicationType.CONFERENCE: "inproceedings",
            PublicationType.THESIS: "phdthesis",
            PublicationType.PREPRINT: "misc",
        }
        entry_type = type_map.get(self.publication_type, "misc")
        
        lines = [f"@{entry_type}{{{key},"]
        lines.append(f'  title = {{{self.title}}},')
        
        if self.authors:
            authors_str = " and ".join(a.name for a in self.authors)
            lines.append(f'  author = {{{authors_str}}},')
        
        if self.year:
            lines.append(f'  year = {{{self.year}}},')
        
        if self.venue:
            if entry_type == "article":
                lines.append(f'  journal = {{{self.venue}}},')
            else:
                lines.append(f'  booktitle = {{{self.venue}}},')
        
        if self.volume:
            lines.append(f'  volume = {{{self.volume}}},')
        if self.issue:
            lines.append(f'  number = {{{self.issue}}},')
        if self.pages:
            lines.append(f'  pages = {{{self.pages}}},')
        if self.doi:
            lines.append(f'  doi = {{{self.doi}}},')
        
        lines.append("}")
        return "\n".join(lines)
