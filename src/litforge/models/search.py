"""
Search-related data models.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from litforge.models.publication import Publication


class SortBy(str, Enum):
    """Sort options for search results."""
    RELEVANCE = "relevance"
    DATE = "date"
    CITATIONS = "citations"
    TITLE = "title"


class SortOrder(str, Enum):
    """Sort order."""
    ASC = "asc"
    DESC = "desc"


class SearchFilter(BaseModel):
    """Filters for search queries."""
    
    # Date filters
    year_from: int | None = Field(default=None, description="Minimum publication year")
    year_to: int | None = Field(default=None, description="Maximum publication year")
    date_from: date | None = Field(default=None, description="Minimum publication date")
    date_to: date | None = Field(default=None, description="Maximum publication date")
    
    # Type filters
    publication_types: list[str] | None = Field(default=None, description="Filter by publication type")
    open_access_only: bool = Field(default=False, description="Only open access papers")
    has_full_text: bool = Field(default=False, description="Only papers with full text available")
    
    # Content filters
    fields_of_study: list[str] | None = Field(default=None, description="Filter by field of study")
    concepts: list[str] | None = Field(default=None, description="Filter by concept")
    venues: list[str] | None = Field(default=None, description="Filter by venue/journal")
    
    # Author filters
    authors: list[str] | None = Field(default=None, description="Filter by author name")
    institutions: list[str] | None = Field(default=None, description="Filter by institution")
    
    # Citation filters
    min_citations: int | None = Field(default=None, description="Minimum citation count")
    max_citations: int | None = Field(default=None, description="Maximum citation count")
    
    # Identifier filters
    dois: list[str] | None = Field(default=None, description="Specific DOIs to find")
    pmids: list[str] | None = Field(default=None, description="Specific PMIDs to find")


class SearchQuery(BaseModel):
    """A search query with all parameters."""
    
    # Query text
    query: str = Field(..., description="Search query text")
    
    # Sources
    sources: list[str] = Field(
        default=["openalex", "semantic_scholar"],
        description="Data sources to search"
    )
    
    # Filters
    filters: SearchFilter = Field(default_factory=SearchFilter)
    
    # Pagination
    limit: int = Field(default=50, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    
    # Sorting
    sort_by: SortBy = Field(default=SortBy.RELEVANCE, description="Sort field")
    sort_order: SortOrder = Field(default=SortOrder.DESC, description="Sort order")
    
    # Options
    include_abstracts: bool = Field(default=True, description="Include abstracts")
    include_authors: bool = Field(default=True, description="Include author details")
    include_citations: bool = Field(default=False, description="Include citation data")
    deduplicate: bool = Field(default=True, description="Deduplicate across sources")


class SearchResult(BaseModel):
    """Results from a search query."""
    
    # Query info
    query: SearchQuery = Field(..., description="Original query")
    
    # Results
    publications: list[Publication] = Field(default_factory=list, description="Found publications")
    
    # Metadata
    total_count: int = Field(default=0, description="Total matching papers (may be estimate)")
    returned_count: int = Field(default=0, description="Number of papers returned")
    
    # Source breakdown
    source_counts: dict[str, int] = Field(default_factory=dict, description="Count per source")
    
    # Timing
    search_time_ms: float = Field(default=0, description="Search time in milliseconds")
    
    # Facets (if available)
    facets: dict[str, dict[str, int]] | None = Field(
        default=None,
        description="Facet counts (e.g., by year, venue)"
    )
    
    # Pagination
    has_more: bool = Field(default=False, description="More results available")
    next_offset: int | None = Field(default=None, description="Offset for next page")
    
    def __len__(self) -> int:
        return len(self.publications)
    
    def __iter__(self):
        return iter(self.publications)
    
    def __getitem__(self, index: int) -> Publication:
        return self.publications[index]
    
    @property
    def papers(self) -> list[Publication]:
        """Alias for publications."""
        return self.publications
