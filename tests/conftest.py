"""Pytest configuration."""

import pytest


@pytest.fixture
def sample_publication():
    """Create a sample publication for testing."""
    from litforge.models import Author, Publication, PublicationType, AccessType
    
    return Publication(
        id="test-pub-001",
        title="CRISPR-Cas9 gene editing in human cells",
        authors=[
            Author(name="John Smith", orcid="0000-0001-2345-6789"),
            Author(name="Jane Doe"),
        ],
        abstract="This paper describes the use of CRISPR-Cas9 for gene editing.",
        doi="10.1234/example.2024",
        year=2024,
        venue="Nature Methods",
        publication_type=PublicationType.ARTICLE,
        citation_count=100,
        access_type=AccessType.OPEN,
    )


@pytest.fixture
def sample_publications(sample_publication):
    """Create multiple sample publications."""
    from litforge.models import Author, Publication
    
    return [
        sample_publication,
        Publication(
            id="test-pub-002",
            title="Machine learning for drug discovery",
            authors=[Author(name="Alice Johnson")],
            abstract="Using ML models to predict drug candidates.",
            doi="10.1234/ml.2024",
            year=2023,
            venue="Journal of Cheminformatics",
            citation_count=50,
        ),
        Publication(
            id="test-pub-003",
            title="Deep learning in genomics",
            authors=[Author(name="Bob Wilson")],
            abstract="Neural networks for genomic analysis.",
            doi="10.1234/dl.2024",
            year=2022,
            citation_count=75,
        ),
    ]
