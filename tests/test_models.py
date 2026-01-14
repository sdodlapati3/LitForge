"""Tests for Publication model."""

import pytest
from datetime import date


class TestPublication:
    """Tests for the Publication model."""
    
    def test_create_publication(self, sample_publication):
        """Test creating a publication."""
        assert sample_publication.title == "CRISPR-Cas9 gene editing in human cells"
        assert len(sample_publication.authors) == 2
        assert sample_publication.doi == "10.1234/example.2024"
        assert sample_publication.year == 2024
    
    def test_publication_to_dict(self, sample_publication):
        """Test converting publication to dict."""
        data = sample_publication.to_dict()
        
        assert data["title"] == sample_publication.title
        assert data["doi"] == sample_publication.doi
        assert len(data["authors"]) == 2
    
    def test_publication_to_bibtex(self, sample_publication):
        """Test BibTeX export."""
        bibtex = sample_publication.to_bibtex()
        
        assert "@article{" in bibtex
        assert "CRISPR" in bibtex
        assert sample_publication.doi in bibtex
    
    def test_publication_merge(self, sample_publication):
        """Test merging publications."""
        from litforge.models import Author, Publication
        
        other = Publication(
            title=sample_publication.title,
            authors=[Author(name="New Author")],
            pmid="12345678",
            citation_count=150,
        )
        
        sample_publication.merge_from(other)
        
        # Should take higher citation count
        assert sample_publication.citation_count == 150
        # Should add pmid
        assert sample_publication.pmid == "12345678"


class TestAuthor:
    """Tests for the Author model."""
    
    def test_create_author(self):
        """Test creating an author."""
        from litforge.models import Author
        
        author = Author(
            name="John Smith",
            orcid="0000-0001-2345-6789",
        )
        
        assert author.name == "John Smith"
        assert author.orcid == "0000-0001-2345-6789"
    
    def test_author_equality(self):
        """Test author comparison."""
        from litforge.models import Author
        
        a1 = Author(name="John Smith", orcid="0000-0001-2345-6789")
        a2 = Author(name="John Smith", orcid="0000-0001-2345-6789")
        a3 = Author(name="Jane Doe")
        
        assert a1.name == a2.name
        assert a1.name != a3.name


class TestSearchModels:
    """Tests for search-related models."""
    
    def test_search_query(self):
        """Test SearchQuery model."""
        from litforge.models import SearchQuery
        
        query = SearchQuery(
            query="CRISPR",
            sources=["openalex", "pubmed"],
            limit=50,
        )
        
        assert query.query == "CRISPR"
        assert "openalex" in query.sources
        assert query.limit == 50
    
    def test_search_filter(self):
        """Test SearchFilter model."""
        from litforge.models import SearchFilter, PublicationType
        
        filters = SearchFilter(
            year_from=2020,
            year_to=2024,
            types=[PublicationType.ARTICLE, PublicationType.REVIEW],
            open_access=True,
        )
        
        assert filters.year_from == 2020
        assert filters.year_to == 2024
        assert filters.open_access is True


class TestNetworkModels:
    """Tests for citation network models."""
    
    def test_citation_network(self, sample_publications):
        """Test CitationNetwork model."""
        from litforge.models import CitationNetwork, NetworkNode, NetworkEdge
        
        nodes = [
            NetworkNode(
                id=pub.doi,
                publication=pub,
                in_degree=0,
                out_degree=0,
            )
            for pub in sample_publications
        ]
        
        edges = [
            NetworkEdge(
                source=sample_publications[0].doi,
                target=sample_publications[1].doi,
            ),
        ]
        
        network = CitationNetwork(
            seed_papers=[sample_publications[0].doi],
            nodes=nodes,
            edges=edges,
        )
        
        assert len(network.nodes) == 3
        assert len(network.edges) == 1
    
    def test_network_to_networkx(self, sample_publications):
        """Test NetworkX export."""
        from litforge.models import CitationNetwork, NetworkNode, NetworkEdge
        
        nodes = [
            NetworkNode(
                id=pub.doi,
                publication=pub,
                in_degree=0,
                out_degree=0,
            )
            for pub in sample_publications
        ]
        
        network = CitationNetwork(
            seed_papers=[],
            nodes=nodes,
            edges=[],
        )
        
        G = network.to_networkx()
        
        assert G.number_of_nodes() == 3
