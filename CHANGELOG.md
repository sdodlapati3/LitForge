# Changelog

All notable changes to LitForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-06-XX

### Added

#### PDF Processing & Text Extraction (Phase 2)
- **PDFExtractor** - Advanced PDF text extraction with multi-column layout detection, header/footer removal, and metadata extraction
- **SectionParser** - Parse scientific papers into standard sections (abstract, introduction, methods, results, discussion, conclusion, references, etc.)
- **ReferenceExtractor** - Extract references, DOIs, and arXiv IDs from paper text
- **`retrieve_with_sections()`** - New method to retrieve full text with automatic section parsing

#### RAG Pipeline Enhancements (Phase 3)
- **Section-aware indexing** - Index papers with section metadata for more targeted retrieval
- **Hybrid search** - Combine semantic similarity with keyword matching for better results
- **Section filtering** - Filter search results by section (e.g., only search methods sections)
- **Enhanced chunking** - Improved text chunking that respects section boundaries

#### Q&A Service Improvements (Phase 4)
- **QAResponse dataclass** - Structured response with answer, sources, confidence, and related questions
- **ChatResponse dataclass** - Structured chat response with context usage tracking
- **ComparisonResult dataclass** - Compare multiple papers with similarity/difference analysis
- **`compare()`** - Compare multiple papers to identify similarities and differences
- **`generate_follow_ups`** - Automatically generate related follow-up questions
- **Confidence scoring** - Rate answer confidence based on source availability

#### Citation Network Analysis (Phase 5)
- **`find_bridge_papers()`** - Find papers that connect different research clusters
- **`get_citation_timeline()`** - Get papers grouped by publication year
- **`export_json()`** - Export network to JSON format
- **`export_graphml()`** - Export network to GraphML for Gephi/Cytoscape
- **`get_network_stats()`** - Get comprehensive network statistics (density, degree distribution, etc.)
- **`find_key_papers()`** - Find most influential papers by PageRank, citations, or in-degree

#### CLI Enhancements (Phase 6)
- **`litforge network`** - Build and export citation networks from the command line
- **`litforge retrieve`** - Retrieve and extract full text with section parsing
- Improved error handling and output formatting
- All commands work correctly with new API

### Changed
- **SearchFilter** - Renamed `open_access` to `open_access_only` and `types` to `publication_types` for clarity
- **Publication ID generation** - All clients now generate consistent IDs using OpenAlex ID format
- **Keywords parsing** - Fixed None handling, now returns empty list
- **Citation network** - Changed `centrality` to `pagerank` for clarity, nodes now stored as dict

### Fixed
- Fixed Publication ID generation in all 5 API clients (OpenAlex, Semantic Scholar, PubMed, arXiv, Crossref)
- Fixed keywords parsing (None → empty list) in OpenAlex and Semantic Scholar clients
- Fixed citation network building (centrality → pagerank, nodes list → dict)
- Fixed Forge service initialization parameter mismatches
- Fixed SearchFilter attribute name mismatches
- Fixed CLI to use correct method signatures and return types
- Fixed PDF text extraction fallback when PyMuPDF not available

## [0.1.0] - 2025-01-14

### Added
- Initial alpha release
- Core data models (Publication, Citation, Author, Network)
- OpenAlex client for paper discovery
- Semantic Scholar client integration
- PubMed/Entrez client
- arXiv client
- Crossref client
- Unpaywall client for open access discovery
- ChromaDB vector store integration
- Qdrant vector store - Production-ready with cloud/self-hosted support
- FAISS vector store - High-performance local vector search
- PDF extraction with pymupdf
- Basic RAG pipeline for Q&A
- MCP server for AI agent integration
- CrewAI integration - Full toolkit for CrewAI agents
- LangGraph integration - Tools and agent factory
- LangChain integration - Tools and retriever
- CLI tool for command-line usage
- Configuration system with YAML/env support
