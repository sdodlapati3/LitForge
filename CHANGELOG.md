# Changelog

All notable changes to LitForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Core data models (Publication, Citation, Author, Network)
- OpenAlex client for paper discovery
- Semantic Scholar client integration
- PubMed/Entrez client
- ChromaDB vector store integration
- PDF extraction with pymupdf
- Basic RAG pipeline for Q&A
- MCP server for AI agent integration
- CLI tool for command-line usage
- Configuration system with YAML/env support
- **Qdrant vector store** - Production-ready vector database with cloud/self-hosted support
- **FAISS vector store** - High-performance local vector search with GPU support
- **MCP server** - Model Context Protocol server for ChemAgent/OmicsOracle integration
- **CrewAI integration** - Full toolkit for CrewAI agents (search, retrieve, cite, ask)
- **LangGraph integration** - Tools and agent factory for LangGraph workflows
- **LangChain integration** - Tools and retriever for LangChain RAG pipelines

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2025-01-14

### Added
- Initial alpha release
- Project scaffolding and architecture
- Core interfaces and base classes
