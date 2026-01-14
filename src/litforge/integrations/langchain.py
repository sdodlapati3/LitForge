"""
LangChain integration for LitForge.

Provides LangChain-compatible tools and retrievers for scientific literature operations.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LitForgeSearchTool:
    """
    LangChain tool for searching scientific papers.
    
    Usage:
        from langchain.agents import AgentExecutor, create_openai_tools_agent
        from litforge.integrations.langchain import LitForgeSearchTool
        
        tool = LitForgeSearchTool()
        agent = create_openai_tools_agent(llm, [tool], prompt)
    """

    name: str = "search_papers"
    description: str = (
        "Search for scientific papers across OpenAlex, Semantic Scholar, "
        "PubMed, and arXiv. Input should be a search query string. "
        "Returns paper metadata including titles, authors, abstracts, and DOIs."
    )

    def __init__(self, forge: Any = None):
        """Initialize the search tool."""
        self._forge = forge

    @property
    def forge(self) -> Any:
        """Get or create Forge instance."""
        if self._forge is None:
            from litforge.core.forge import Forge
            self._forge = Forge()
        return self._forge

    def _run(self, query: str, limit: int = 20) -> str:
        """Execute search synchronously."""
        import asyncio
        return asyncio.run(self._arun(query, limit))

    async def _arun(self, query: str, limit: int = 20) -> str:
        """Execute search asynchronously."""
        try:
            papers = await self.forge.search(query=query, limit=limit)
            results = []
            for p in papers:
                results.append({
                    "doi": p.doi,
                    "title": p.title,
                    "authors": [a.name for a in p.authors][:5] if p.authors else [],
                    "year": p.year,
                    "abstract": p.abstract[:500] if p.abstract else None,
                    "citation_count": p.citation_count,
                })
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error searching papers: {e}"

    @property
    def args_schema(self) -> type[Any]:
        """Get the args schema for the tool."""
        try:
            from pydantic import BaseModel, Field
        except ImportError:
            return None

        class SearchPapersInput(BaseModel):
            query: str = Field(description="Search query for finding papers")
            limit: int = Field(default=20, description="Maximum papers to return")

        return SearchPapersInput

    def as_tool(self) -> Any:
        """Convert to LangChain Tool object."""
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "LangChain core not installed. Install with: pip install langchain-core"
            )

        return StructuredTool.from_function(
            func=self._run,
            coroutine=self._arun,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )


class LitForgeRetrieveTool:
    """
    LangChain tool for retrieving full-text papers.
    """

    name: str = "retrieve_fulltext"
    description: str = (
        "Download and extract full text from a paper by DOI. "
        "Finds open access versions from Unpaywall, PMC, or arXiv."
    )

    def __init__(self, forge: Any = None):
        """Initialize the retrieve tool."""
        self._forge = forge

    @property
    def forge(self) -> Any:
        """Get or create Forge instance."""
        if self._forge is None:
            from litforge.core.forge import Forge
            self._forge = Forge()
        return self._forge

    def _run(self, doi: str) -> str:
        """Execute retrieval synchronously."""
        import asyncio
        return asyncio.run(self._arun(doi))

    async def _arun(self, doi: str) -> str:
        """Execute retrieval asynchronously."""
        try:
            content = await self.forge.retrieve(doi=doi.strip())
            if content:
                if len(content) > 15000:
                    return content[:15000] + "\n\n[Content truncated...]"
                return content
            return "Full text not available"
        except Exception as e:
            return f"Error retrieving full text: {e}"

    @property
    def args_schema(self) -> type[Any]:
        """Get the args schema for the tool."""
        try:
            from pydantic import BaseModel, Field
        except ImportError:
            return None

        class RetrieveInput(BaseModel):
            doi: str = Field(description="DOI of the paper to retrieve")

        return RetrieveInput

    def as_tool(self) -> Any:
        """Convert to LangChain Tool object."""
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "LangChain core not installed. Install with: pip install langchain-core"
            )

        return StructuredTool.from_function(
            func=self._run,
            coroutine=self._arun,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )


class LitForgeCitationTool:
    """
    LangChain tool for citation analysis.
    """

    name: str = "analyze_citations"
    description: str = (
        "Get papers that cite a given paper or papers referenced by it. "
        "Input: DOI and direction ('citations' or 'references')."
    )

    def __init__(self, forge: Any = None):
        """Initialize the citation tool."""
        self._forge = forge

    @property
    def forge(self) -> Any:
        """Get or create Forge instance."""
        if self._forge is None:
            from litforge.core.forge import Forge
            self._forge = Forge()
        return self._forge

    def _run(self, doi: str, direction: str = "citations", limit: int = 20) -> str:
        """Execute citation analysis synchronously."""
        import asyncio
        return asyncio.run(self._arun(doi, direction, limit))

    async def _arun(self, doi: str, direction: str = "citations", limit: int = 20) -> str:
        """Execute citation analysis asynchronously."""
        try:
            if direction == "citations":
                papers = await self.forge.get_citations(doi=doi.strip(), limit=limit)
            else:
                papers = await self.forge.get_references(doi=doi.strip(), limit=limit)

            results = []
            for p in papers:
                results.append({
                    "doi": p.doi,
                    "title": p.title,
                    "year": p.year,
                    "citation_count": p.citation_count,
                })
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error analyzing citations: {e}"

    @property
    def args_schema(self) -> type[Any]:
        """Get the args schema for the tool."""
        try:
            from pydantic import BaseModel, Field
        except ImportError:
            return None

        class CitationInput(BaseModel):
            doi: str = Field(description="DOI of the paper")
            direction: str = Field(
                default="citations",
                description="'citations' or 'references'"
            )
            limit: int = Field(default=20, description="Maximum papers to return")

        return CitationInput

    def as_tool(self) -> Any:
        """Convert to LangChain Tool object."""
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "LangChain core not installed. Install with: pip install langchain-core"
            )

        return StructuredTool.from_function(
            func=self._run,
            coroutine=self._arun,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )


class LitForgeQATool:
    """
    LangChain tool for Q&A over indexed papers.
    """

    name: str = "ask_papers"
    description: str = (
        "Ask a question about indexed papers using RAG. "
        "Returns an answer with citations. Papers must be indexed first."
    )

    def __init__(self, forge: Any = None):
        """Initialize the Q&A tool."""
        self._forge = forge

    @property
    def forge(self) -> Any:
        """Get or create Forge instance."""
        if self._forge is None:
            from litforge.core.forge import Forge
            self._forge = Forge()
        return self._forge

    def _run(self, question: str, max_sources: int = 5) -> str:
        """Execute Q&A synchronously."""
        import asyncio
        return asyncio.run(self._arun(question, max_sources))

    async def _arun(self, question: str, max_sources: int = 5) -> str:
        """Execute Q&A asynchronously."""
        try:
            answer = await self.forge.ask(question=question, max_sources=max_sources)
            return json.dumps({
                "answer": answer.text,
                "citations": [
                    {"doi": c.doi, "title": c.title, "relevance": c.relevance}
                    for c in answer.citations
                ],
            }, indent=2)
        except Exception as e:
            return f"Error asking papers: {e}"

    @property
    def args_schema(self) -> type[Any]:
        """Get the args schema for the tool."""
        try:
            from pydantic import BaseModel, Field
        except ImportError:
            return None

        class QAInput(BaseModel):
            question: str = Field(description="Question to answer")
            max_sources: int = Field(default=5, description="Max sources to cite")

        return QAInput

    def as_tool(self) -> Any:
        """Convert to LangChain Tool object."""
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "LangChain core not installed. Install with: pip install langchain-core"
            )

        return StructuredTool.from_function(
            func=self._run,
            coroutine=self._arun,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )


class LitForgeRetriever:
    """
    LangChain retriever for semantic search over indexed papers.
    
    Can be used with LangChain's RAG chains for custom Q&A pipelines.
    
    Usage:
        from langchain.chains import RetrievalQA
        from litforge.integrations.langchain import LitForgeRetriever
        
        retriever = LitForgeRetriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    """

    def __init__(
        self,
        forge: Any = None,
        k: int = 5,
        score_threshold: float = 0.0,
    ):
        """
        Initialize the retriever.
        
        Args:
            forge: Optional Forge instance.
            k: Number of documents to retrieve.
            score_threshold: Minimum similarity score.
        """
        self._forge = forge
        self.k = k
        self.score_threshold = score_threshold

    @property
    def forge(self) -> Any:
        """Get or create Forge instance."""
        if self._forge is None:
            from litforge.core.forge import Forge
            self._forge = Forge()
        return self._forge

    def _get_relevant_documents(self, query: str) -> list[Any]:
        """Get relevant documents synchronously."""
        import asyncio
        return asyncio.run(self._aget_relevant_documents(query))

    async def _aget_relevant_documents(self, query: str) -> list[Any]:
        """Get relevant documents asynchronously."""
        try:
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError(
                "LangChain core not installed. Install with: pip install langchain-core"
            )

        # Search the knowledge base
        results = await self.forge.knowledge.search(query=query, limit=self.k)

        documents = []
        for result in results:
            if result.score >= self.score_threshold:
                documents.append(
                    Document(
                        page_content=result.text,
                        metadata={
                            "doi": result.metadata.get("doi"),
                            "title": result.metadata.get("title"),
                            "score": result.score,
                        },
                    )
                )

        return documents

    def as_retriever(self) -> Any:
        """Convert to LangChain Retriever object."""
        try:
            from langchain_core.callbacks import CallbackManagerForRetrieverRun
            from langchain_core.documents import Document
            from langchain_core.retrievers import BaseRetriever
        except ImportError:
            raise ImportError(
                "LangChain core not installed. Install with: pip install langchain-core"
            )

        parent = self

        class LitForgeBaseRetriever(BaseRetriever):
            """LangChain BaseRetriever implementation."""

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun = None,
            ) -> list[Document]:
                return parent._get_relevant_documents(query)

            async def _aget_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun = None,
            ) -> list[Document]:
                return await parent._aget_relevant_documents(query)

        return LitForgeBaseRetriever()


def get_all_tools(forge: Any = None) -> list[Any]:
    """
    Get all LitForge tools as LangChain Tool objects.
    
    Args:
        forge: Optional Forge instance to share across tools.
        
    Returns:
        List of LangChain Tool objects.
    """
    return [
        LitForgeSearchTool(forge).as_tool(),
        LitForgeRetrieveTool(forge).as_tool(),
        LitForgeCitationTool(forge).as_tool(),
        LitForgeQATool(forge).as_tool(),
    ]
