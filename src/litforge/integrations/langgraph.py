"""
LangGraph integration for LitForge.

Provides LangGraph-compatible tools and toolkit for scientific literature operations.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Annotated, Any

logger = logging.getLogger(__name__)


class LitForgeToolkit:
    """
    LitForge toolkit for LangGraph.
    
    Provides a set of tools compatible with LangGraph's tool-calling framework
    for literature search, retrieval, citation analysis, and Q&A.
    
    Usage:
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from litforge.integrations.langgraph import LitForgeToolkit
        
        toolkit = LitForgeToolkit()
        tools = toolkit.get_tools()
        
        llm = ChatOpenAI(model="gpt-4o-mini")
        agent = create_react_agent(llm, tools)
    """

    def __init__(self, forge: Any = None):
        """
        Initialize LitForge toolkit for LangGraph.
        
        Args:
            forge: Optional Forge instance. If not provided, will be created lazily.
        """
        self._forge = forge

    @property
    def forge(self) -> Any:
        """Get or create Forge instance."""
        if self._forge is None:
            from litforge.core.forge import Forge
            self._forge = Forge()
        return self._forge

    def get_tools(self) -> list[Callable]:
        """
        Get all available tools as LangGraph-compatible functions.
        
        Returns:
            List of tool functions decorated for LangGraph.
        """
        try:
            from langchain_core.tools import tool
        except ImportError:
            raise ImportError(
                "LangChain core not installed. Install with: pip install langchain-core"
            )

        forge = self.forge

        @tool
        async def search_papers(
            query: Annotated[str, "Search query for finding scientific papers"],
            limit: Annotated[int, "Maximum number of papers to return"] = 20,
        ) -> str:
            """
            Search for scientific papers across OpenAlex, Semantic Scholar, PubMed, and arXiv.
            
            Returns paper metadata including titles, authors, abstracts, and DOIs.
            """
            try:
                papers = await forge.search(query=query, limit=limit)
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

        @tool
        async def lookup_paper(
            doi: Annotated[str, "Digital Object Identifier (DOI) of the paper"],
        ) -> str:
            """
            Look up a specific paper by its DOI.
            
            Returns detailed paper metadata including full abstract and citation information.
            """
            try:
                paper = await forge.lookup(doi=doi.strip())
                if paper:
                    return json.dumps({
                        "doi": paper.doi,
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors] if paper.authors else [],
                        "year": paper.year,
                        "abstract": paper.abstract,
                        "venue": paper.venue,
                        "citation_count": paper.citation_count,
                        "open_access": paper.open_access,
                    }, indent=2)
                return "Paper not found"
            except Exception as e:
                return f"Error looking up paper: {e}"

        @tool
        async def retrieve_fulltext(
            doi: Annotated[str, "DOI of the paper to retrieve full text for"],
        ) -> str:
            """
            Download and extract full text from a paper.
            
            Finds open access versions from Unpaywall, PMC, or arXiv.
            """
            try:
                content = await forge.retrieve(doi=doi.strip())
                if content:
                    if len(content) > 15000:
                        return content[:15000] + "\n\n[Content truncated...]"
                    return content
                return "Full text not available"
            except Exception as e:
                return f"Error retrieving full text: {e}"

        @tool
        async def get_citations(
            doi: Annotated[str, "DOI of the paper to get citations for"],
            limit: Annotated[int, "Maximum number of citing papers to return"] = 20,
        ) -> str:
            """
            Get papers that cite a given paper.
            
            Useful for finding follow-up research and assessing impact.
            """
            try:
                citations = await forge.get_citations(doi=doi.strip(), limit=limit)
                results = []
                for p in citations:
                    results.append({
                        "doi": p.doi,
                        "title": p.title,
                        "year": p.year,
                        "citation_count": p.citation_count,
                    })
                return json.dumps(results, indent=2)
            except Exception as e:
                return f"Error getting citations: {e}"

        @tool
        async def get_references(
            doi: Annotated[str, "DOI of the paper to get references for"],
            limit: Annotated[int, "Maximum number of referenced papers to return"] = 20,
        ) -> str:
            """
            Get papers referenced by a given paper.
            
            Useful for understanding foundational work and context.
            """
            try:
                refs = await forge.get_references(doi=doi.strip(), limit=limit)
                results = []
                for p in refs:
                    results.append({
                        "doi": p.doi,
                        "title": p.title,
                        "year": p.year,
                        "citation_count": p.citation_count,
                    })
                return json.dumps(results, indent=2)
            except Exception as e:
                return f"Error getting references: {e}"

        @tool
        async def index_papers(
            dois: Annotated[list[str], "List of DOIs to index for Q&A"],
            include_fulltext: Annotated[bool, "Whether to include full text in index"] = False,
        ) -> str:
            """
            Index papers into the knowledge base for semantic search and Q&A.
            
            Papers must be indexed before using the ask_papers tool.
            """
            try:
                indexed = await forge.index(dois=dois, include_fulltext=include_fulltext)
                return f"Successfully indexed {indexed} papers"
            except Exception as e:
                return f"Error indexing papers: {e}"

        @tool
        async def ask_papers(
            question: Annotated[str, "Question to answer based on indexed papers"],
            max_sources: Annotated[int, "Maximum number of source papers to cite"] = 5,
        ) -> str:
            """
            Ask a question about indexed papers using RAG.
            
            Returns an answer with citations to source papers.
            Must index papers first using index_papers tool.
            """
            try:
                answer = await forge.ask(question=question, max_sources=max_sources)
                return json.dumps({
                    "answer": answer.text,
                    "citations": [
                        {"doi": c.doi, "title": c.title, "relevance": c.relevance}
                        for c in answer.citations
                    ],
                }, indent=2)
            except Exception as e:
                return f"Error asking papers: {e}"

        @tool
        async def build_citation_network(
            seed_dois: Annotated[list[str], "DOIs of seed papers to start from"],
            depth: Annotated[int, "Citation levels to traverse (1-3)"] = 1,
            max_papers: Annotated[int, "Maximum total papers in network"] = 100,
        ) -> str:
            """
            Build a citation network starting from seed papers.
            
            Useful for exploring research landscapes and finding influential papers.
            """
            try:
                network = await forge.build_network(
                    seed_dois=seed_dois,
                    depth=depth,
                    max_papers=max_papers,
                )
                return json.dumps({
                    "node_count": network.node_count,
                    "edge_count": network.edge_count,
                    "most_cited": [
                        {
                            "doi": p.doi,
                            "title": p.title,
                            "citation_count": p.citation_count,
                        }
                        for p in network.most_cited(5)
                    ],
                }, indent=2)
            except Exception as e:
                return f"Error building network: {e}"

        return [
            search_papers,
            lookup_paper,
            retrieve_fulltext,
            get_citations,
            get_references,
            index_papers,
            ask_papers,
            build_citation_network,
        ]


def create_literature_agent(
    llm: Any = None,
    forge: Any = None,
    system_prompt: str | None = None,
) -> Any:
    """
    Create a LangGraph ReAct agent with LitForge tools.
    
    Args:
        llm: LangChain-compatible LLM. If not provided, uses ChatOpenAI.
        forge: Optional Forge instance.
        system_prompt: Custom system prompt for the agent.
        
    Returns:
        LangGraph agent executor.
    """
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        raise ImportError(
            "LangGraph not installed. Install with: pip install langgraph"
        )

    if llm is None:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini")
        except ImportError:
            raise ImportError(
                "LangChain OpenAI not installed. Install with: pip install langchain-openai"
            )

    toolkit = LitForgeToolkit(forge)
    tools = toolkit.get_tools()

    default_prompt = """You are a helpful research assistant with access to scientific literature databases.

You can search for papers, look up specific papers by DOI, retrieve full text, 
analyze citation networks, and answer questions about indexed papers.

When helping users with literature research:
1. Start by searching for relevant papers based on their query
2. Look up specific papers for more details when needed
3. Use citation analysis to explore related work
4. Index papers and use Q&A for in-depth analysis

Always cite your sources with DOIs when providing information from papers."""

    return create_react_agent(
        llm,
        tools,
        state_modifier=system_prompt or default_prompt,
    )
