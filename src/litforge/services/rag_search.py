"""
RAG-Enhanced Literature Search Pipeline.

This module implements a multi-stage search pipeline:
1. Scout Search - Quick preliminary search to gather context
2. RAG Augmentation - Feed scout results to LLM for informed expansion
3. Multi-API Search - Search across multiple sources with expanded queries
4. Deduplication - Remove duplicates by DOI/title
5. Ensemble Scoring - Rank with multiple signals
6. LLM Verification - Verify only if uncertain

Architecture:
    User Query 
        ↓
    [Scout Search] → Top-5 quick results (OpenAlex)
        ↓
    [RAG-Augmented LLM] → Informed query expansion
        ↓
    [Multi-API Search] → OpenAlex + Semantic Scholar + arXiv
        ↓
    [Deduplicate] → By DOI, then by title similarity
        ↓
    [Ensemble Score] → Keyword + Embedding + Citations + Recency
        ↓
    [LLM Verify] → Only if uncertainty > threshold
"""

from __future__ import annotations

import logging
import re
from typing import Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoutResult:
    """Results from scout search."""
    titles: list[str]
    authors: list[str]
    keywords: list[str]
    sample_abstracts: list[str]


def scout_search(
    query: str,
    limit: int = 5,
) -> ScoutResult:
    """
    Quick preliminary search to gather context for LLM.
    
    Uses OpenAlex for speed (no API key needed).
    Extracts key terms from natural language and applies ML context hints.
    
    Args:
        query: User's raw query (can be natural language)
        limit: Number of scout results
        
    Returns:
        ScoutResult with titles, authors, and keywords
    """
    try:
        from litforge.clients.openalex import OpenAlexClient
        
        client = OpenAlexClient()
        query_lower = query.lower()
        
        # Extract the core topic from natural language using a single comprehensive regex
        import re
        
        # Pattern to match common request structures and extract the topic
        # Matches: "I want you to find me papers about X" → captures "X"
        extraction_pattern = r"""
            (?:i\s+)?                                    # optional "I "
            (?:want|need|would\s+like)?                  # want/need
            (?:\s+you)?                                  # optional "you"
            (?:\s+to)?                                   # optional "to"
            (?:\s+(?:find|search|look\s+for|get|show))? # action verb
            (?:\s+me)?                                   # optional "me"
            (?:\s+all)?                                  # optional "all"
            (?:\s+the)?                                  # optional "the"
            (?:\s+(?:papers?|articles?|publications?|research))? # document type
            (?:\s+(?:on|about|related\s+to|that\s+(?:are\s+)?related\s+to|that))? # connector
            \s*(.+)                                      # THE TOPIC (captured)
        """
        
        match = re.match(extraction_pattern, query_lower, re.VERBOSE | re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
        else:
            # Fallback: just use the query
            topic = query_lower
        
        # Clean any remaining noise words at the start
        topic = re.sub(r"^(related\s+to|about|on)\s+", "", topic)
        topic = " ".join(topic.split()).strip()
        
        logger.info(f"Scout: extracted topic '{topic}' from '{query_lower}'")
        
        # Apply ML-specific context hints for ambiguous terms
        # Use specific search terms that will find the actual papers
        ml_hints = {
            "liquid foundation": "Liquid Time-constant Networks Hasani",  # Original LTC paper
            "liquid model": "Liquid Time-constant Networks",
            "liquid neural": "Liquid Time-constant Networks",
            "liquid": "Liquid Time-constant Networks Hasani Lechner",  # Key authors
            "foundation model": "foundation models large language GPT BERT",
            "transformer": "transformer attention neural network Vaswani",
        }
        
        enhanced_query = topic
        for term, enhancement in ml_hints.items():
            if term in topic:
                enhanced_query = enhancement
                logger.info(f"Scout: enhanced '{topic}' → '{enhanced_query}'")
                break
        
        # If no enhancement matched, use the extracted topic
        if enhanced_query == topic and topic:
            logger.info(f"Scout: using extracted topic '{topic}'")
        
        results = client.search(enhanced_query, limit=limit)
        
        titles = []
        authors = set()
        keywords = set()
        abstracts = []
        
        for paper in results:
            if paper.title:
                titles.append(paper.title)
            
            # Extract authors
            if hasattr(paper, 'authors') and paper.authors:
                if isinstance(paper.authors, list):
                    for a in paper.authors[:2]:  # Top 2 authors per paper
                        if isinstance(a, dict):
                            name = a.get('name', '')
                        else:
                            name = str(a)
                        if name:
                            authors.add(name)
            
            # Extract keywords from title
            if paper.title:
                words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', paper.title)
                for w in words:
                    if len(w) > 3:
                        keywords.add(w)
            
            # Get abstract snippet
            if paper.abstract:
                abstracts.append(paper.abstract[:200])
        
        return ScoutResult(
            titles=titles,
            authors=list(authors)[:5],
            keywords=list(keywords)[:10],
            sample_abstracts=abstracts[:3],
        )
        
    except Exception as e:
        logger.warning(f"Scout search failed: {e}")
        return ScoutResult(titles=[], authors=[], keywords=[], sample_abstracts=[])


def expand_with_rag(
    query: str,
    scout: ScoutResult,
    llm: Any,
) -> dict:
    """
    Use LLM with RAG context to expand query intelligently.
    
    Args:
        query: User's raw query
        scout: Scout search results for context
        llm: LLM instance
        
    Returns:
        Expansion dict with search_queries, must_contain, exclude_terms
    """
    import json
    
    # Build RAG context from scout results
    rag_context = ""
    if scout.titles:
        rag_context += f"\n\nRELATED PAPERS FOUND:\n"
        for i, t in enumerate(scout.titles[:5], 1):
            rag_context += f"{i}. {t}\n"
    
    if scout.authors:
        rag_context += f"\nKEY AUTHORS IN THIS AREA: {', '.join(scout.authors)}\n"
    
    if scout.keywords:
        rag_context += f"\nCOMMON TERMS: {', '.join(scout.keywords)}\n"
    
    if scout.sample_abstracts:
        rag_context += f"\nSAMPLE ABSTRACT SNIPPET: {scout.sample_abstracts[0][:150]}...\n"
    
    prompt = f"""You are a scientific literature search expert specializing in COMPUTER SCIENCE and MACHINE LEARNING.

USER'S REQUEST (natural language): "{query}"
{rag_context}

FIRST: Extract the core research topic from the user's natural language request.
For example: "I want you to find me all the papers related to liquid foundation models" → topic is "liquid foundation models"

CRITICAL: Interpret topics in the context of MACHINE LEARNING / AI:
- "liquid" in ML = "Liquid Neural Networks" or "Liquid Time-constant Networks" (NOT chemistry)
- "foundation models" = Large pretrained AI models like GPT, BERT, LLaMA
- "transformer" = Attention-based neural networks (NOT electrical transformers)

If the related papers above seem UNRELATED to ML/AI, IGNORE them and focus on ML interpretation.

KEY RESEARCHERS for Liquid Networks: Ramin Hasani, Daniela Rus, Mathias Lechner (MIT)

Generate 3-5 search queries that will find relevant ML/AI papers.

Return ONLY valid JSON:
{{
    "search_queries": ["query1", "query2", "query3"],
    "must_contain": ["key_term1", "key_term2"],
    "exclude_terms": ["unrelated_term1"],
    "target_authors": ["Author Name"],
    "explanation": "Brief explanation of search strategy"
}}
"""
    
    try:
        response = llm.generate(
            prompt=prompt,
            system_prompt="You are a scientific search expert. Use the provided context to inform your expansion. Return only valid JSON.",
            max_tokens=600
        )
        
        # Parse JSON using shared utility
        from litforge.services.utils import parse_llm_json
        result = parse_llm_json(response)
        
        if not result:
            return None
        
        result["provider"] = llm.provider_name if hasattr(llm, 'provider_name') else "unknown"
        result["rag_enhanced"] = True
        
        return result
        
    except Exception as e:
        logger.warning(f"RAG expansion failed: {e}")
        return None


def multi_api_search(
    queries: list[str],
    limit_per_query: int = 15,
    year_from: int | None = None,
    year_to: int | None = None,
    use_semantic_scholar: bool = True,
    use_arxiv: bool = True,
) -> list[Any]:
    """
    Search across multiple APIs for maximum coverage.
    
    Args:
        queries: List of search queries
        limit_per_query: Max results per query per API
        year_from: Filter by year
        year_to: Filter by year
        use_semantic_scholar: Include Semantic Scholar
        use_arxiv: Include arXiv
        
    Returns:
        Combined list of papers from all sources
    """
    all_papers = []
    
    # Build filters if needed
    search_filter = None
    if year_from or year_to:
        from litforge.models import SearchFilter
        search_filter = SearchFilter(year_from=year_from, year_to=year_to)
    
    # 1. OpenAlex (always, no API key needed)
    try:
        from litforge.clients.openalex import OpenAlexClient
        client = OpenAlexClient()
        
        for q in queries[:4]:  # Limit queries
            try:
                results = client.search(
                    q, 
                    limit=limit_per_query,
                    filters=search_filter,
                )
                all_papers.extend(results)
                logger.info(f"OpenAlex: {len(results)} results for '{q[:30]}...'")
            except Exception as e:
                logger.warning(f"OpenAlex search failed for '{q}': {e}")
    except ImportError:
        logger.warning("OpenAlex client not available")
    
    # 2. Semantic Scholar (if enabled)
    if use_semantic_scholar:
        try:
            from litforge.clients.semantic_scholar import SemanticScholarClient
            client = SemanticScholarClient()
            
            for q in queries[:3]:  # Limit to avoid rate limits
                try:
                    results = client.search(q, limit=limit_per_query)
                    all_papers.extend(results)
                    logger.info(f"Semantic Scholar: {len(results)} results for '{q[:30]}...'")
                except Exception as e:
                    logger.warning(f"Semantic Scholar search failed: {e}")
        except ImportError:
            logger.warning("Semantic Scholar client not available")
    
    # 3. arXiv (if enabled)
    if use_arxiv:
        try:
            from litforge.clients.arxiv import ArxivClient
            client = ArxivClient()
            
            for q in queries[:2]:  # arXiv is slower
                try:
                    results = client.search(q, limit=limit_per_query)
                    all_papers.extend(results)
                    logger.info(f"arXiv: {len(results)} results for '{q[:30]}...'")
                except Exception as e:
                    logger.warning(f"arXiv search failed: {e}")
        except ImportError:
            logger.warning("arXiv client not available")
    
    logger.info(f"Multi-API search: {len(all_papers)} total raw results")
    return all_papers


# Import from shared utils
from litforge.services.utils import deduplicate_papers


def rag_search_pipeline(
    query: str,
    llm: Any | None = None,
    max_results: int = 25,
    year_from: int | None = None,
    year_to: int | None = None,
    use_scout: bool = True,
    use_multi_api: bool = True,
    use_local_index: bool = True,
    index_name: str = "default",
) -> tuple[list[Any], dict]:
    """
    Full RAG-enhanced search pipeline with optional local index.
    
    Args:
        query: User's raw query
        llm: LLM instance for expansion
        max_results: Maximum results to return
        year_from: Filter by year
        year_to: Filter by year
        use_scout: Enable scout search for RAG
        use_multi_api: Enable multi-API search
        use_local_index: Enable local embedding index search
        index_name: Name of local index to use
        
    Returns:
        (papers, metadata) tuple
    """
    metadata = {
        "query": query,
        "scout_enabled": use_scout,
        "multi_api": use_multi_api,
        "use_local_index": use_local_index,
        "rag_enhanced": False,
    }
    
    # Step 1: Scout search (quick preliminary)
    scout = None
    if use_scout:
        scout = scout_search(query, limit=5)
        metadata["scout_results"] = len(scout.titles)
        metadata["scout_authors"] = scout.authors
    
    # Step 2: LLM expansion (with or without RAG)
    expansion = None
    if llm:
        if scout and scout.titles:
            # RAG-augmented expansion
            expansion = expand_with_rag(query, scout, llm)
            if expansion:
                metadata["rag_enhanced"] = True
        else:
            # Fall back to basic expansion using understand_query from semantic_search
            try:
                from litforge.services.semantic_search import understand_query
                understanding = understand_query(query, llm)
                if understanding:
                    expansion = {
                        "search_queries": understanding.search_terms,
                        "must_contain": [],
                        "exclude_terms": understanding.exclude_domains,
                        "target_authors": understanding.seed_authors,
                        "explanation": understanding.explanation,
                        "provider": llm.provider_name if hasattr(llm, 'provider_name') else "unknown",
                    }
            except Exception as e:
                logger.warning(f"Basic LLM expansion failed: {e}")
    
    if expansion:
        metadata["expansion"] = {
            "queries": expansion.get("search_queries", []),
            "provider": expansion.get("provider", "unknown"),
            "explanation": expansion.get("explanation", ""),
        }
    
    # Step 3: Multi-API search + Local Index
    search_queries = expansion.get("search_queries", [query]) if expansion else [query]
    all_papers = []
    
    # Step 3a: Local index search (if available)
    if use_local_index:
        try:
            from litforge.services.hybrid_search import HybridSearchService
            
            hybrid = HybridSearchService(
                index_name=index_name,
                use_semantic_scholar=False,  # We'll do API separately
                use_openalex=False,
            )
            
            if hybrid.index is not None and hybrid.index.size > 0:
                logger.info(f"Searching local index '{index_name}' ({hybrid.index.size} papers)")
                
                # Search with primary query
                results = hybrid.search(query, k=30, api_limit=0)
                local_papers = hybrid.get_publications(results)
                all_papers.extend(local_papers)
                
                metadata["local_index"] = {
                    "name": index_name,
                    "size": hybrid.index.size,
                    "results": len(local_papers),
                }
                logger.info(f"Local index: {len(local_papers)} results")
            else:
                logger.info("Local index empty or not available")
                metadata["local_index"] = {"available": False}
        except Exception as e:
            logger.warning(f"Local index search failed: {e}")
            metadata["local_index"] = {"error": str(e)}
    
    # Step 3b: Multi-API search
    if use_multi_api:
        api_papers = multi_api_search(
            queries=search_queries,
            limit_per_query=15,
            year_from=year_from,
            year_to=year_to,
        )
        all_papers.extend(api_papers)
    else:
        # Single API fallback
        from litforge.api import search
        all_papers = []
        for q in search_queries[:4]:
            results = search(q, limit=max_results // 2, year_from=year_from, year_to=year_to)
            all_papers.extend(results)
    
    metadata["raw_results"] = len(all_papers)
    
    # Step 4: Deduplicate
    papers = deduplicate_papers(all_papers)
    metadata["after_dedup"] = len(papers)
    
    # Step 5: Ensemble scoring (imported from scoring module)
    try:
        from litforge.services.scoring import score_and_verify, EnsembleScorer
        
        must_contain = [t.lower() for t in expansion.get("must_contain", [])] if expansion else []
        exclude_terms = [t.lower() for t in expansion.get("exclude_terms", [])] if expansion else []
        target_authors = expansion.get("target_authors", []) if expansion else []
        ai_insight = expansion.get("explanation", "") if expansion else ""
        
        scored_papers, scoring_meta = score_and_verify(
            papers=papers,
            query=query,
            must_contain=must_contain,
            exclude_terms=exclude_terms,
            target_authors=target_authors,
            ai_insight=ai_insight,
            llm=llm,
            verify_threshold=0.25,
            top_k=10,
            use_embeddings=True,
        )
        
        metadata["scoring"] = scoring_meta
        papers = [sp.paper for sp in scored_papers[:max_results]]
        
    except ImportError:
        logger.warning("Scoring module not available, using raw results")
    
    metadata["final_results"] = len(papers)
    
    return papers, metadata
