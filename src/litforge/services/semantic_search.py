"""
Semantic Search Service - LLM-first query understanding + embedding retrieval.

This is the proper way to handle natural language queries:
1. LLM interprets the query and identifies seed papers/concepts
2. Use Semantic Scholar's embedding-based recommendations
3. Cross-encoder re-ranking for precision

Best Practices from Elicit.ai / Consensus.app:
- No regex pre-processing - LLM handles natural language
- Embedding retrieval, not keyword matching
- Seed paper expansion for finding related work
"""

from __future__ import annotations

import logging
from typing import Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryUnderstanding:
    """LLM's interpretation of the user query."""
    intent: str  # What the user wants
    topic: str  # Core research topic
    search_terms: list[str]  # Specific terms to search
    seed_papers: list[str]  # Known papers/DOIs to use as seeds
    seed_authors: list[str]  # Key researchers
    exclude_domains: list[str]  # Domains to exclude (e.g., chemistry)
    explanation: str


def understand_query(query: str, llm: Any) -> QueryUnderstanding | None:
    """
    Use LLM to understand the user's query - NO regex, pure LLM interpretation.
    
    This is the proper way to parse natural language queries.
    The LLM identifies:
    - The core research topic
    - Known papers in this area (for seeding recommendations)
    - Key authors
    - What domains to exclude
    
    Args:
        query: Raw user query (natural language)
        llm: LLM instance
        
    Returns:
        QueryUnderstanding with structured interpretation
    """
    prompt = f'''You are a scientific literature search expert. Interpret this query.

USER QUERY: "{query}"

Analyze this query and return a JSON object with:
1. "intent": What the user wants (e.g., "find papers about Liquid Neural Networks")
2. "topic": The core research topic in formal terminology
3. "search_terms": 3-5 specific technical terms for searching databases
4. "seed_papers": Known seminal papers in this area (title or DOI if you know it)
5. "seed_authors": Key researchers who publish in this area
6. "exclude_domains": Fields to exclude (e.g., if they mean ML "liquid", exclude chemistry)
7. "explanation": Brief explanation of your interpretation

EXAMPLES of proper interpretation:
- "liquid foundation models" → topic: "Liquid Neural Networks / Liquid Time-constant Networks"
  (NOT chemistry liquids, but the ML architecture by Hasani et al.)
- "transformer papers" → topic: "Transformer neural networks / Attention mechanism"
  (NOT electrical transformers)

Return ONLY valid JSON:
{{
    "intent": "find papers about X",
    "topic": "formal topic name",
    "search_terms": ["term1", "term2", "term3"],
    "seed_papers": ["Paper Title 1", "10.xxxx/doi"],
    "seed_authors": ["Author Name"],
    "exclude_domains": ["chemistry", "biology"],
    "explanation": "brief explanation"
}}
'''
    
    try:
        response = llm.generate(
            prompt=prompt,
            system_prompt="You are a scientific search expert. Return only valid JSON.",
            max_tokens=800
        )
        
        # Parse JSON using shared utility
        from litforge.services.utils import parse_llm_json
        result = parse_llm_json(response)
        
        if not result:
            return None
        
        return QueryUnderstanding(
            intent=result.get("intent", query),
            topic=result.get("topic", query),
            search_terms=result.get("search_terms", []),
            seed_papers=result.get("seed_papers", []),
            seed_authors=result.get("seed_authors", []),
            exclude_domains=result.get("exclude_domains", []),
            explanation=result.get("explanation", ""),
        )
        
    except Exception as e:
        logger.warning(f"Query understanding failed: {e}")
        return None


def find_seed_paper_ids(
    understanding: QueryUnderstanding,
    limit: int = 5,
) -> list[str]:
    """
    Find Semantic Scholar paper IDs for seed papers.
    
    Uses the LLM's identified seed papers and authors to find
    concrete paper IDs for the recommendations API.
    
    Args:
        understanding: LLM's query interpretation
        limit: Max seed papers to find
        
    Returns:
        List of Semantic Scholar paper IDs
    """
    from litforge.clients.semantic_scholar import SemanticScholarClient
    
    client = SemanticScholarClient()
    paper_ids = []
    
    # Search for seed papers by title
    for seed_title in understanding.seed_papers[:3]:
        try:
            # Clean the seed - it might be a title or DOI
            if seed_title.startswith("10."):
                # It's a DOI
                paper = client.get_paper(doi=seed_title)
                if paper and paper.semantic_scholar_id:
                    paper_ids.append(paper.semantic_scholar_id)
            else:
                # It's a title - search for it
                results = client.search(seed_title, limit=1)
                if results and results[0].semantic_scholar_id:
                    paper_ids.append(results[0].semantic_scholar_id)
                    logger.info(f"Found seed paper: {results[0].title}")
        except Exception as e:
            logger.debug(f"Seed paper lookup failed for '{seed_title}': {e}")
    
    # If no seeds found, search by topic + author
    if not paper_ids and understanding.seed_authors:
        for author in understanding.seed_authors[:2]:
            try:
                search_query = f"{understanding.topic} {author}"
                results = client.search(search_query, limit=2)
                for r in results:
                    if r.semantic_scholar_id and r.semantic_scholar_id not in paper_ids:
                        paper_ids.append(r.semantic_scholar_id)
                        logger.info(f"Found seed via author: {r.title}")
            except Exception as e:
                logger.debug(f"Author seed search failed: {e}")
    
    # Fallback: just search the topic
    if not paper_ids:
        try:
            results = client.search(understanding.topic, limit=3)
            for r in results:
                if r.semantic_scholar_id:
                    paper_ids.append(r.semantic_scholar_id)
        except Exception as e:
            logger.debug(f"Topic search failed: {e}")
    
    return paper_ids[:limit]


def semantic_search(
    query: str,
    llm: Any,
    max_results: int = 25,
    use_recommendations: bool = True,
    use_keyword_fallback: bool = True,
) -> tuple[list[Any], dict]:
    """
    Semantic search pipeline using LLM query understanding + embedding retrieval.
    
    This implements the best practices from Elicit.ai / Consensus.app:
    1. LLM interprets the query (no regex)
    2. Find seed papers in the field
    3. Use Semantic Scholar recommendations (embedding-based)
    4. Merge with keyword search for coverage
    5. Re-rank with ensemble scoring
    
    Args:
        query: Raw natural language query
        llm: LLM instance
        max_results: Maximum papers to return
        use_recommendations: Use embedding-based recommendations
        use_keyword_fallback: Also do keyword search for coverage
        
    Returns:
        (papers, metadata) tuple
    """
    from litforge.clients.semantic_scholar import SemanticScholarClient
    from litforge.clients.openalex import OpenAlexClient
    
    metadata = {
        "method": "semantic_search",
        "query": query,
    }
    
    all_papers = []
    
    # Step 1: LLM Query Understanding (NO REGEX)
    logger.info(f"Understanding query: '{query}'")
    understanding = understand_query(query, llm)
    
    if understanding:
        metadata["understanding"] = {
            "intent": understanding.intent,
            "topic": understanding.topic,
            "search_terms": understanding.search_terms,
            "seed_papers": understanding.seed_papers,
            "seed_authors": understanding.seed_authors,
            "exclude_domains": understanding.exclude_domains,
            "explanation": understanding.explanation,
        }
        logger.info(f"Query understood: {understanding.topic}")
    else:
        # Fallback to basic search
        logger.warning("LLM understanding failed, using basic search")
        metadata["understanding"] = {"fallback": True}
        understanding = QueryUnderstanding(
            intent=query,
            topic=query,
            search_terms=[query],
            seed_papers=[],
            seed_authors=[],
            exclude_domains=[],
            explanation="Fallback mode",
        )
    
    # Step 2: Find seed paper IDs for recommendations
    ss_client = SemanticScholarClient()
    seed_ids = []
    
    if use_recommendations:
        seed_ids = find_seed_paper_ids(understanding, limit=5)
        metadata["seed_papers_found"] = len(seed_ids)
        logger.info(f"Found {len(seed_ids)} seed papers for recommendations")
    
    # Step 3: Get recommendations (embedding-based similarity)
    if seed_ids and use_recommendations:
        try:
            recs = ss_client.get_recommendations(seed_ids, limit=max_results)
            all_papers.extend(recs)
            metadata["recommendations"] = len(recs)
            logger.info(f"Got {len(recs)} recommendations from embeddings")
        except Exception as e:
            logger.warning(f"Recommendations failed: {e}")
            metadata["recommendations_error"] = str(e)
    
    # Step 4: Keyword search for coverage (multiple terms)
    if use_keyword_fallback:
        oa_client = OpenAlexClient()
        
        for term in understanding.search_terms[:3]:
            try:
                results = ss_client.search(term, limit=10)
                all_papers.extend(results)
                logger.info(f"Keyword search '{term}': {len(results)} results")
            except Exception as e:
                logger.debug(f"S2 keyword search failed: {e}")
            
            try:
                results = oa_client.search(term, limit=10)
                all_papers.extend(results)
            except Exception as e:
                logger.debug(f"OpenAlex search failed: {e}")
    
    metadata["raw_results"] = len(all_papers)
    
    # Step 5: Deduplicate
    from litforge.services.utils import deduplicate_papers
    papers = deduplicate_papers(all_papers)
    metadata["after_dedup"] = len(papers)
    
    # Step 6: Filter by exclude_domains with smarter ML detection
    if understanding.exclude_domains:
        exclude_lower = [d.lower() for d in understanding.exclude_domains]
        
        # ML-specific terms that indicate the paper is actually about ML
        ml_indicators = [
            "liquid neural network", "liquid time-constant", "ltc network",
            "recurrent neural network", "machine learning", "deep learning",
            "neural architecture", "transformer", "attention mechanism",
        ]
        
        # Chemistry/physics terms that indicate wrong domain
        wrong_domain_indicators = [
            "ionic liquid", "liquid water", "molecular dynamics", "polarizability",
            "raman spectrum", "density functional", "chemical", "solvent",
            "aqueous", "viscosity", "thermodynamic", "electrochemical",
        ]
        
        filtered = []
        for p in papers:
            text = f"{p.title or ''} {p.abstract or ''}".lower()
            
            # Check if it's actually about ML liquid networks
            is_ml_paper = any(ind in text for ind in ml_indicators)
            is_wrong_domain = any(ind in text for ind in wrong_domain_indicators)
            
            # Keep if it's ML, exclude if it's wrong domain
            if is_ml_paper and not is_wrong_domain:
                filtered.append(p)
            elif not is_wrong_domain:
                # Unknown domain - check for neural/network mentions
                if "neural" in text or "network" in text:
                    filtered.append(p)
        
        metadata["filtered_by_domain"] = len(papers) - len(filtered)
        papers = filtered
    
    # Step 7: Score and rank
    try:
        from litforge.services.scoring import score_and_verify
        
        scored_papers, scoring_meta = score_and_verify(
            papers=papers,
            query=understanding.topic,  # Use interpreted topic
            must_contain=[],
            exclude_terms=understanding.exclude_domains,
            target_authors=understanding.seed_authors,
            ai_insight=understanding.explanation,
            llm=llm,
            verify_threshold=0.25,
            top_k=10,
            use_embeddings=True,
        )
        
        metadata["scoring"] = scoring_meta
        papers = [sp.paper for sp in scored_papers[:max_results]]
        
    except ImportError as e:
        logger.warning(f"Scoring module not available: {e}")
    
    metadata["final_results"] = len(papers)
    
    return papers, metadata
