"""
Ensemble Scoring and LLM Verification for Literature Search Results.

This module provides a multi-signal ranking system that combines:
1. Keyword matching (title + abstract)
2. Embedding similarity (semantic relevance)
3. Citation count (impact/quality proxy)
4. Recency (publication date)
5. Title vs Abstract emphasis (title matches weighted higher)

LLM verification is only triggered when uncertainty is high.

Architecture:
    Query → Search → Ensemble Scoring → [Uncertainty Check] → LLM Verify (if needed)
                          ↑                     ↓
                   (5 FREE signals)      (only when score variance high)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ScoredPaper:
    """Paper with ensemble scores."""
    paper: Any  # Publication object
    
    # Individual signal scores (0-1 normalized)
    keyword_score: float = 0.0
    embedding_score: float = 0.0
    citation_score: float = 0.0
    recency_score: float = 0.0
    title_match_score: float = 0.0
    author_match_score: float = 0.0
    
    # Ensemble results
    ensemble_score: float = 0.0
    uncertainty: float = 0.0  # Variance across signals
    
    # LLM verification (if performed)
    llm_verified: bool = False
    llm_relevance: float | None = None
    llm_reason: str | None = None
    
    # Final ranking
    final_rank: int = 0


class EnsembleScorer:
    """
    Multi-signal ensemble scorer for literature search results.
    
    Uses multiple FREE signals to rank papers, then optionally
    triggers LLM verification only for uncertain cases.
    """
    
    # Signal weights (sum to 1.0)
    WEIGHTS = {
        "keyword": 0.25,      # Keyword matching
        "embedding": 0.30,    # Semantic similarity (most important)
        "citation": 0.15,     # Impact/quality
        "recency": 0.10,      # Prefer recent papers
        "title_match": 0.15,  # Title matches weighted higher
        "author_match": 0.05, # Author name matches
    }
    
    # Uncertainty threshold for LLM verification
    UNCERTAINTY_THRESHOLD = 0.25  # Trigger LLM if variance > 25%
    
    def __init__(
        self,
        embedder: Any | None = None,
        use_embeddings: bool = True,
    ):
        """
        Initialize ensemble scorer.
        
        Args:
            embedder: Optional embedder instance (LocalEmbedder or OpenAIEmbedder)
            use_embeddings: Whether to use embedding similarity
        """
        self._embedder = embedder
        self.use_embeddings = use_embeddings
        self._query_embedding: list[float] | None = None
    
    @property
    def embedder(self):
        """Lazy-load embedder if not provided."""
        if self._embedder is None and self.use_embeddings:
            try:
                from litforge.embedding.sentence_transformers import LocalEmbedder
                self._embedder = LocalEmbedder(model="all-MiniLM-L6-v2")
                logger.info("Loaded LocalEmbedder for ensemble scoring")
            except ImportError:
                logger.warning("sentence-transformers not available, disabling embeddings")
                self.use_embeddings = False
        return self._embedder
    
    def score_papers(
        self,
        papers: list[Any],
        query: str,
        must_contain: list[str] | None = None,
        exclude_terms: list[str] | None = None,
        target_authors: list[str] | None = None,
    ) -> list[ScoredPaper]:
        """
        Score papers using ensemble of signals.
        
        Args:
            papers: List of Publication objects
            query: User's search query
            must_contain: Terms that should appear in relevant papers
            exclude_terms: Terms to exclude
            target_authors: Author names to boost
            
        Returns:
            List of ScoredPaper objects, sorted by ensemble score
        """
        if not papers:
            return []
        
        must_contain = [t.lower() for t in (must_contain or [])]
        exclude_terms = [t.lower() for t in (exclude_terms or [])]
        target_authors = [a.lower() for a in (target_authors or [])]
        
        # Pre-compute query embedding
        query_embedding = None
        if self.use_embeddings and self.embedder:
            try:
                query_embedding = self.embedder.embed([query])[0]
            except Exception as e:
                logger.warning(f"Failed to embed query: {e}")
        
        # Pre-compute paper embeddings in batch
        paper_embeddings = {}
        if query_embedding is not None and self.embedder:
            try:
                texts = [f"{p.title or ''} {p.abstract or ''}"[:500] for p in papers]
                embeddings = self.embedder.embed(texts)
                for i, p in enumerate(papers):
                    paper_embeddings[id(p)] = embeddings[i]
            except Exception as e:
                logger.warning(f"Failed to embed papers: {e}")
        
        # Compute raw scores for normalization
        raw_citations = []
        raw_years = []
        
        for p in papers:
            raw_citations.append(getattr(p, 'citation_count', 0) or 0)
            if hasattr(p, 'publication_date') and p.publication_date:
                try:
                    if isinstance(p.publication_date, str):
                        year = int(p.publication_date[:4])
                    else:
                        year = p.publication_date.year
                    raw_years.append(year)
                except (ValueError, AttributeError):
                    raw_years.append(2000)  # Default
            else:
                raw_years.append(2000)
        
        # Normalization bounds
        max_citations = max(raw_citations) if raw_citations else 1
        min_year = min(raw_years) if raw_years else 2000
        max_year = max(raw_years) if raw_years else datetime.now().year
        year_range = max(1, max_year - min_year)
        
        # Score each paper
        scored_papers = []
        
        for i, paper in enumerate(papers):
            title = (paper.title or "").lower()
            abstract = (paper.abstract or "").lower()
            text = f"{title} {abstract}"
            
            # Filter out excluded terms
            if exclude_terms and any(t in text for t in exclude_terms):
                continue
            
            # 1. Keyword Score
            keyword_score = 0.0
            if must_contain:
                matches = sum(1 for t in must_contain if t in text)
                keyword_score = matches / len(must_contain)
            else:
                # Fall back to query word matching
                query_words = query.lower().split()
                if query_words:
                    matches = sum(1 for w in query_words if w in text and len(w) > 2)
                    keyword_score = min(1.0, matches / len(query_words))
            
            # 2. Embedding Score
            embedding_score = 0.0
            if query_embedding is not None and id(paper) in paper_embeddings:
                embedding_score = self._cosine_similarity(
                    query_embedding, paper_embeddings[id(paper)]
                )
                # Normalize to 0-1 (cosine sim is -1 to 1, but usually 0 to 1 for similar texts)
                embedding_score = max(0, min(1, (embedding_score + 1) / 2))
            
            # 3. Citation Score (log-normalized)
            citation_count = raw_citations[i]
            if max_citations > 0:
                # Log scale to avoid domination by highly-cited papers
                citation_score = math.log1p(citation_count) / math.log1p(max_citations)
            else:
                citation_score = 0.0
            
            # 4. Recency Score
            year = raw_years[i]
            recency_score = (year - min_year) / year_range if year_range > 0 else 0.5
            
            # 5. Title Match Score (title matches weighted 2x)
            title_match_score = 0.0
            if must_contain:
                title_matches = sum(1 for t in must_contain if t in title)
                title_match_score = title_matches / len(must_contain)
            else:
                query_words = query.lower().split()
                if query_words:
                    title_matches = sum(1 for w in query_words if w in title and len(w) > 2)
                    title_match_score = min(1.0, title_matches / len(query_words))
            
            # 6. Author Match Score
            author_match_score = 0.0
            if target_authors:
                authors_str = ""
                if hasattr(paper, 'authors') and paper.authors:
                    if isinstance(paper.authors, list):
                        if paper.authors and isinstance(paper.authors[0], dict):
                            authors_str = " ".join(a.get('name', '') for a in paper.authors).lower()
                        else:
                            authors_str = " ".join(str(a) for a in paper.authors).lower()
                    else:
                        authors_str = str(paper.authors).lower()
                
                author_matches = sum(1 for a in target_authors if a in authors_str)
                author_match_score = author_matches / len(target_authors)
            
            # Compute ensemble score
            scores = {
                "keyword": keyword_score,
                "embedding": embedding_score,
                "citation": citation_score,
                "recency": recency_score,
                "title_match": title_match_score,
                "author_match": author_match_score,
            }
            
            ensemble_score = sum(
                scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS
            )
            
            # Compute uncertainty (variance of normalized scores)
            score_values = [keyword_score, embedding_score, citation_score, 
                          recency_score, title_match_score]
            if score_values:
                mean_score = sum(score_values) / len(score_values)
                variance = sum((s - mean_score) ** 2 for s in score_values) / len(score_values)
                uncertainty = math.sqrt(variance)  # Standard deviation
            else:
                uncertainty = 0.0
            
            scored_paper = ScoredPaper(
                paper=paper,
                keyword_score=keyword_score,
                embedding_score=embedding_score,
                citation_score=citation_score,
                recency_score=recency_score,
                title_match_score=title_match_score,
                author_match_score=author_match_score,
                ensemble_score=ensemble_score,
                uncertainty=uncertainty,
            )
            scored_papers.append(scored_paper)
        
        # Sort by ensemble score descending
        scored_papers.sort(key=lambda x: -x.ensemble_score)
        
        # Assign ranks
        for i, sp in enumerate(scored_papers):
            sp.final_rank = i + 1
        
        return scored_papers
    
    def needs_llm_verification(
        self,
        scored_papers: list[ScoredPaper],
        top_k: int = 10,
    ) -> bool:
        """
        Determine if LLM verification is needed based on uncertainty.
        
        Triggers LLM when:
        1. High average uncertainty in top_k papers
        2. Score clustering (many papers with similar scores)
        3. Low confidence in top result
        
        Args:
            scored_papers: List of scored papers
            top_k: Number of top papers to consider
            
        Returns:
            True if LLM verification recommended
        """
        if not scored_papers:
            return False
        
        top_papers = scored_papers[:top_k]
        
        # Check 1: Average uncertainty
        avg_uncertainty = sum(p.uncertainty for p in top_papers) / len(top_papers)
        if avg_uncertainty > self.UNCERTAINTY_THRESHOLD:
            logger.info(f"LLM verification needed: high uncertainty ({avg_uncertainty:.2f})")
            return True
        
        # Check 2: Score clustering (small gap between top scores)
        if len(top_papers) >= 2:
            score_gap = top_papers[0].ensemble_score - top_papers[-1].ensemble_score
            if score_gap < 0.1:  # Very similar scores
                logger.info(f"LLM verification needed: clustered scores (gap={score_gap:.2f})")
                return True
        
        # Check 3: Low absolute confidence in top result
        if top_papers[0].ensemble_score < 0.4:
            logger.info(f"LLM verification needed: low top score ({top_papers[0].ensemble_score:.2f})")
            return True
        
        return False
    
    def get_uncertainty_details(
        self,
        scored_papers: list[ScoredPaper],
        top_k: int = 10,
    ) -> dict:
        """Get detailed uncertainty metrics for debugging."""
        if not scored_papers:
            return {"needs_verification": False, "reason": "No papers"}
        
        top_papers = scored_papers[:top_k]
        avg_uncertainty = sum(p.uncertainty for p in top_papers) / len(top_papers)
        
        score_gap = 0.0
        if len(top_papers) >= 2:
            score_gap = top_papers[0].ensemble_score - top_papers[-1].ensemble_score
        
        return {
            "needs_verification": self.needs_llm_verification(scored_papers, top_k),
            "avg_uncertainty": avg_uncertainty,
            "uncertainty_threshold": self.UNCERTAINTY_THRESHOLD,
            "top_score": top_papers[0].ensemble_score if top_papers else 0,
            "score_gap": score_gap,
            "top_paper_scores": {
                p.paper.title[:50]: {
                    "ensemble": round(p.ensemble_score, 3),
                    "keyword": round(p.keyword_score, 3),
                    "embedding": round(p.embedding_score, 3),
                    "citation": round(p.citation_score, 3),
                    "uncertainty": round(p.uncertainty, 3),
                }
                for p in top_papers[:5]
            }
        }
    
    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)


def verify_with_llm(
    scored_papers: list[ScoredPaper],
    user_query: str,
    ai_insight: str,
    top_k: int = 10,
    llm: Any | None = None,
) -> list[ScoredPaper]:
    """
    Use LLM to verify/re-rank top_k papers in a SINGLE batch call.
    
    This is only called when ensemble scoring has high uncertainty.
    
    Args:
        scored_papers: Papers already scored by ensemble
        user_query: Original user query
        ai_insight: LLM's interpretation of what user wants
        top_k: Number of papers to verify
        llm: LLM instance (from router)
        
    Returns:
        Re-ranked list of scored papers with LLM verification
    """
    import json
    
    if not scored_papers or not llm:
        return scored_papers
    
    top_papers = scored_papers[:top_k]
    remaining = scored_papers[top_k:]
    
    # Build batch prompt with more context
    paper_list = []
    for i, sp in enumerate(top_papers):
        title = sp.paper.title or "Unknown"
        # Include abstract snippet for better judgment
        abstract = (sp.paper.abstract or "")[:150]
        paper_list.append(f"{i+1}. \"{title}\" - {abstract}...")
    
    prompt = f"""You are a scientific literature relevance judge. Evaluate which papers are relevant to the user's topic.

USER'S TOPIC: {ai_insight}

PAPERS TO EVALUATE:
{chr(10).join(paper_list)}

TASK: Identify papers that are RELEVANT to the user's topic.

KEEP papers that:
- Directly address the topic or closely related concepts
- Are from the correct research field (e.g., ML/AI for machine learning queries)
- Use similar terminology or methodology

EXCLUDE papers that:
- Are clearly from unrelated fields (e.g., biology papers for an ML query)
- Only share superficial keyword overlap

Return a JSON object with indices (1-based) of papers to KEEP:
{{"keep": [1, 2, 3, 5]}}

Be reasonably inclusive - when in doubt, KEEP the paper.
"""
    
    try:
        response = llm.generate(
            prompt=prompt,
            system_prompt="You are a scientific literature expert. Be STRICT about relevance. Return only valid JSON.",
            max_tokens=1000
        )
        
        logger.info(f"LLM verification raw response: {response[:500]}...")
        
        # Parse JSON (handle markdown blocks)
        response = response.strip()
        if "```" in response:
            parts = response.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    response = part
                    break
        
        if not response.startswith("{"):
            brace_idx = response.find("{")
            if brace_idx != -1:
                response = response[brace_idx:]
        
        if response.startswith("{"):
            last_brace = response.rfind("}")
            if last_brace != -1:
                response = response[:last_brace + 1]
        
        result = json.loads(response)
        logger.info(f"LLM verification parsed result: {result}")
        
        # New simplified format: {"keep": [1, 2, 3]}
        keep_indices = result.get("keep", [])
        
        if keep_indices:
            # Only keep papers that LLM approved
            filtered = []
            for idx in keep_indices:
                real_idx = idx - 1  # Convert to 0-based
                if 0 <= real_idx < len(top_papers):
                    sp = top_papers[real_idx]
                    sp.llm_verified = True
                    sp.llm_relevance = 1.0
                    filtered.append(sp)
            
            # Mark papers NOT in keep list as irrelevant
            kept_set = set(idx - 1 for idx in keep_indices)
            for i, sp in enumerate(top_papers):
                if i not in kept_set:
                    sp.llm_verified = True
                    sp.llm_relevance = 0.0
                    sp.llm_reason = "Filtered by LLM verification"
            
            # Update ranks
            for i, sp in enumerate(filtered):
                sp.final_rank = i + 1
            
            logger.info(f"LLM verification kept {len(filtered)}/{len(top_papers)} papers")
            return filtered + remaining
        
        # Fallback: old format with "verified" list
        verified_list = result.get("verified", [])
        if verified_list:
            for v in verified_list:
                idx = v.get("index", 0) - 1
                if 0 <= idx < len(top_papers):
                    top_papers[idx].llm_verified = True
                    top_papers[idx].llm_relevance = v.get("confidence", 0.5) if v.get("relevant") else 0.0
                    top_papers[idx].llm_reason = v.get("reason", "")
            
            # Filter to only relevant papers
            filtered = [sp for sp in top_papers if sp.llm_relevance is None or sp.llm_relevance >= 0.5]
            for i, sp in enumerate(filtered):
                sp.final_rank = i + 1
            return filtered + remaining
        
    except Exception as e:
        logger.warning(f"LLM verification failed: {e}")
    
    return scored_papers


# Convenience function for easy integration
def score_and_verify(
    papers: list[Any],
    query: str,
    must_contain: list[str] | None = None,
    exclude_terms: list[str] | None = None,
    target_authors: list[str] | None = None,
    ai_insight: str | None = None,
    llm: Any | None = None,
    verify_threshold: float = 0.25,
    top_k: int = 10,
    use_embeddings: bool = True,
) -> tuple[list[ScoredPaper], dict]:
    """
    Complete pipeline: ensemble score, check uncertainty, optionally verify with LLM.
    
    Args:
        papers: Raw papers from search
        query: User's query
        must_contain: Terms that must appear
        exclude_terms: Terms to exclude
        target_authors: Authors to boost
        ai_insight: LLM's interpretation of user intent
        llm: LLM instance for verification (optional)
        verify_threshold: Uncertainty threshold for LLM verification
        top_k: Number of papers to verify
        use_embeddings: Whether to use embedding similarity
        
    Returns:
        (scored_papers, metadata) where metadata includes uncertainty info
    """
    scorer = EnsembleScorer(use_embeddings=use_embeddings)
    scorer.UNCERTAINTY_THRESHOLD = verify_threshold
    
    # Step 1: Ensemble scoring
    scored_papers = scorer.score_papers(
        papers=papers,
        query=query,
        must_contain=must_contain,
        exclude_terms=exclude_terms,
        target_authors=target_authors,
    )
    
    # Step 2: Check if LLM verification needed
    uncertainty_info = scorer.get_uncertainty_details(scored_papers, top_k)
    
    metadata = {
        "total_papers": len(papers),
        "scored_papers": len(scored_papers),
        "uncertainty": uncertainty_info,
        "llm_verified": False,
    }
    
    # Step 3: Optionally verify with LLM
    if llm and ai_insight and uncertainty_info["needs_verification"]:
        scored_papers = verify_with_llm(
            scored_papers=scored_papers,
            user_query=query,
            ai_insight=ai_insight,
            top_k=top_k,
            llm=llm,
        )
        metadata["llm_verified"] = True
    
    return scored_papers, metadata
