"""
Shared utilities for LitForge services.
"""

import logging
import json
from typing import Any

logger = logging.getLogger(__name__)


def parse_llm_json(response: str) -> dict | None:
    """
    Parse JSON from LLM response, handling common formats.
    
    Handles:
    - Raw JSON
    - JSON in markdown code blocks (```json ... ```)
    - JSON with text before/after
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Parsed dict or None if parsing fails
    """
    if not response:
        return None
    
    response = response.strip()
    
    # Case 1: JSON wrapped in code blocks
    if "```" in response:
        parts = response.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                response = part
                break
    
    # Case 2: Text before JSON - find the first {
    if not response.startswith("{"):
        brace_idx = response.find("{")
        if brace_idx != -1:
            response = response[brace_idx:]
    
    # Case 3: Text after JSON - find the last }
    if response.startswith("{"):
        last_brace = response.rfind("}")
        if last_brace != -1:
            response = response[:last_brace + 1]
    
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        return None


def deduplicate_papers(
    papers: list[Any],
    similarity_threshold: float = 0.8,
) -> list[Any]:
    """
    Deduplicate papers by DOI first, then by title similarity.
    
    Works with both Publication objects and generic paper-like objects.
    
    Args:
        papers: List of papers (may have duplicates)
        similarity_threshold: Title similarity threshold for fuzzy matching
        
    Returns:
        Deduplicated list of papers
    """
    if not papers:
        return []
    
    unique = []
    seen_dois: set[str] = set()
    seen_titles: list[str] = []
    
    for paper in papers:
        # Check DOI first (exact match)
        doi = getattr(paper, 'doi', None)
        if doi:
            doi_clean = doi.lower().strip()
            if doi_clean in seen_dois:
                continue
            seen_dois.add(doi_clean)
        
        # Check title similarity
        title = getattr(paper, 'title', None)
        if title:
            title_clean = title.lower().strip()[:60]
            
            # Simple similarity: check if any existing title is very similar
            is_duplicate = False
            for seen_title in seen_titles:
                # Basic overlap check
                words1 = set(title_clean.split())
                words2 = set(seen_title.split())
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1 & words2) / max(len(words1), len(words2))
                    if overlap >= similarity_threshold:
                        is_duplicate = True
                        break
            
            if is_duplicate:
                continue
            
            seen_titles.append(title_clean)
        
        unique.append(paper)
    
    logger.debug(f"Deduplication: {len(papers)} → {len(unique)} papers")
    return unique
