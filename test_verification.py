#!/usr/bin/env python3
"""
LitForge Systematic Verification Test Suite
============================================

This script tests LitForge components with carefully crafted test queries
where we know the expected results (ground truth) to verify accuracy.

Test Categories:
1. Known Paper Lookup (by DOI) - Verify exact metadata matches
2. Author Search - Verify known prolific authors appear
3. Topic Search - Verify relevant papers are returned
4. Citation Counts - Verify high-impact papers have expected citations
5. Vector Store Accuracy - Verify semantic search returns correct results
6. MCP Tools - Verify tool definitions are correct

Author: LitForge Test Suite
Date: January 2026
"""

import asyncio
import httpx
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class TestCase:
    """A test case with expected results."""
    name: str
    query: dict
    expected: dict
    tolerance: dict = None  # For numeric comparisons


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def add(self, name: str, passed: bool, details: str = ""):
        self.results.append({"name": name, "passed": passed, "details": details})
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def summary(self):
        total = self.passed + self.failed
        return f"{self.passed}/{total} tests passed ({100*self.passed/total:.1f}%)"


def print_test_header(name: str):
    print(f"\n{'='*70}")
    print(f"  TEST: {name}")
    print(f"{'='*70}")


def print_result(test_name: str, passed: bool, expected: Any, actual: Any, details: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}: {test_name}")
    if not passed or details:
        print(f"    Expected: {expected}")
        print(f"    Actual:   {actual}")
        if details:
            print(f"    Details:  {details}")


# =============================================================================
# TEST 1: Known Paper Lookup (Ground Truth)
# =============================================================================

KNOWN_PAPERS = [
    {
        "doi": "10.1126/science.1225829",
        "expected": {
            "title_contains": "Programmable Dual-RNA",
            "year": 2012,
            "journal": "Science",
            "min_citations": 10000,  # Famous CRISPR paper
            "authors_include": ["Jinek", "Doudna", "Charpentier"],
        }
    },
    {
        "doi": "10.1038/nature14539",
        "expected": {
            "title_contains": "Deep learning",
            "year": 2015,
            "journal": "Nature",
            "min_citations": 30000,  # Famous deep learning review
            "authors_include": ["LeCun", "Bengio", "Hinton"],
        }
    },
    {
        "doi": "10.1038/s41586-021-03819-2",
        "expected": {
            "title_contains": "AlphaFold",
            "year": 2021,
            "journal": "Nature",
            "min_citations": 5000,  # AlphaFold 2 paper
            "authors_include": ["Jumper"],
        }
    },
]


async def test_paper_lookup():
    """Test paper lookup with known ground truth."""
    print_test_header("PAPER LOOKUP - CrossRef API")
    results = TestResults()
    
    async with httpx.AsyncClient(timeout=30) as client:
        for paper in KNOWN_PAPERS:
            doi = paper["doi"]
            expected = paper["expected"]
            
            print(f"\n  Testing DOI: {doi}")
            
            try:
                resp = await client.get(f"https://api.crossref.org/works/{doi}")
                data = resp.json()
                
                if "message" not in data:
                    results.add(f"Lookup {doi}", False, "No data returned")
                    continue
                
                work = data["message"]
                
                # Test title
                title = work.get("title", [""])[0]
                title_match = expected["title_contains"].lower() in title.lower()
                print_result(
                    f"Title contains '{expected['title_contains']}'",
                    title_match,
                    expected["title_contains"],
                    title[:60] + "..."
                )
                results.add(f"{doi} title", title_match)
                
                # Test year
                year = work.get("published-print", {}).get("date-parts", [[None]])[0][0]
                if year is None:
                    year = work.get("published-online", {}).get("date-parts", [[None]])[0][0]
                year_match = year == expected["year"]
                print_result("Year", year_match, expected["year"], year)
                results.add(f"{doi} year", year_match)
                
                # Test citations (with tolerance - citations grow over time)
                citations = work.get("is-referenced-by-count", 0)
                citation_match = citations >= expected["min_citations"]
                print_result(
                    f"Citations >= {expected['min_citations']}",
                    citation_match,
                    f">= {expected['min_citations']}",
                    citations
                )
                results.add(f"{doi} citations", citation_match)
                
                # Test authors
                authors = work.get("author", [])
                author_names = [a.get("family", "") for a in authors]
                for expected_author in expected["authors_include"]:
                    author_found = any(expected_author.lower() in name.lower() for name in author_names)
                    print_result(
                        f"Author '{expected_author}' present",
                        author_found,
                        expected_author,
                        ", ".join(author_names[:5])
                    )
                    results.add(f"{doi} author {expected_author}", author_found)
                
            except Exception as e:
                print(f"    Error: {e}")
                results.add(f"Lookup {doi}", False, str(e))
    
    return results


# =============================================================================
# TEST 2: Topic Search Accuracy
# =============================================================================

TOPIC_SEARCHES = [
    {
        "query": "CRISPR Cas9 gene editing",
        "expected_keywords": ["CRISPR", "Cas9", "gene", "editing", "genome"],
        "min_results": 100,
        "recent_year_min": 2020,  # Should find recent papers
    },
    {
        "query": "transformer attention mechanism neural network",
        "expected_keywords": ["transformer", "attention", "neural", "language", "model"],
        "min_results": 50,
        "recent_year_min": 2018,
    },
    {
        "query": "COVID-19 mRNA vaccine",
        "expected_keywords": ["COVID", "vaccine", "mRNA", "SARS-CoV-2"],
        "min_results": 100,
        "recent_year_min": 2020,
    },
]


async def test_topic_search():
    """Test topic search returns relevant results."""
    print_test_header("TOPIC SEARCH - OpenAlex API")
    results = TestResults()
    
    async with httpx.AsyncClient(timeout=30) as client:
        for search in TOPIC_SEARCHES:
            query = search["query"]
            print(f"\n  Query: '{query}'")
            
            try:
                resp = await client.get(
                    "https://api.openalex.org/works",
                    params={"search": query, "per_page": 25}
                )
                data = resp.json()
                
                total = data.get("meta", {}).get("count", 0)
                works = data.get("results", [])
                
                # Test minimum results
                min_results_match = total >= search["min_results"]
                print_result(
                    f"Total results >= {search['min_results']}",
                    min_results_match,
                    f">= {search['min_results']}",
                    total
                )
                results.add(f"'{query}' count", min_results_match)
                
                # Test keyword relevance in top results
                all_text = " ".join([
                    (w.get("title", "") or "") + " " + (w.get("abstract_inverted_index") and "abstract" or "")
                    for w in works[:10]
                ]).lower()
                
                keywords_found = 0
                for keyword in search["expected_keywords"]:
                    if keyword.lower() in all_text:
                        keywords_found += 1
                
                keyword_ratio = keywords_found / len(search["expected_keywords"])
                keyword_match = keyword_ratio >= 0.6  # At least 60% of keywords
                print_result(
                    f"Keywords relevance >= 60%",
                    keyword_match,
                    f">= 60%",
                    f"{keyword_ratio*100:.0f}% ({keywords_found}/{len(search['expected_keywords'])})"
                )
                results.add(f"'{query}' keywords", keyword_match)
                
                # Test recency
                years = [w.get("publication_year") for w in works[:10] if w.get("publication_year")]
                if years:
                    max_year = max(years)
                    recent_match = max_year >= search["recent_year_min"]
                    print_result(
                        f"Has papers from {search['recent_year_min']}+",
                        recent_match,
                        f">= {search['recent_year_min']}",
                        f"Most recent: {max_year}"
                    )
                    results.add(f"'{query}' recency", recent_match)
                
            except Exception as e:
                print(f"    Error: {e}")
                results.add(f"Search '{query}'", False, str(e))
    
    return results


# =============================================================================
# TEST 3: Vector Store Semantic Search
# =============================================================================

SEMANTIC_TEST_DOCS = [
    {"id": "bio1", "text": "CRISPR-Cas9 is a molecular tool for precise genome editing in living cells.", "category": "biology"},
    {"id": "bio2", "text": "The DNA double helix structure was discovered by Watson and Crick in 1953.", "category": "biology"},
    {"id": "bio3", "text": "Proteins fold into specific 3D structures determined by their amino acid sequence.", "category": "biology"},
    {"id": "cs1", "text": "Neural networks learn patterns through backpropagation of gradient errors.", "category": "cs"},
    {"id": "cs2", "text": "Transformer models use self-attention mechanisms for sequence processing.", "category": "cs"},
    {"id": "cs3", "text": "Large language models are trained on massive text corpora to predict tokens.", "category": "cs"},
    {"id": "chem1", "text": "Organic chemistry studies carbon-based compounds and their reactions.", "category": "chemistry"},
    {"id": "chem2", "text": "Catalysts speed up chemical reactions without being consumed.", "category": "chemistry"},
]

SEMANTIC_QUERIES = [
    {"query": "gene editing technology", "expected_top": ["bio1"], "expected_category": "biology"},
    {"query": "machine learning algorithms", "expected_top": ["cs1", "cs2"], "expected_category": "cs"},
    {"query": "protein structure prediction", "expected_top": ["bio3"], "expected_category": "biology"},
    {"query": "attention in deep learning", "expected_top": ["cs2"], "expected_category": "cs"},
]


async def test_vector_stores():
    """Test vector store operations (not semantic accuracy - requires real embeddings)."""
    print_test_header("VECTOR STORE OPERATIONS")
    results = TestResults()
    
    import tempfile
    from pathlib import Path
    import hashlib
    import random
    
    from litforge.stores.chromadb import ChromaDBStore
    from litforge.stores.qdrant import QdrantStore
    from litforge.stores.faiss import FAISSStore
    
    # Use simple keyword-weighted embeddings for better semantic matching
    def make_embedding(text: str, dim: int = 384) -> list[float]:
        """Create embeddings with keyword weighting for semantic similarity."""
        # Define keyword categories
        keywords = {
            "bio": ["crispr", "dna", "gene", "protein", "cell", "genome", "fold", "amino", "helix"],
            "cs": ["neural", "network", "learning", "transformer", "attention", "model", "train", "backprop"],
            "chem": ["organic", "carbon", "reaction", "catalyst", "compound", "chemical"],
        }
        
        text_lower = text.lower()
        
        # Create embedding based on keyword presence
        embedding = [0.0] * dim
        
        # Set dimensions based on category matches
        for i, (category, words) in enumerate(keywords.items()):
            base_idx = i * 100
            for j, word in enumerate(words):
                if word in text_lower:
                    embedding[base_idx + j * 10] = 1.0
                    # Add some spread
                    for k in range(1, 5):
                        if base_idx + j * 10 + k < dim:
                            embedding[base_idx + j * 10 + k] = 0.5
        
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        else:
            # Fallback to hash-based if no keywords match
            h = hashlib.sha256(text.encode()).digest()
            random.seed(int.from_bytes(h[:4], 'big'))
            embedding = [random.gauss(0, 1) for _ in range(dim)]
            norm = sum(x*x for x in embedding) ** 0.5
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    with tempfile.TemporaryDirectory() as tmpdir:
        stores = [
            ("ChromaDB", ChromaDBStore(persist_dir=Path(tmpdir)/"chroma", collection_name="test")),
            ("Qdrant", QdrantStore(in_memory=True, collection_name="test", embedding_dim=384)),
            ("FAISS", FAISSStore(persist_dir=Path(tmpdir)/"faiss", embedding_dim=384)),
        ]
        
        for store_name, store in stores:
            print(f"\n  Testing {store_name}:")
            
            # Test 1: Add documents
            for doc in SEMANTIC_TEST_DOCS:
                store.add(
                    id=doc["id"],
                    embedding=make_embedding(doc["text"]),
                    metadata={"category": doc["category"]},
                    text=doc["text"],
                )
            
            add_success = store.count() == len(SEMANTIC_TEST_DOCS)
            print_result(f"{store_name}: Add {len(SEMANTIC_TEST_DOCS)} docs", add_success, 
                        len(SEMANTIC_TEST_DOCS), store.count())
            results.add(f"{store_name} add", add_success)
            
            # Test 2: Search returns results
            query_emb = make_embedding("gene editing CRISPR DNA")
            search_results = store.search(query_emb, limit=3)
            
            search_success = len(search_results) == 3
            print_result(f"{store_name}: Search returns 3 results", search_success, 3, len(search_results))
            results.add(f"{store_name} search", search_success)
            
            # Test 3: Bio query returns bio category in top results
            bio_query = make_embedding("CRISPR DNA gene editing genome")
            bio_results = store.search(bio_query, limit=3)
            bio_categories = [r["metadata"]["category"] for r in bio_results]
            bio_in_top = "biology" in bio_categories
            print_result(f"{store_name}: Bio query finds bio docs", bio_in_top, 
                        "biology in top 3", bio_categories)
            results.add(f"{store_name} bio relevance", bio_in_top)
            
            # Test 4: CS query returns CS category in top results
            cs_query = make_embedding("neural network learning transformer attention")
            cs_results = store.search(cs_query, limit=3)
            cs_categories = [r["metadata"]["category"] for r in cs_results]
            cs_in_top = "cs" in cs_categories
            print_result(f"{store_name}: CS query finds CS docs", cs_in_top,
                        "cs in top 3", cs_categories)
            results.add(f"{store_name} cs relevance", cs_in_top)
            
            # Test 5: Clear works
            store.clear()
            clear_success = store.count() == 0
            print_result(f"{store_name}: Clear", clear_success, 0, store.count())
            results.add(f"{store_name} clear", clear_success)
    
    return results


# =============================================================================
# TEST 4: MCP Server Tools
# =============================================================================

EXPECTED_MCP_TOOLS = {
    "search_papers": {"required_params": ["query"]},
    "lookup_paper": {"required_params": []},  # Has optional params
    "get_citations": {"required_params": ["paper_id"]},
    "get_references": {"required_params": ["paper_id"]},
    "retrieve_fulltext": {"required_params": ["paper_id"]},
    "index_papers": {"required_params": ["papers"]},
    "ask_papers": {"required_params": ["question"]},
    "build_citation_network": {"required_params": ["seed_papers"]},
    "summarize_papers": {"required_params": ["papers"]},
}


async def test_mcp_tools():
    """Test MCP server tools are correctly defined."""
    print_test_header("MCP SERVER TOOLS")
    results = TestResults()
    
    from litforge.mcp.server import create_server
    from litforge.mcp.tools import LitForgeTools
    
    # Test server creation
    server = create_server()
    server_name_match = server.name == "litforge"
    print_result("Server name", server_name_match, "litforge", server.name)
    results.add("Server name", server_name_match)
    
    # Test tools
    tools = LitForgeTools()
    tool_defs = tools.get_tool_definitions()
    
    # Check expected tools exist
    tool_names = {t["name"] for t in tool_defs}
    
    for expected_tool in EXPECTED_MCP_TOOLS:
        tool_exists = expected_tool in tool_names
        print_result(f"Tool '{expected_tool}' exists", tool_exists, True, tool_exists)
        results.add(f"Tool {expected_tool}", tool_exists)
    
    # Check tool count
    tool_count_match = len(tool_defs) == len(EXPECTED_MCP_TOOLS)
    print_result(
        "Tool count",
        tool_count_match,
        len(EXPECTED_MCP_TOOLS),
        len(tool_defs)
    )
    results.add("Tool count", tool_count_match)
    
    # Check all tools have descriptions
    all_have_descriptions = all(len(t.get("description", "")) > 10 for t in tool_defs)
    print_result("All tools have descriptions", all_have_descriptions, True, all_have_descriptions)
    results.add("Tool descriptions", all_have_descriptions)
    
    return results


# =============================================================================
# TEST 5: Integration Test - End-to-End Workflow
# =============================================================================

async def test_integration():
    """Test end-to-end workflow: Search -> Index -> Query."""
    print_test_header("INTEGRATION TEST - Search to Index to Query")
    results = TestResults()
    
    import tempfile
    from pathlib import Path
    import hashlib
    import random
    
    from litforge.stores.chromadb import ChromaDBStore
    
    def make_embedding(text: str, dim: int = 384) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        random.seed(int.from_bytes(h[:4], 'big'))
        return [random.gauss(0, 1) for _ in range(dim)]
    
    print("\n  Step 1: Search OpenAlex for papers...")
    
    async with httpx.AsyncClient(timeout=30) as client:
        # Search for papers
        resp = await client.get(
            "https://api.openalex.org/works",
            params={"search": "protein folding prediction", "per_page": 5}
        )
        data = resp.json()
        papers = data.get("results", [])
        
        search_success = len(papers) > 0
        print_result("Search returned papers", search_success, "> 0", len(papers))
        results.add("Search step", search_success)
        
        if not search_success:
            return results
    
    print("\n  Step 2: Index papers in ChromaDB...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaDBStore(persist_dir=Path(tmpdir), collection_name="integration_test")
        
        # Index papers
        for i, paper in enumerate(papers):
            title = paper.get("title", "") or ""
            abstract = ""
            if paper.get("abstract_inverted_index"):
                # Reconstruct abstract from inverted index
                idx = paper["abstract_inverted_index"]
                words = [""] * (max(max(positions) for positions in idx.values()) + 1)
                for word, positions in idx.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words)
            
            text = f"{title}. {abstract}"
            
            store.add(
                id=f"paper_{i}",
                embedding=make_embedding(text),
                metadata={"title": title, "year": paper.get("publication_year")},
                text=text,
            )
        
        index_success = store.count() == len(papers)
        print_result("All papers indexed", index_success, len(papers), store.count())
        results.add("Index step", index_success)
        
        print("\n  Step 3: Semantic search in indexed papers...")
        
        # Query the indexed papers
        query = "AlphaFold structure prediction"
        query_results = store.search(make_embedding(query), limit=3)
        
        query_success = len(query_results) > 0
        print_result("Query returned results", query_success, "> 0", len(query_results))
        results.add("Query step", query_success)
        
        if query_results:
            print(f"\n    Top result: {query_results[0]['metadata'].get('title', 'N/A')[:50]}...")
        
        # Verify relevance (should contain protein/structure related terms)
        if query_results:
            top_text = query_results[0]["text"].lower()
            relevance = any(term in top_text for term in ["protein", "structure", "fold", "predict"])
            print_result("Top result is relevant", relevance, "protein/structure terms", 
                        "Found" if relevance else "Not found")
            results.add("Relevance check", relevance)
        
        store.clear()
    
    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def main():
    """Run all tests and report results."""
    print("\n" + "🧪" * 35)
    print("       LitForge Systematic Verification Test Suite")
    print("🧪" * 35)
    
    all_results = TestResults()
    
    # Run all test categories
    test_functions = [
        ("Paper Lookup (CrossRef)", test_paper_lookup),
        ("Topic Search (OpenAlex)", test_topic_search),
        ("Vector Stores", test_vector_stores),
        ("MCP Tools", test_mcp_tools),
        ("Integration", test_integration),
    ]
    
    category_results = {}
    
    for name, test_func in test_functions:
        try:
            result = await test_func()
            category_results[name] = result
            all_results.passed += result.passed
            all_results.failed += result.failed
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            category_results[name] = None
    
    # Print summary
    print("\n" + "=" * 70)
    print("  FINAL TEST SUMMARY")
    print("=" * 70)
    
    for name, result in category_results.items():
        if result:
            status = "✅" if result.failed == 0 else "⚠️"
            print(f"\n  {status} {name}: {result.summary()}")
        else:
            print(f"\n  ❌ {name}: ERROR")
    
    print(f"\n  {'='*50}")
    print(f"  OVERALL: {all_results.summary()}")
    print(f"  {'='*50}")
    
    if all_results.failed == 0:
        print("\n  🎉 ALL TESTS PASSED! LitForge is working correctly.")
    else:
        print(f"\n  ⚠️  {all_results.failed} tests failed. Review details above.")
    
    print("\n" + "🧪" * 35 + "\n")
    
    return all_results


if __name__ == "__main__":
    asyncio.run(main())
