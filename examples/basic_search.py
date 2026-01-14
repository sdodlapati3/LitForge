"""
Basic paper search example.

This example demonstrates how to use LitForge to search for papers
across multiple scientific databases.
"""

from litforge import Forge


def main():
    # Initialize LitForge
    forge = Forge()
    
    # Simple search
    print("🔍 Searching for CRISPR papers...")
    papers = forge.search("CRISPR gene editing mechanisms", limit=10)
    
    print(f"\nFound {len(papers)} papers:\n")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.title}")
        print(f"   Authors: {', '.join(a.name for a in paper.authors[:3])}")
        print(f"   Year: {paper.year}")
        print(f"   DOI: {paper.doi}")
        print(f"   Citations: {paper.citation_count}")
        print()
    
    # Advanced search with filters
    print("\n🔍 Searching with filters (2020-2024, open access)...")
    papers = forge.search(
        "transformer attention mechanisms",
        limit=5,
        year_from=2020,
        year_to=2024,
        open_access=True,
    )
    
    print(f"\nFound {len(papers)} open access papers:\n")
    for paper in papers:
        print(f"• {paper.title} ({paper.year})")


if __name__ == "__main__":
    main()
