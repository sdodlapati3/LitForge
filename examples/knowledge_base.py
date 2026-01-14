"""
Knowledge base and RAG Q&A example.

This example demonstrates how to build a knowledge base from papers
and use RAG (Retrieval Augmented Generation) to answer questions.
"""

import os
from litforge import Forge


def main():
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Set it for embeddings and Q&A.")
        print("   export OPENAI_API_KEY='sk-...'")
        print()
    
    forge = Forge()
    
    # Search for papers on a topic
    print("🔍 Searching for papers on protein folding...")
    papers = forge.search("AlphaFold protein structure prediction", limit=20)
    print(f"Found {len(papers)} papers")
    
    # Index papers into knowledge base
    print("\n📚 Indexing papers into knowledge base...")
    indexed_count = forge.index(papers)
    print(f"Indexed {indexed_count} papers")
    
    # Ask questions
    questions = [
        "What is AlphaFold and how does it predict protein structures?",
        "What are the main limitations of current protein folding methods?",
        "How accurate is AlphaFold compared to experimental methods?",
    ]
    
    print("\n💬 Asking questions about the literature:\n")
    for question in questions:
        print(f"Q: {question}")
        
        answer = forge.ask(question)
        
        print(f"A: {answer.text}\n")
        print("Sources:")
        for citation in answer.citations[:3]:
            print(f"  • {citation.title} (DOI: {citation.doi})")
        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
