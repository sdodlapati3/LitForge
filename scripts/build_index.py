#!/usr/bin/env python3
"""
Download and build embedding index for LitForge.

Usage:
    # Download arXiv embeddings from HuggingFace (~3.78GB, ~296K papers)
    python scripts/build_index.py --source huggingface --dataset sproos/arxiv-embeddings --name arxiv
    
    # Download full S2AG embeddings (~50GB, ~200M papers)
    python scripts/build_index.py --source s2ag --name s2ag
    
    # List existing indices
    python scripts/build_index.py --list
    
    # Test an index
    python scripts/build_index.py --test arxiv --query "transformer neural network"
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from litforge.stores.embedding_index import EmbeddingIndexManager, get_embedding_index


def main():
    parser = argparse.ArgumentParser(description="Build embedding index for LitForge")
    parser.add_argument("--source", choices=["huggingface", "s2ag"], 
                       help="Source for embeddings")
    parser.add_argument("--dataset", default="sproos/arxiv-embeddings",
                       help="HuggingFace dataset name")
    parser.add_argument("--name", default="arxiv",
                       help="Name for the index")
    parser.add_argument("--list", action="store_true",
                       help="List existing indices")
    parser.add_argument("--test", type=str,
                       help="Test an index with a query")
    parser.add_argument("--query", type=str, default="machine learning",
                       help="Query for testing")
    parser.add_argument("--index-dir", type=str,
                       help="Directory for index storage")
    
    args = parser.parse_args()
    
    # Initialize manager
    index_dir = Path(args.index_dir) if args.index_dir else None
    manager = EmbeddingIndexManager(index_dir)
    
    if args.list:
        print("\n📚 Available Embedding Indices:\n")
        indices = manager.list_indices()
        if not indices:
            print("   No indices found. Build one with:")
            print("   python scripts/build_index.py --source huggingface --name arxiv")
        else:
            for idx in indices:
                print(f"   📁 {idx['name']}")
                print(f"      Size: {idx['size']:,} papers")
                if idx.get('index_size_mb'):
                    print(f"      Disk: {idx['index_size_mb']:.1f} MB")
                if idx.get('year_range'):
                    print(f"      Years: {idx['year_range'][0]} - {idx['year_range'][1]}")
                print()
        return
    
    if args.test:
        print(f"\n🔍 Testing index '{args.test}' with query: {args.query}\n")
        
        index = manager.get_index(args.test)
        if index.size == 0:
            print(f"   ❌ Index '{args.test}' is empty or doesn't exist")
            return
        
        # Get embedding for query
        try:
            from litforge.embedding.sentence_transformers import LocalEmbedder
            import numpy as np
            
            embedder = LocalEmbedder(model="allenai/specter2")
            query_embedding = embedder.embed([args.query])[0]
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
            results = index.search(query_embedding, k=10)
            
            print(f"   Found {len(results)} similar papers:\n")
            for i, (paper, score) in enumerate(results, 1):
                print(f"   {i}. [{score:.3f}] {paper.title[:80]}")
                if paper.year:
                    print(f"      Year: {paper.year}")
                print()
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print("   Make sure SPECTER2 is installed: pip install sentence-transformers adapters")
        return
    
    if args.source:
        print(f"\n🚀 Building index from {args.source}...\n")
        print(f"   Dataset: {args.dataset}")
        print(f"   Name: {args.name}")
        print()
        
        try:
            index = manager.download_index(
                source=args.source,
                dataset=args.dataset,
                name=args.name,
            )
            
            print(f"\n✅ Index built successfully!")
            print(f"   Papers: {index.size:,}")
            stats = index.get_stats()
            if stats.get('index_size_mb'):
                print(f"   Size: {stats['index_size_mb']:.1f} MB")
            print(f"   Path: {index.index_path}")
            
        except Exception as e:
            print(f"\n❌ Failed to build index: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
