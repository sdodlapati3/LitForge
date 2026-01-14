"""
Citation network analysis example.

This example demonstrates how to build and analyze citation networks
to discover influential papers and research clusters.
"""

from litforge import Forge


def main():
    forge = Forge()
    
    # Start with a seed paper (CRISPR discovery paper)
    print("📄 Looking up seed paper...")
    seed_paper = forge.lookup(doi="10.1126/science.1225829")
    
    if not seed_paper:
        print("Could not find seed paper. Using search instead...")
        papers = forge.search("CRISPR Cas9 Doudna", limit=1)
        if papers:
            seed_paper = papers[0]
        else:
            print("No papers found!")
            return
    
    print(f"Seed: {seed_paper.title}")
    print(f"Citations: {seed_paper.citation_count}")
    
    # Build citation network
    print("\n🕸️ Building citation network (depth=2)...")
    network = forge.build_network(
        seeds=[seed_paper],
        depth=2,
        max_papers=100,
    )
    
    print(f"\nNetwork statistics:")
    print(f"  Nodes (papers): {network.node_count}")
    print(f"  Edges (citations): {network.edge_count}")
    
    # Find most influential papers
    print("\n📊 Most cited papers in network:")
    for i, paper in enumerate(network.most_cited(n=5), 1):
        print(f"  {i}. {paper.title[:60]}... ({paper.citation_count} citations)")
    
    # Find clusters
    print("\n🔬 Research clusters:")
    clusters = network.find_clusters()
    for cluster in clusters[:3]:
        print(f"  • {cluster.label}: {len(cluster.papers)} papers")
    
    # Export for visualization
    output_file = "citation_network.graphml"
    network.export(output_file)
    print(f"\n💾 Network exported to {output_file}")
    print("   (Open with Gephi or Cytoscape for visualization)")


if __name__ == "__main__":
    main()
