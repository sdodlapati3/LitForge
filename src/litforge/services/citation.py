"""
Citation Service - Citation network analysis.

Builds and analyzes citation networks, finding clusters and key papers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Sequence

from litforge.config import LitForgeConfig, get_config
from litforge.models import (
    Citation,
    CitationNetwork,
    Cluster,
    NetworkEdge,
    NetworkNode,
    Publication,
)

logger = logging.getLogger(__name__)


class CitationService:
    """
    Service for building and analyzing citation networks.
    
    Provides functionality for:
    - Building citation networks from seed papers
    - Finding clusters of related papers
    - Identifying key/influential papers
    - Exporting networks for visualization
    """
    
    def __init__(self, config: LitForgeConfig | None = None):
        """
        Initialize the citation service.
        
        Args:
            config: LitForge configuration
        """
        self.config = config or get_config()
        self._discovery: Any | None = None
    
    @property
    def discovery(self) -> Any:
        """Lazy-load discovery service."""
        if self._discovery is None:
            from litforge.services.discovery import DiscoveryService
            self._discovery = DiscoveryService(self.config)
        return self._discovery
    
    def build_network(
        self,
        seed_papers: Sequence[Publication] | Sequence[str],
        depth: int = 2,
        max_papers: int = 500,
        direction: str = "both",  # "cited_by", "references", "both"
    ) -> CitationNetwork:
        """
        Build a citation network from seed papers.
        
        Args:
            seed_papers: Starting papers (Publication objects or DOIs)
            depth: How many citation hops to traverse
            max_papers: Maximum papers to include
            direction: Direction to traverse ("cited_by", "references", "both")
            
        Returns:
            Citation network
        """
        # Convert strings to publications
        papers: list[Publication] = []
        for p in seed_papers:
            if isinstance(p, str):
                pub = self.discovery.lookup(doi=p)
                if pub:
                    papers.append(pub)
            else:
                papers.append(p)
        
        if not papers:
            return CitationNetwork(
                seed_papers=[],
                nodes=[],
                edges=[],
                clusters=[],
            )
        
        # Build network through BFS
        all_papers: dict[str, Publication] = {}
        edges: list[tuple[str, str, str]] = []  # (source, target, type)
        
        queue: list[tuple[Publication, int]] = [(p, 0) for p in papers]
        processed: set[str] = set()
        
        while queue and len(all_papers) < max_papers:
            paper, current_depth = queue.pop(0)
            
            paper_id = self._get_paper_id(paper)
            if paper_id in processed:
                continue
            processed.add(paper_id)
            
            # Add paper to network
            all_papers[paper_id] = paper
            
            # Don't expand beyond depth
            if current_depth >= depth:
                continue
            
            # Get citations if not at max depth
            if direction in ("cited_by", "both"):
                citing_papers = self._get_citing_papers(paper)
                for citing in citing_papers[:20]:  # Limit per paper
                    citing_id = self._get_paper_id(citing)
                    edges.append((citing_id, paper_id, "cites"))
                    if citing_id not in processed and len(all_papers) < max_papers:
                        queue.append((citing, current_depth + 1))
            
            if direction in ("references", "both"):
                referenced_papers = self._get_referenced_papers(paper)
                for ref in referenced_papers[:20]:  # Limit per paper
                    ref_id = self._get_paper_id(ref)
                    edges.append((paper_id, ref_id, "cites"))
                    if ref_id not in processed and len(all_papers) < max_papers:
                        queue.append((ref, current_depth + 1))
        
        # Build nodes
        nodes = [
            NetworkNode(
                id=pid,
                publication=pub,
                in_degree=0,
                out_degree=0,
                centrality=0.0,
            )
            for pid, pub in all_papers.items()
        ]
        
        # Build edges and compute degrees
        node_map = {n.id: n for n in nodes}
        network_edges = []
        
        for source, target, edge_type in edges:
            if source in node_map and target in node_map:
                network_edges.append(
                    NetworkEdge(
                        source=source,
                        target=target,
                        weight=1.0,
                        edge_type=edge_type,
                    )
                )
                node_map[source].out_degree += 1
                node_map[target].in_degree += 1
        
        # Compute centrality (simple PageRank approximation)
        self._compute_centrality(nodes, network_edges)
        
        # Find clusters
        clusters = self._find_clusters(nodes, network_edges)
        
        return CitationNetwork(
            seed_papers=[self._get_paper_id(p) for p in papers],
            nodes=nodes,
            edges=network_edges,
            clusters=clusters,
        )
    
    def find_key_papers(
        self,
        network: CitationNetwork,
        metric: str = "centrality",  # "centrality", "citations", "in_degree"
        limit: int = 10,
    ) -> list[Publication]:
        """
        Find the most important papers in a network.
        
        Args:
            network: Citation network
            metric: Metric to rank by
            limit: Number of papers to return
            
        Returns:
            List of key papers
        """
        if metric == "centrality":
            sorted_nodes = sorted(network.nodes, key=lambda n: n.centrality, reverse=True)
        elif metric == "citations":
            sorted_nodes = sorted(
                network.nodes,
                key=lambda n: n.publication.citation_count,
                reverse=True,
            )
        elif metric == "in_degree":
            sorted_nodes = sorted(network.nodes, key=lambda n: n.in_degree, reverse=True)
        else:
            sorted_nodes = network.nodes
        
        return [n.publication for n in sorted_nodes[:limit]]
    
    def find_clusters(
        self,
        network: CitationNetwork,
        algorithm: str = "louvain",
    ) -> list[Cluster]:
        """
        Find clusters of related papers in the network.
        
        Args:
            network: Citation network
            algorithm: Clustering algorithm ("louvain", "label_propagation")
            
        Returns:
            List of clusters
        """
        return self._find_clusters(network.nodes, network.edges, algorithm)
    
    def _get_paper_id(self, paper: Publication) -> str:
        """Get a unique identifier for a paper."""
        return paper.doi or paper.openalex_id or paper.semantic_scholar_id or paper.title
    
    def _get_citing_papers(self, paper: Publication) -> list[Publication]:
        """Get papers that cite this paper."""
        try:
            from litforge.clients.semantic_scholar import SemanticScholarClient
            client = SemanticScholarClient(
                api_key=self.config.sources.semantic_scholar_api_key
            )
            return client.get_citations(
                paper_id=paper.semantic_scholar_id or paper.doi,
                limit=50,
            )
        except Exception as e:
            logger.warning(f"Error getting citations: {e}")
            return []
    
    def _get_referenced_papers(self, paper: Publication) -> list[Publication]:
        """Get papers referenced by this paper."""
        try:
            from litforge.clients.semantic_scholar import SemanticScholarClient
            client = SemanticScholarClient(
                api_key=self.config.sources.semantic_scholar_api_key
            )
            return client.get_references(
                paper_id=paper.semantic_scholar_id or paper.doi,
                limit=50,
            )
        except Exception as e:
            logger.warning(f"Error getting references: {e}")
            return []
    
    def _compute_centrality(
        self,
        nodes: list[NetworkNode],
        edges: list[NetworkEdge],
    ) -> None:
        """Compute PageRank centrality for nodes."""
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            for node in nodes:
                G.add_node(node.id)
            for edge in edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
            
            if G.number_of_nodes() > 0:
                pagerank = nx.pagerank(G, alpha=0.85)
                
                for node in nodes:
                    node.centrality = pagerank.get(node.id, 0.0)
                    
        except ImportError:
            logger.warning("NetworkX not installed - centrality not computed")
            # Fall back to simple degree centrality
            total_edges = len(edges) or 1
            for node in nodes:
                node.centrality = (node.in_degree + node.out_degree) / total_edges
    
    def _find_clusters(
        self,
        nodes: list[NetworkNode],
        edges: list[NetworkEdge],
        algorithm: str = "louvain",
    ) -> list[Cluster]:
        """Find clusters using community detection."""
        try:
            import networkx as nx
            
            G = nx.Graph()  # Undirected for clustering
            for node in nodes:
                G.add_node(node.id)
            for edge in edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
            
            if G.number_of_nodes() == 0:
                return []
            
            # Use Louvain community detection
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
            except ImportError:
                # Fall back to label propagation
                from networkx.algorithms import community
                communities = community.label_propagation_communities(G)
                partition = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        partition[node] = i
            
            # Group nodes by cluster
            cluster_nodes: dict[int, list[str]] = defaultdict(list)
            for node_id, cluster_id in partition.items():
                cluster_nodes[cluster_id].append(node_id)
            
            # Create cluster objects
            node_map = {n.id: n for n in nodes}
            clusters = []
            
            for cluster_id, member_ids in cluster_nodes.items():
                members = [node_map[mid] for mid in member_ids if mid in node_map]
                if not members:
                    continue
                
                # Find most central node as representative
                representative = max(members, key=lambda n: n.centrality)
                
                # Extract common keywords for label
                all_keywords: list[str] = []
                for m in members:
                    if m.publication.keywords:
                        all_keywords.extend(m.publication.keywords)
                
                # Get top keywords
                keyword_counts: dict[str, int] = defaultdict(int)
                for kw in all_keywords:
                    keyword_counts[kw.lower()] += 1
                
                top_keywords = sorted(
                    keyword_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
                
                label = ", ".join(kw for kw, _ in top_keywords) if top_keywords else f"Cluster {cluster_id}"
                
                clusters.append(
                    Cluster(
                        id=str(cluster_id),
                        label=label,
                        members=member_ids,
                        representative=representative.id,
                        keywords=[kw for kw, _ in top_keywords],
                    )
                )
            
            return clusters
            
        except ImportError:
            logger.warning("NetworkX not installed - clustering not available")
            return []
