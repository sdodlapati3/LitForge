"""
Citation Service - Citation network analysis.

Builds and analyzes citation networks, finding clusters and key papers.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
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
                pagerank=0.0,
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
                    )
                )
                node_map[source].out_degree += 1
                node_map[target].in_degree += 1
        
        # Compute centrality (simple PageRank approximation)
        self._compute_centrality(nodes, network_edges)
        
        # Find clusters
        clusters = self._find_clusters(nodes, network_edges)
        
        # Convert nodes list to dict
        nodes_dict = {n.id: n for n in nodes}
        
        return CitationNetwork(
            seed_ids=[self._get_paper_id(p) for p in papers],
            nodes=nodes_dict,
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
        nodes = list(network.nodes.values())
        if metric == "centrality" or metric == "pagerank":
            sorted_nodes = sorted(
                nodes,
                key=lambda n: n.pagerank or 0.0,
                reverse=True,
            )
        elif metric == "citations":
            sorted_nodes = sorted(
                nodes,
                key=lambda n: n.publication.citation_count if n.publication else 0,
                reverse=True,
            )
        elif metric == "in_degree":
            sorted_nodes = sorted(nodes, key=lambda n: n.in_degree, reverse=True)
        else:
            sorted_nodes = nodes
        
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
        return getattr(paper, 'doi', None) or getattr(paper, 'openalex_id', None) or getattr(paper, 'semantic_scholar_id', None) or getattr(paper, 'title', 'unknown')
    
    def _get_ss_paper_id(self, paper: Publication) -> str | None:
        """Get the Semantic Scholar paper ID or DOI formatted for SS API."""
        ss_id = getattr(paper, 'semantic_scholar_id', None)
        if ss_id:
            return ss_id
        doi = getattr(paper, 'doi', None)
        if doi:
            return f"DOI:{doi}"
        arxiv_id = getattr(paper, 'arxiv_id', None)
        if arxiv_id:
            return f"arXiv:{arxiv_id}"
        pmid = getattr(paper, 'pmid', None)
        if pmid:
            return f"PMID:{pmid}"
        return None
    
    def _get_citing_papers(self, paper: Publication) -> list[Publication]:
        """Get papers that cite this paper. Prefers OpenAlex, falls back to Semantic Scholar."""
        # Try OpenAlex first (more complete coverage)
        openalex_id = getattr(paper, 'openalex_id', None)
        if openalex_id:
            try:
                from litforge.clients.openalex import OpenAlexClient
                client = OpenAlexClient(email=self.config.sources.openalex_email)
                return client.get_citations(paper.openalex_id, limit=50)
            except Exception as e:
                logger.debug(f"OpenAlex citations failed: {e}")
        
        # Fall back to Semantic Scholar
        paper_id = self._get_ss_paper_id(paper)
        if not paper_id:
            return []
        
        try:
            from litforge.clients.semantic_scholar import SemanticScholarClient
            client = SemanticScholarClient(
                api_key=self.config.sources.semantic_scholar_api_key
            )
            return client.get_citations(
                paper_id=paper_id,
                limit=50,
            )
        except Exception as e:
            logger.warning(f"Error getting citations: {e}")
            return []
    
    def _get_referenced_papers(self, paper: Publication) -> list[Publication]:
        """Get papers referenced by this paper. Prefers OpenAlex, falls back to Semantic Scholar."""
        # Try OpenAlex first (more complete coverage)
        openalex_id = getattr(paper, 'openalex_id', None)
        if openalex_id:
            try:
                from litforge.clients.openalex import OpenAlexClient
                client = OpenAlexClient(email=self.config.sources.openalex_email)
                return client.get_references(paper.openalex_id, limit=50)
            except Exception as e:
                logger.debug(f"OpenAlex references failed: {e}")
        
        # Fall back to Semantic Scholar
        paper_id = self._get_ss_paper_id(paper)
        if not paper_id:
            return []
        
        try:
            from litforge.clients.semantic_scholar import SemanticScholarClient
            client = SemanticScholarClient(
                api_key=self.config.sources.semantic_scholar_api_key
            )
            return client.get_references(
                paper_id=paper_id,
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
                    node.pagerank = pagerank.get(node.id, 0.0)
                    
        except ImportError:
            logger.warning("NetworkX not installed - centrality not computed")
            # Fall back to simple degree centrality
            total_edges = len(edges) or 1
            for node in nodes:
                node.pagerank = (node.in_degree + node.out_degree) / total_edges
    
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
                representative = max(members, key=lambda n: n.pagerank or 0.0)
                
                # Extract common keywords for label
                all_keywords: list[str] = []
                for m in members:
                    if m.publication and m.publication.keywords:
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
    
    def find_bridge_papers(
        self,
        network: CitationNetwork,
        limit: int = 10,
    ) -> list[Publication]:
        """
        Find bridge papers that connect different clusters.
        
        These papers cite or are cited by papers from multiple clusters,
        making them important for understanding cross-topic connections.
        
        Args:
            network: Citation network with clusters
            limit: Maximum papers to return
            
        Returns:
            List of bridge papers
        """
        if not network.clusters or len(network.clusters) < 2:
            return []
        
        # Build cluster membership map
        node_to_cluster: dict[str, int] = {}
        for cluster in network.clusters:
            for member_id in cluster.papers:
                node_to_cluster[member_id] = cluster.id
        
        # Count cross-cluster connections for each node
        bridge_scores: dict[str, int] = defaultdict(int)
        
        for edge in network.edges:
            source_cluster = node_to_cluster.get(edge.source)
            target_cluster = node_to_cluster.get(edge.target)
            
            if source_cluster and target_cluster and source_cluster != target_cluster:
                bridge_scores[edge.source] += 1
                bridge_scores[edge.target] += 1
        
        # Sort by bridge score
        sorted_nodes = sorted(
            bridge_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        # Get publications
        bridge_papers = []
        for node_id, score in sorted_nodes[:limit]:
            if node_id in network.nodes:
                bridge_papers.append(network.nodes[node_id].publication)
        
        return bridge_papers
    
    def get_citation_timeline(
        self,
        network: CitationNetwork,
    ) -> dict[int, list[Publication]]:
        """
        Get papers grouped by publication year.
        
        Args:
            network: Citation network
            
        Returns:
            Dict mapping year to list of publications
        """
        timeline: dict[int, list[Publication]] = defaultdict(list)
        
        for node in network.nodes.values():
            if node.publication and node.publication.year:
                timeline[node.publication.year].append(node.publication)
        
        # Sort by year
        return dict(sorted(timeline.items()))
    
    def export_json(
        self,
        network: CitationNetwork,
        path: str | Path,
    ) -> None:
        """
        Export network to JSON format.
        
        Args:
            network: Citation network
            path: Output file path
        """
        data = {
            "nodes": [
                {
                    "id": node.id,
                    "title": node.publication.title if node.publication else "",
                    "authors": [a.name for a in node.publication.authors[:5]] if node.publication else [],
                    "year": node.publication.year if node.publication else None,
                    "doi": node.publication.doi if node.publication else None,
                    "citations": node.publication.citation_count if node.publication else 0,
                    "in_degree": node.in_degree,
                    "out_degree": node.out_degree,
                    "pagerank": node.pagerank,
                }
                for node in network.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                }
                for edge in network.edges
            ],
            "clusters": [
                {
                    "id": cluster.id,
                    "label": cluster.label,
                    "members": cluster.papers,
                    "keywords": cluster.keywords,
                }
                for cluster in network.clusters
            ],
        }
        
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info(f"Exported network to {path}")
    
    def export_graphml(
        self,
        network: CitationNetwork,
        path: str | Path,
    ) -> None:
        """
        Export network to GraphML format for Gephi/Cytoscape.
        
        Args:
            network: Citation network
            path: Output file path
        """
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            
            # Add nodes with attributes
            for node in network.nodes.values():
                attrs = {
                    "title": node.publication.title if node.publication else "",
                    "year": node.publication.year if node.publication else 0,
                    "citations": node.publication.citation_count if node.publication else 0,
                    "in_degree": node.in_degree,
                    "out_degree": node.out_degree,
                    "pagerank": node.pagerank,
                }
                if node.publication and node.publication.doi:
                    attrs["doi"] = node.publication.doi
                
                G.add_node(node.id, **attrs)
            
            # Add edges
            for edge in network.edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
            
            # Write to file
            nx.write_graphml(G, str(path))
            logger.info(f"Exported network to {path}")
            
        except ImportError:
            raise ImportError("NetworkX is required for GraphML export. Install with: pip install networkx")
    
    def get_network_stats(
        self,
        network: CitationNetwork,
    ) -> dict[str, Any]:
        """
        Get statistics about the citation network.
        
        Args:
            network: Citation network
            
        Returns:
            Dictionary of network statistics
        """
        nodes = list(network.nodes.values())
        
        if not nodes:
            return {
                "node_count": 0,
                "edge_count": 0,
                "density": 0.0,
            }
        
        # Basic stats
        node_count = len(nodes)
        edge_count = len(network.edges)
        
        # Density
        max_edges = node_count * (node_count - 1)
        density = edge_count / max_edges if max_edges > 0 else 0.0
        
        # Degree stats
        in_degrees = [n.in_degree for n in nodes]
        out_degrees = [n.out_degree for n in nodes]
        
        # Year range
        years = [n.publication.year for n in nodes if n.publication and n.publication.year]
        
        # Citation stats
        citations = [
            n.publication.citation_count
            for n in nodes
            if n.publication and n.publication.citation_count
        ]
        
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "cluster_count": len(network.clusters),
            "density": round(density, 4),
            "avg_in_degree": round(sum(in_degrees) / node_count, 2),
            "avg_out_degree": round(sum(out_degrees) / node_count, 2),
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "year_range": (min(years), max(years)) if years else None,
            "total_citations": sum(citations) if citations else 0,
            "avg_citations": round(sum(citations) / len(citations), 1) if citations else 0,
        }

    def to_pyvis_html(
        self,
        network: CitationNetwork,
        height: str = "600px",
        width: str = "100%",
        bgcolor: str = "#1a1a2e",
        font_color: str = "#ffffff",
    ) -> str:
        """
        Generate interactive HTML visualization using Pyvis.
        
        Creates a force-directed graph with:
        - Node size based on citation count
        - Node color based on cluster
        - Hover info showing paper details
        - Drag, zoom, and pan interactions
        
        Args:
            network: Citation network to visualize
            height: Height of the visualization
            width: Width of the visualization  
            bgcolor: Background color
            font_color: Font color for labels
            
        Returns:
            HTML string that can be embedded in Streamlit
        """
        try:
            from pyvis.network import Network as PyvisNetwork
        except ImportError:
            raise ImportError("Pyvis is required for interactive visualization. Install with: pip install pyvis")
        
        # Create pyvis network
        net = PyvisNetwork(
            height=height,
            width=width,
            bgcolor=bgcolor,
            font_color=font_color,
            directed=True,
            notebook=False,
            cdn_resources='remote',  # Use CDN for vis.js
        )
        
        # Physics settings for better layout
        net.set_options("""
        {
            "nodes": {
                "font": {"size": 12, "face": "arial"},
                "scaling": {"min": 10, "max": 50}
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
                "color": {"opacity": 0.5},
                "smooth": {"type": "continuous"}
            },
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {"iterations": 150}
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "hideEdgesOnDrag": true,
                "navigationButtons": true,
                "keyboard": {"enabled": true}
            }
        }
        """)
        
        # Color palette for clusters
        cluster_colors = [
            "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#e91e63", "#00bcd4", "#ff5722", "#607d8b",
            "#8bc34a", "#ffc107", "#795548", "#673ab7", "#03a9f4",
        ]
        
        # Build cluster ID to color mapping
        cluster_map: dict[str, str] = {}
        for i, cluster in enumerate(network.clusters):
            color = cluster_colors[i % len(cluster_colors)]
            for node_id in cluster.papers:
                cluster_map[node_id] = color
        
        # Add nodes
        for node_id, node in network.nodes.items():
            pub = node.publication
            if not pub:
                continue
            
            # Node size based on citations (log scale)
            citations = getattr(pub, 'citation_count', 0) or 0
            import math
            size = 10 + min(40, math.log(citations + 1) * 5)
            
            # Node color based on cluster or seed status
            if node.is_seed:
                color = "#FFD700"  # Gold for seed papers
                border_color = "#FFA500"
                border_width = 3
            else:
                color = cluster_map.get(node_id, "#6c757d")
                border_color = color
                border_width = 1
            
            # Build hover title
            year = getattr(pub, 'year', 'Unknown')
            doi = getattr(pub, 'doi', None)
            title_short = pub.title[:80] + '...' if len(pub.title) > 80 else pub.title
            pagerank_str = f"{node.pagerank:.4f}" if node.pagerank else "N/A"
            
            hover_html = f"""
            <div style="max-width: 300px; padding: 10px;">
                <b>{pub.title}</b><br>
                <small>Year: {year}</small><br>
                <small>Citations: {citations:,}</small><br>
                <small>PageRank: {pagerank_str}</small>
                {f'<br><small>DOI: {doi}</small>' if doi else ''}
            </div>
            """
            
            # Label: short title
            label = title_short[:30] + '...' if len(title_short) > 30 else title_short
            
            net.add_node(
                node_id,
                label=label,
                title=hover_html,
                size=size,
                color={
                    "background": color,
                    "border": border_color,
                    "highlight": {"background": "#ffffff", "border": color}
                },
                borderWidth=border_width,
                font={"size": 10, "color": font_color},
            )
        
        # Add edges
        for edge in network.edges:
            if edge.source in network.nodes and edge.target in network.nodes:
                net.add_edge(
                    edge.source,
                    edge.target,
                    color={"color": "#666666", "opacity": 0.6},
                    width=1,
                )
        
        # Generate HTML
        html = net.generate_html()
        
        # Fix for Streamlit iframe embedding - remove scrollbars
        html = html.replace(
            '<body>',
            '<body style="margin:0; padding:0; overflow:hidden;">'
        )
        
        return html
