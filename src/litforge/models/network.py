"""
Citation network data models.
"""

from __future__ import annotations

from typing import Any, Iterator

from pydantic import BaseModel, Field

from litforge.models.publication import Publication


class NetworkNode(BaseModel):
    """A node in the citation network."""
    
    id: str = Field(..., description="Node ID (usually publication ID)")
    publication: Publication | None = Field(default=None, description="Publication data")
    
    # Graph metrics
    in_degree: int = Field(default=0, description="Number of incoming citations")
    out_degree: int = Field(default=0, description="Number of outgoing references")
    
    # Centrality metrics (computed later)
    pagerank: float | None = Field(default=None, description="PageRank score")
    betweenness: float | None = Field(default=None, description="Betweenness centrality")
    
    # Cluster membership
    cluster_id: int | None = Field(default=None, description="Cluster/community ID")
    
    # Metadata
    depth: int = Field(default=0, description="Depth from seed papers")
    is_seed: bool = Field(default=False, description="Is this a seed paper")


class NetworkEdge(BaseModel):
    """An edge in the citation network."""
    
    source: str = Field(..., description="Source node ID (citing paper)")
    target: str = Field(..., description="Target node ID (cited paper)")
    
    # Edge metadata
    weight: float = Field(default=1.0, description="Edge weight")
    context: str | None = Field(default=None, description="Citation context")
    section: str | None = Field(default=None, description="Section where citation appears")


class Cluster(BaseModel):
    """A cluster of related papers in the network."""
    
    id: int = Field(..., description="Cluster ID")
    papers: list[str] = Field(default_factory=list, description="Paper IDs in cluster")
    
    # Cluster metadata
    size: int = Field(default=0, description="Number of papers")
    label: str | None = Field(default=None, description="Auto-generated label")
    keywords: list[str] = Field(default_factory=list, description="Common keywords")
    
    # Representative paper
    centroid_id: str | None = Field(default=None, description="Most central paper ID")
    
    # Metrics
    density: float | None = Field(default=None, description="Internal edge density")
    modularity_contribution: float | None = Field(default=None, description="Contribution to modularity")


class CitationNetwork(BaseModel):
    """
    A citation network graph.
    
    Represents papers and their citation relationships as a directed graph.
    """
    
    # Nodes and edges
    nodes: dict[str, NetworkNode] = Field(default_factory=dict, description="Nodes by ID")
    edges: list[NetworkEdge] = Field(default_factory=list, description="Citation edges")
    
    # Seed papers
    seed_ids: list[str] = Field(default_factory=list, description="Seed paper IDs")
    
    # Network metadata
    depth: int = Field(default=0, description="Max depth traversed")
    directed: bool = Field(default=True, description="Is directed graph")
    
    # Clusters (if computed)
    clusters: list[Cluster] = Field(default_factory=list, description="Paper clusters")
    
    # Stats
    created_at: str | None = Field(default=None, description="Creation timestamp")
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the network."""
        return len(self.nodes)
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the network."""
        return len(self.edges)
    
    @property
    def papers(self) -> list[Publication]:
        """Get all publications in the network."""
        return [n.publication for n in self.nodes.values() if n.publication]
    
    def __len__(self) -> int:
        return self.num_nodes
    
    def __contains__(self, paper_id: str) -> bool:
        return paper_id in self.nodes
    
    def __iter__(self) -> Iterator[NetworkNode]:
        return iter(self.nodes.values())
    
    def get_node(self, paper_id: str) -> NetworkNode | None:
        """Get a node by paper ID."""
        return self.nodes.get(paper_id)
    
    def get_paper(self, paper_id: str) -> Publication | None:
        """Get a publication by ID."""
        node = self.nodes.get(paper_id)
        return node.publication if node else None
    
    def add_node(self, publication: Publication, is_seed: bool = False, depth: int = 0) -> NetworkNode:
        """Add a paper to the network."""
        if publication.id in self.nodes:
            return self.nodes[publication.id]
        
        node = NetworkNode(
            id=publication.id,
            publication=publication,
            is_seed=is_seed,
            depth=depth,
        )
        self.nodes[publication.id] = node
        
        if is_seed and publication.id not in self.seed_ids:
            self.seed_ids.append(publication.id)
        
        return node
    
    def add_edge(
        self,
        citing_id: str,
        cited_id: str,
        context: str | None = None,
        section: str | None = None,
    ) -> NetworkEdge:
        """Add a citation edge."""
        edge = NetworkEdge(
            source=citing_id,
            target=cited_id,
            context=context,
            section=section,
        )
        self.edges.append(edge)
        
        # Update degrees
        if citing_id in self.nodes:
            self.nodes[citing_id].out_degree += 1
        if cited_id in self.nodes:
            self.nodes[cited_id].in_degree += 1
        
        return edge
    
    def citing_papers(self, paper_id: str) -> list[Publication]:
        """Get papers that cite the given paper."""
        citing_ids = [e.source for e in self.edges if e.target == paper_id]
        return [
            self.nodes[cid].publication 
            for cid in citing_ids 
            if cid in self.nodes and self.nodes[cid].publication
        ]
    
    def cited_papers(self, paper_id: str) -> list[Publication]:
        """Get papers cited by the given paper."""
        cited_ids = [e.target for e in self.edges if e.source == paper_id]
        return [
            self.nodes[cid].publication
            for cid in cited_ids
            if cid in self.nodes and self.nodes[cid].publication
        ]
    
    def most_cited(self, n: int = 10) -> list[Publication]:
        """Get the n most cited papers in the network."""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda x: x.in_degree,
            reverse=True
        )
        return [
            node.publication
            for node in sorted_nodes[:n]
            if node.publication
        ]
    
    def most_central(self, n: int = 10, metric: str = "pagerank") -> list[Publication]:
        """Get the n most central papers by the given metric."""
        if metric == "pagerank":
            key = lambda x: x.pagerank or 0
        elif metric == "betweenness":
            key = lambda x: x.betweenness or 0
        else:
            key = lambda x: x.in_degree
        
        sorted_nodes = sorted(self.nodes.values(), key=key, reverse=True)
        return [node.publication for node in sorted_nodes[:n] if node.publication]
    
    def get_seeds(self) -> list[Publication]:
        """Get seed papers."""
        return [
            self.nodes[sid].publication
            for sid in self.seed_ids
            if sid in self.nodes and self.nodes[sid].publication
        ]
    
    def find_clusters(self, algorithm: str = "louvain") -> list[Cluster]:
        """
        Find clusters in the network.
        
        Requires networkx to be installed.
        """
        # Implementation delegated to CitationService
        raise NotImplementedError(
            "Use CitationService.find_clusters(network) for cluster detection"
        )
    
    def to_networkx(self) -> Any:
        """
        Convert to NetworkX DiGraph.
        
        Returns:
            networkx.DiGraph
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx required: pip install litforge[graph]")
        
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            attrs = {
                "in_degree": node.in_degree,
                "out_degree": node.out_degree,
                "depth": node.depth,
                "is_seed": node.is_seed,
            }
            if node.publication:
                attrs.update({
                    "title": node.publication.title,
                    "year": node.publication.year,
                    "citations": node.publication.citation_count,
                })
            if node.pagerank:
                attrs["pagerank"] = node.pagerank
            if node.cluster_id is not None:
                attrs["cluster"] = node.cluster_id
            
            G.add_node(node_id, **attrs)
        
        # Add edges
        for edge in self.edges:
            G.add_edge(
                edge.source,
                edge.target,
                weight=edge.weight,
            )
        
        return G
    
    def export(self, path: str, format: str = "graphml") -> None:
        """
        Export network to file.
        
        Args:
            path: Output file path
            format: Export format (graphml, gexf, json, pajek)
        """
        G = self.to_networkx()
        
        import networkx as nx
        
        if format == "graphml":
            nx.write_graphml(G, path)
        elif format == "gexf":
            nx.write_gexf(G, path)
        elif format == "json":
            import json
            from networkx.readwrite import json_graph
            data = json_graph.node_link_data(G)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif format == "pajek":
            nx.write_pajek(G, path)
        else:
            raise ValueError(f"Unknown format: {format}")
