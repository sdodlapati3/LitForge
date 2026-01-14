"""
LitForge CLI - Command-line interface for literature research.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import click

from litforge import Forge
from litforge.config import LitForgeConfig


@click.group()
@click.version_option()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[str]) -> None:
    """LitForge - Forging Knowledge from Literature.
    
    A powerful tool for scientific literature discovery, retrieval, and analysis.
    """
    ctx.ensure_object(dict)
    
    if config:
        ctx.obj["config"] = LitForgeConfig.load(Path(config))
    else:
        ctx.obj["config"] = None


@cli.command()
@click.argument("query")
@click.option(
    "--sources",
    "-s",
    multiple=True,
    default=["openalex"],
    help="Data sources to search (openalex, semantic_scholar, pubmed, arxiv)",
)
@click.option("--limit", "-n", default=10, help="Maximum results")
@click.option("--year-from", type=int, help="Filter by year (from)")
@click.option("--year-to", type=int, help="Filter by year (to)")
@click.option("--open-access", is_flag=True, help="Only open access papers")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json", "bibtex"]),
    default="text",
    help="Output format",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    sources: tuple[str, ...],
    limit: int,
    year_from: Optional[int],
    year_to: Optional[int],
    open_access: bool,
    output: str,
) -> None:
    """Search for scientific publications.
    
    Example: litforge search "CRISPR gene editing" --limit 5
    """
    forge = Forge(config=ctx.obj["config"])
    
    # Build filters
    filters = {}
    if year_from:
        filters["year_from"] = year_from
    if year_to:
        filters["year_to"] = year_to
    if open_access:
        filters["open_access_only"] = True
    
    # Search
    papers = forge.search(
        query,
        sources=list(sources),
        limit=limit,
        **filters,
    )
    
    # Output
    if output == "json":
        click.echo(json.dumps([p.model_dump() for p in papers], indent=2, default=str))
    elif output == "bibtex":
        for pub in papers:
            click.echo(pub.to_bibtex())
            click.echo()
    else:
        click.echo(f"Found {len(papers)} papers:\n")
        for i, pub in enumerate(papers, 1):
            _print_paper(i, pub)


@cli.command()
@click.argument("identifier")
@click.option(
    "--type",
    "-t",
    "id_type",
    type=click.Choice(["doi", "pmid", "arxiv"]),
    default="doi",
    help="Identifier type",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json", "bibtex"]),
    default="text",
    help="Output format",
)
@click.pass_context
def lookup(
    ctx: click.Context,
    identifier: str,
    id_type: str,
    output: str,
) -> None:
    """Look up a specific paper by identifier.
    
    Example: litforge lookup "10.1038/nature12373" --type doi
    """
    forge = Forge(config=ctx.obj["config"])
    
    kwargs = {id_type: identifier}
    paper = forge.lookup(**kwargs)
    
    if not paper:
        click.echo(f"Paper not found: {identifier}", err=True)
        sys.exit(1)
    
    if output == "json":
        click.echo(json.dumps(paper.model_dump(), indent=2, default=str))
    elif output == "bibtex":
        click.echo(paper.to_bibtex())
    else:
        _print_paper(1, paper, verbose=True)


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Maximum context chunks")
@click.pass_context
def ask(ctx: click.Context, query: str, limit: int) -> None:
    """Ask a question about indexed papers.
    
    Example: litforge ask "What are the main CRISPR delivery methods?"
    """
    forge = Forge(config=ctx.obj["config"])
    
    result = forge.ask(query, max_sources=limit)
    
    click.echo(f"\n{result.answer}\n")
    
    if result.sources:
        click.echo("Sources:")
        for src in result.sources:
            click.echo(f"  - {src.get('title', 'Unknown')}")
    
    if result.related_questions:
        click.echo("\nRelated questions:")
        for q in result.related_questions:
            click.echo(f"  - {q}")


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=20, help="Maximum papers to index")
@click.option("--fulltext/--no-fulltext", default=False, help="Include full text")
@click.pass_context
def index(
    ctx: click.Context,
    query: str,
    limit: int,
    fulltext: bool,
) -> None:
    """Search and index papers into knowledge base.
    
    Example: litforge index "machine learning drug discovery" --limit 50
    """
    forge = Forge(config=ctx.obj["config"])
    
    click.echo(f"Searching for papers matching: {query}")
    result = forge.search(query, limit=limit)
    
    click.echo(f"Found {len(result)} papers. Indexing...")
    
    chunks = forge.index(result, include_full_text=fulltext)
    
    click.echo(f"Indexed {chunks} chunks into knowledge base.")


@cli.command()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Start interactive chat session.
    
    Example: litforge chat
    """
    forge = Forge(config=ctx.obj["config"])
    
    click.echo("LitForge Chat - Ask questions about your indexed papers.")
    click.echo("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            message = click.prompt("You", prompt_suffix="> ")
        except click.Abort:
            break
        
        if message.lower() in ("quit", "exit", "q"):
            break
        
        result = forge.chat(message)
        click.echo(f"\nAssistant: {result.response}\n")


@cli.command()
@click.pass_context
def clear(ctx: click.Context) -> None:
    """Clear the knowledge base."""
    forge = Forge(config=ctx.obj["config"])
    
    if click.confirm("Are you sure you want to clear the knowledge base?"):
        forge.clear_knowledge()
        click.echo("Knowledge base cleared.")


@cli.command()
@click.argument("doi")
@click.option("--depth", "-d", default=2, help="Citation traversal depth")
@click.option("--max-papers", "-n", default=100, help="Maximum papers to include")
@click.option(
    "--direction",
    type=click.Choice(["both", "cited_by", "references"]),
    default="both",
    help="Citation direction to traverse",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path (JSON or GraphML)")
@click.pass_context
def network(
    ctx: click.Context,
    doi: str,
    depth: int,
    max_papers: int,
    direction: str,
    output: Optional[str],
) -> None:
    """Build a citation network from a seed paper.
    
    Example: litforge network "10.1038/nature14539" --depth 2 --output network.json
    """
    forge = Forge(config=ctx.obj["config"])
    
    click.echo(f"Looking up paper: {doi}")
    paper = forge.lookup(doi=doi)
    
    if not paper:
        click.echo(f"Paper not found: {doi}", err=True)
        sys.exit(1)
    
    click.echo(f"Building citation network (depth={depth}, max={max_papers})...")
    network = forge.build_network(
        [paper],
        depth=depth,
        max_papers=max_papers,
        direction=direction,
    )
    
    # Get stats
    stats = forge.get_network_stats(network)
    
    click.echo(f"\nNetwork Statistics:")
    click.echo(f"  Papers: {stats['node_count']}")
    click.echo(f"  Citations: {stats['edge_count']}")
    click.echo(f"  Clusters: {stats['cluster_count']}")
    click.echo(f"  Density: {stats['density']}")
    if stats.get('year_range'):
        click.echo(f"  Year range: {stats['year_range'][0]} - {stats['year_range'][1]}")
    
    # Find key papers
    key_papers = forge.find_key_papers(network, limit=5)
    if key_papers:
        click.echo(f"\nTop 5 Key Papers:")
        for i, p in enumerate(key_papers, 1):
            click.echo(f"  {i}. {p.title[:60]}...")
    
    # Export if requested
    if output:
        output_path = Path(output)
        if output_path.suffix == ".graphml":
            forge.export_network(network, output, format="graphml")
        else:
            forge.export_network(network, output, format="json")
        click.echo(f"\nNetwork exported to: {output}")


@cli.command()
@click.argument("identifier")
@click.option("--with-sections/--no-sections", default=True, help="Include section parsing")
@click.option("--output", "-o", type=click.Path(), help="Save full text to file")
@click.pass_context
def retrieve(
    ctx: click.Context,
    identifier: str,
    with_sections: bool,
    output: Optional[str],
) -> None:
    """Retrieve and extract text from a paper.
    
    Example: litforge retrieve "1706.03762" --with-sections
    """
    forge = Forge(config=ctx.obj["config"])
    
    click.echo(f"Retrieving paper: {identifier}")
    
    try:
        if with_sections:
            paper = forge.retrieve_with_sections(identifier)
        else:
            paper = forge.lookup(doi=identifier) or forge.lookup(arxiv_id=identifier)
            if paper:
                paper = forge.retrieve(paper)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    
    if not paper:
        click.echo(f"Paper not found: {identifier}", err=True)
        sys.exit(1)
    
    click.echo(f"\nTitle: {paper.title}")
    
    if paper.full_text:
        click.echo(f"Full text: {len(paper.full_text)} characters")
        
        if paper.sections:
            click.echo(f"\nSections found: {list(paper.sections.keys())}")
            for name, content in paper.sections.items():
                preview = content[:100].replace('\n', ' ').strip()
                click.echo(f"  - {name}: {preview}...")
        
        if output:
            with open(output, 'w') as f:
                f.write(f"# {paper.title}\n\n")
                if paper.sections:
                    for name, content in paper.sections.items():
                        f.write(f"## {name.upper()}\n\n{content}\n\n")
                else:
                    f.write(paper.full_text)
            click.echo(f"\nSaved to: {output}")
    else:
        click.echo("No full text available.")

def _print_paper(
    num: int,
    pub: Any,
    verbose: bool = False,
) -> None:
    """Print a paper to console."""
    # Authors
    authors = ", ".join(a.name for a in pub.authors[:3])
    if len(pub.authors) > 3:
        authors += " et al."
    
    click.echo(f"{num}. {pub.title}")
    click.echo(f"   Authors: {authors}")
    
    if pub.year:
        click.echo(f"   Year: {pub.year}")
    
    if pub.venue:
        click.echo(f"   Venue: {pub.venue}")
    
    if pub.doi:
        click.echo(f"   DOI: {pub.doi}")
    
    if pub.citation_count:
        click.echo(f"   Citations: {pub.citation_count}")
    
    if verbose and pub.abstract:
        click.echo(f"   Abstract: {pub.abstract[:500]}...")
    
    if pub.is_open_access:
        click.echo(f"   Open Access: Yes")
    
    click.echo()


@cli.command()
@click.option("--port", "-p", default=8501, help="Port to run the web UI on")
@click.option("--host", "-h", default="localhost", help="Host to bind to")
def ui(port: int, host: str) -> None:
    """Launch the LitForge web UI.
    
    Example: litforge ui --port 8501
    """
    import subprocess
    import sys
    from pathlib import Path
    
    ui_path = Path(__file__).parent / "ui" / "app.py"
    
    click.echo(f"🔥 Starting LitForge Web UI at http://{host}:{port}")
    click.echo("   Press Ctrl+C to stop")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(ui_path),
        "--server.port", str(port),
        "--server.address", host,
    ])


@cli.command()
def mcp() -> None:
    """Start the LitForge MCP server for AI assistants.
    
    Example: litforge mcp
    """
    click.echo("🔥 Starting LitForge MCP Server...")
    
    try:
        from litforge.mcp.server import main as mcp_main
        mcp_main()
    except ImportError:
        click.echo("Error: MCP not installed. Run: pip install mcp")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
