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
from litforge.config import load_config


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
        ctx.obj["config"] = load_config(Path(config))
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
    from litforge.models import SearchFilter
    filters = SearchFilter(
        year_from=year_from,
        year_to=year_to,
        open_access=open_access if open_access else None,
    )
    
    # Search
    result = forge.search(
        query,
        sources=list(sources),
        limit=limit,
        filters=filters,
    )
    
    # Output
    if output == "json":
        click.echo(json.dumps([p.to_dict() for p in result.publications], indent=2))
    elif output == "bibtex":
        for pub in result.publications:
            click.echo(pub.to_bibtex())
            click.echo()
    else:
        click.echo(f"Found {result.total_count} papers:\n")
        for i, pub in enumerate(result.publications, 1):
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
        click.echo(json.dumps(paper.to_dict(), indent=2))
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
    
    result = forge.ask(query, context_limit=limit)
    
    click.echo(f"\n{result['answer']}\n")
    
    if result.get("sources"):
        click.echo("Sources:")
        for src in result["sources"]:
            click.echo(f"  - {src.get('title', 'Unknown')}")


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
    
    click.echo(f"Found {len(result.publications)} papers. Indexing...")
    
    chunks = forge.index(result.publications, include_fulltext=fulltext)
    
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
        click.echo(f"\nAssistant: {result['response']}\n")


@cli.command()
@click.pass_context
def clear(ctx: click.Context) -> None:
    """Clear the knowledge base."""
    forge = Forge(config=ctx.obj["config"])
    
    if click.confirm("Are you sure you want to clear the knowledge base?"):
        forge.clear_knowledge()
        click.echo("Knowledge base cleared.")


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


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
