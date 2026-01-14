"""
MCP server usage example.

This file shows how to configure Claude Desktop to use LitForge
as an MCP server for literature research.

To use LitForge with Claude Desktop:

1. Install LitForge with MCP support:
   pip install litforge[mcp]

2. Add to your Claude Desktop config file:
   
   macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
   Windows: %APPDATA%/Claude/claude_desktop_config.json
   Linux: ~/.config/Claude/claude_desktop_config.json

   {
     "mcpServers": {
       "litforge": {
         "command": "python",
         "args": ["-m", "litforge.mcp"],
         "env": {
           "OPENAI_API_KEY": "sk-your-key-here"
         }
       }
     }
   }

3. Restart Claude Desktop

4. You can now ask Claude to:
   - "Search for papers on CRISPR gene editing"
   - "Look up the paper with DOI 10.1038/nature12373"
   - "Get papers that cite this important CRISPR paper"
   - "Download the full text of this open access paper"
   - "Build a citation network starting from these seed papers"

Available MCP Tools:
- search_papers: Search across OpenAlex, Semantic Scholar, PubMed, arXiv
- lookup_paper: Get details for a specific paper by DOI/PMID/arXiv ID
- get_citations: Find papers that cite a given paper
- get_references: Find papers referenced by a given paper
- retrieve_fulltext: Download full text from open access sources
- index_papers: Add papers to knowledge base for Q&A
- ask_papers: Ask questions about indexed papers (RAG)
- build_citation_network: Create citation graphs for visualization
- summarize_papers: Generate summaries of multiple papers
"""

# You can also run the MCP server directly for testing:
if __name__ == "__main__":
    from litforge.mcp import serve
    import asyncio
    
    print("Starting LitForge MCP server...")
    print("(This communicates via stdin/stdout for MCP protocol)")
    print("Press Ctrl+C to stop")
    
    asyncio.run(serve())
