"""
LitForge Web UI - Streamlit-based frontend for literature search.

Run with:
    streamlit run src/litforge/ui/app.py
    
Or via CLI:
    litforge ui
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from litforge.api import search, lookup, citations, references, Paper


def make_doi_link(doi: str) -> str:
    """Create a clickable DOI link."""
    if not doi or doi == "-":
        return ""
    clean_doi = doi.replace("https://doi.org/", "")
    return f"https://doi.org/{clean_doi}"


def main():
    """Main Streamlit app."""
    
    # Page config
    st.set_page_config(
        page_title="LitForge - Scientific Literature Search",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .paper-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: #fafafa;
    }
    .paper-title {
        font-size: 16px;
        font-weight: bold;
        color: #1a1a1a;
        margin-bottom: 8px;
    }
    .paper-meta {
        color: #666;
        font-size: 14px;
    }
    .paper-abstract {
        color: #444;
        font-size: 13px;
        line-height: 1.5;
        margin-top: 10px;
    }
    .doi-link {
        color: #0066cc;
        text-decoration: none;
    }
    .doi-link:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🔥 LitForge")
    st.markdown("**Forging Knowledge from Literature** - Search 200M+ scientific papers")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        max_results = st.slider("Max Results", 5, 100, 25)
        
        sort_by = st.selectbox(
            "Sort By",
            ["citations", "date", "relevance"],
            format_func=lambda x: {
                "citations": "📊 Most Cited",
                "date": "📅 Most Recent",
                "relevance": "🎯 Most Relevant"
            }[x]
        )
        
        st.markdown("---")
        st.markdown("**Year Filter**")
        col1, col2 = st.columns(2)
        with col1:
            year_from = st.number_input("From", min_value=1900, max_value=2026, value=2018)
        with col2:
            year_to = st.number_input("To", min_value=1900, max_value=2026, value=2026)
        
        st.markdown("---")
        st.markdown("**Data Sources**")
        st.info("🌐 OpenAlex (200M+ papers)\n📚 CrossRef (130M+ papers)")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        LitForge is an open-source tool for:
        - 🔍 Literature search
        - 📊 Citation analysis  
        - 🧠 Knowledge synthesis
        
        [GitHub](https://github.com/sdodlapati/litforge)
        """)
    
    # Initialize session state for storing results
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "search_performed" not in st.session_state:
        st.session_state.search_performed = False
    
    # Main search tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📄 Lookup DOI", "📊 Citation Explorer"])
    
    with tab1:
        st.subheader("Search Scientific Literature")
        
        # Search input with clear guidance
        st.caption("🔍 Enter keywords to find papers (e.g., 'linear attention transformer', 'CRISPR gene editing')")
        query = st.text_input(
            "Search query",
            placeholder="e.g., linear attention transformers, CRISPR gene editing...",
            key="search_query",
            label_visibility="collapsed"
        )
        
        search_button = st.button("🔍 Search", type="primary", key="search_btn")
        
        # Perform search
        if search_button and query:
            # Check if this looks like a command rather than a search query
            command_phrases = ["download", "can you", "please", "export", "save", "get me", "show me the"]
            is_command = any(phrase in query.lower() for phrase in command_phrases)
            
            if is_command:
                st.warning("💡 **Tip:** This search box is for finding papers, not commands. Use the **export buttons** above to download papers!")
            else:
                with st.spinner(f"Searching for '{query}'..."):
                    try:
                        papers = search(
                            query,
                            limit=max_results,
                            year_from=year_from,
                            year_to=year_to,
                            sort_by=sort_by,
                        )
                        # Store in session state only if we got results
                        if papers:
                            st.session_state.search_results = papers
                            st.session_state.last_query = query
                        st.session_state.search_performed = True
                    except Exception as e:
                        st.error(f"Search error: {e}")
                        st.session_state.search_performed = True
        
        # Display results from session state
        papers = st.session_state.search_results
        
        if papers:
            st.success(f"Found {len(papers)} papers for: **{st.session_state.last_query}**")
            
            # Quick export buttons at top for convenience - PROMINENT
            st.markdown("---")
            st.markdown("### 📥 Download Papers")
            st.info("👆 **To download papers:** Select how many, then click a format button below.")
            qcol1, qcol2, qcol3, qcol4 = st.columns(4)
            with qcol1:
                num_export = st.selectbox("How many papers?", [5, 10, 15, 25, len(papers)], index=1, key="num_export")
            with qcol2:
                export_papers = papers[:num_export]
                csv = papers_to_csv(export_papers)
                st.download_button(
                    f"📥 CSV ({num_export})",
                    csv,
                    f"litforge_top{num_export}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key="quick_csv"
                )
            with qcol3:
                bibtex = papers_to_bibtex(export_papers)
                st.download_button(
                    f"📚 BibTeX ({num_export})",
                    bibtex,
                    f"litforge_top{num_export}_{datetime.now().strftime('%Y%m%d')}.bib",
                    "text/plain",
                    key="quick_bib"
                )
            with qcol4:
                json_data = papers_to_json(export_papers)
                st.download_button(
                    f"📋 JSON ({num_export})",
                    json_data,
                    f"litforge_top{num_export}_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json",
                    key="quick_json"
                )
            
            st.markdown("---")
            
            # Display results
            display_papers_enhanced(papers)
            
            # Export options at bottom too
            st.markdown("---")
            st.subheader("📥 Export All Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = papers_to_csv(papers)
                st.download_button(
                    "📥 Download All CSV",
                    csv,
                    f"litforge_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key="all_csv"
                )
            with col2:
                bibtex = papers_to_bibtex(papers)
                st.download_button(
                    "📚 Download All BibTeX",
                    bibtex,
                    f"litforge_results_{datetime.now().strftime('%Y%m%d')}.bib",
                    "text/plain",
                    key="all_bib"
                )
            with col3:
                json_data = papers_to_json(papers)
                st.download_button(
                    "📋 Download All JSON",
                    json_data,
                    f"litforge_results_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json",
                    key="all_json"
                )
            
            # New search section at bottom
            st.markdown("---")
            st.subheader("🔍 New Search")
            new_query = st.text_input(
                "Enter a new search query",
                placeholder="Enter new query...",
                key="new_search_query"
            )
            if st.button("🔍 Search Again", type="primary", key="new_search_btn"):
                if new_query:
                    st.session_state.search_results = []  # Clear old results
                    # Trigger rerun with new query
                    st.rerun()
        
        elif not papers and st.session_state.search_performed:
            # Only show "no papers found" after a search was performed
            st.warning("No papers found. Try a different query.")
    
    with tab2:
        st.subheader("Look Up Paper by DOI")
        
        doi_input = st.text_input(
            "Enter DOI",
            placeholder="e.g., 10.1038/nature14539",
            key="doi_input"
        )
        
        lookup_button = st.button("🔍 Look Up", type="primary", key="lookup_btn")
        
        if lookup_button and doi_input:
            with st.spinner(f"Looking up {doi_input}..."):
                try:
                    paper = lookup(doi_input)
                    
                    if paper:
                        st.success("Paper found!")
                        display_paper_card(paper)
                    else:
                        st.warning("Paper not found. Check the DOI.")
                        
                except Exception as e:
                    st.error(f"Lookup error: {e}")
    
    with tab3:
        st.subheader("Citation Explorer")
        
        cite_doi = st.text_input(
            "Enter DOI to explore citations",
            placeholder="e.g., 10.1126/science.1225829",
            key="cite_doi"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            get_citing = st.button("📈 Papers Citing This", type="primary")
        with col2:
            get_refs = st.button("📚 References")
        
        if get_citing and cite_doi:
            with st.spinner("Finding citing papers..."):
                try:
                    citing_papers = citations(cite_doi, limit=max_results)
                    
                    if citing_papers:
                        st.success(f"Found {len(citing_papers)} citing papers")
                        display_papers_enhanced(citing_papers)
                    else:
                        st.info("No citing papers found.")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if get_refs and cite_doi:
            with st.spinner("Finding references..."):
                try:
                    ref_papers = references(cite_doi, limit=max_results)
                    
                    if ref_papers:
                        st.success(f"Found {len(ref_papers)} references")
                        display_papers_enhanced(ref_papers)
                    else:
                        st.info("No references found.")
                        
                except Exception as e:
                    st.error(f"Error: {e}")


def display_papers_enhanced(papers: list[Paper]):
    """Display papers with clickable links and expandable abstracts."""
    
    # View mode toggle
    view_mode = st.radio(
        "View Mode",
        ["📋 Card View", "📊 Table View"],
        horizontal=True,
        key=f"view_mode_{id(papers)}"
    )
    
    if view_mode == "📊 Table View":
        display_papers_table(papers)
    else:
        display_papers_cards(papers)


def display_papers_cards(papers: list[Paper]):
    """Display papers as expandable cards."""
    
    for i, p in enumerate(papers, 1):
        doi_link = make_doi_link(p.doi)
        
        with st.expander(f"**{i}. {p.title}** ({p.year}) - {p.citations:,} citations", expanded=False):
            # Metadata row
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**👥 Authors:** {', '.join(p.authors)}")
            with col2:
                st.markdown(f"**📅 Year:** {p.year}")
            with col3:
                st.markdown(f"**📊 Citations:** {p.citations:,}")
            
            if p.venue:
                st.markdown(f"**📚 Venue:** {p.venue}")
            
            # DOI with clickable link
            if p.doi and p.doi != "-":
                st.markdown(f"**🔗 DOI:** [{p.doi}]({doi_link})")
                st.link_button("🌐 Open Paper in New Tab", doi_link)
            
            # Full abstract
            if p.abstract:
                st.markdown("---")
                st.markdown("**📝 Abstract:**")
                st.markdown(p.abstract)
            else:
                st.info("Abstract not available")


def display_papers_table(papers: list[Paper]):
    """Display papers in a table with clickable DOI links."""
    
    # Create DataFrame with clickable links
    data = []
    for p in papers:
        doi_link = make_doi_link(p.doi)
        data.append({
            "Title": p.title,
            "Authors": ", ".join(p.authors[:3]) + ("..." if len(p.authors) > 3 else ""),
            "Year": p.year,
            "Citations": p.citations,
            "Venue": p.venue[:40] + "..." if p.venue and len(p.venue) > 40 else (p.venue or "-"),
            "Link": doi_link if p.doi else None,
        })
    
    df = pd.DataFrame(data)
    
    # Configure column display with clickable link
    st.dataframe(
        df,
        column_config={
            "Title": st.column_config.TextColumn("📄 Title", width="large"),
            "Authors": st.column_config.TextColumn("👥 Authors", width="medium"),
            "Year": st.column_config.NumberColumn("📅 Year", format="%d"),
            "Citations": st.column_config.NumberColumn("📊 Citations", format="%d"),
            "Venue": st.column_config.TextColumn("📚 Venue", width="medium"),
            "Link": st.column_config.LinkColumn("🔗 Open", display_text="Open"),
        },
        hide_index=True,
        use_container_width=True,
    )
    
    # Expandable abstracts section
    st.markdown("---")
    with st.expander("📖 View All Abstracts", expanded=False):
        for i, p in enumerate(papers, 1):
            st.markdown(f"### {i}. {p.title}")
            
            doi_link = make_doi_link(p.doi)
            st.markdown(f"*{', '.join(p.authors[:3])}{'...' if len(p.authors) > 3 else ''} ({p.year})*")
            
            if p.doi:
                st.markdown(f"[🔗 Open Paper]({doi_link})")
            
            if p.abstract:
                st.markdown(p.abstract)
            else:
                st.info("Abstract not available")
            
            st.markdown("---")


def display_paper_card(paper: Paper):
    """Display a single paper as a detailed card."""
    
    st.markdown(f"## {paper.title}")
    
    doi_link = make_doi_link(paper.doi)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 Year", paper.year or "N/A")
    with col2:
        st.metric("📊 Citations", f"{paper.citations:,}" if paper.citations else "N/A")
    with col3:
        if paper.doi:
            st.markdown(f"**🔗 DOI**")
            st.markdown(f"[{paper.doi}]({doi_link})")
        else:
            st.metric("🔗 DOI", "N/A")
    with col4:
        if paper.doi:
            st.link_button("🌐 Open Paper", doi_link)
    
    st.markdown("---")
    
    # Authors
    st.markdown(f"**👥 Authors:** {', '.join(paper.authors)}")
    
    # Venue
    if paper.venue:
        st.markdown(f"**📚 Venue:** {paper.venue}")
    
    # Abstract
    st.markdown("---")
    st.markdown("### 📝 Abstract")
    if paper.abstract:
        st.markdown(paper.abstract)
    else:
        st.info("Abstract not available for this paper.")
    
    # URL link
    if paper.url:
        st.markdown("---")
        st.link_button("🔗 View on OpenAlex", paper.url)


def papers_to_csv(papers: list[Paper]) -> str:
    """Convert papers to CSV string."""
    data = []
    for p in papers:
        data.append({
            "title": p.title,
            "authors": "; ".join(p.authors),
            "year": p.year,
            "citations": p.citations,
            "venue": p.venue,
            "doi": p.doi,
            "doi_url": make_doi_link(p.doi),
            "abstract": p.abstract,
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def papers_to_json(papers: list[Paper]) -> str:
    """Convert papers to JSON string."""
    import json
    
    data = []
    for p in papers:
        data.append({
            "title": p.title,
            "authors": p.authors,
            "year": p.year,
            "citations": p.citations,
            "venue": p.venue,
            "doi": p.doi,
            "doi_url": make_doi_link(p.doi),
            "abstract": p.abstract,
            "url": p.url,
        })
    
    return json.dumps(data, indent=2)


def papers_to_bibtex(papers: list[Paper]) -> str:
    """Convert papers to BibTeX string."""
    
    bibtex_entries = []
    
    for i, p in enumerate(papers, 1):
        # Generate citation key
        first_author = p.authors[0].split()[-1] if p.authors else "unknown"
        # Clean author name for key
        first_author = ''.join(c for c in first_author if c.isalnum())
        key = f"{first_author.lower()}{p.year or 'nd'}_{i}"
        
        # Escape special characters in title
        safe_title = p.title.replace("{", "\\{").replace("}", "\\}")
        
        entry = f"""@article{{{key},
  title = {{{safe_title}}},
  author = {{{' and '.join(p.authors)}}},
  year = {{{p.year or 'n.d.'}}},
  journal = {{{p.venue or ''}}},
  doi = {{{p.doi or ''}}}
}}"""
        bibtex_entries.append(entry)
    
    return "\n\n".join(bibtex_entries)


if __name__ == "__main__":
    main()
