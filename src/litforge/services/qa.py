"""
QA Service - Question answering and summarization.

Provides RAG-based Q&A over the literature corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from litforge.config import LitForgeConfig, get_config
from litforge.models import Publication

logger = logging.getLogger(__name__)


@dataclass
class QAResponse:
    """Response from the Q&A service."""
    
    answer: str
    sources: list[dict[str, str]] = field(default_factory=list)
    confidence: str = "medium"  # low, medium, high
    context_used: bool = True
    related_questions: list[str] = field(default_factory=list)


@dataclass
class ChatResponse:
    """Response from the chat service."""
    
    response: str
    context_used: bool = False
    history_length: int = 0


@dataclass 
class ComparisonResult:
    """Result of comparing papers."""
    
    summary: str
    similarities: list[str] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)
    papers: list[str] = field(default_factory=list)


# Default prompts
SYSTEM_PROMPT = """You are a helpful research assistant with expertise in scientific literature.
Answer questions based on the provided context from scientific papers.
Always cite your sources by mentioning paper titles or authors.
If the context doesn't contain enough information to answer, say so clearly.
Be concise but thorough."""

RAG_PROMPT_TEMPLATE = """Context from scientific literature:

{context}

---

Question: {question}

Please answer based on the context above. Cite relevant sources."""

SUMMARIZE_PROMPT_TEMPLATE = """Please summarize the following scientific papers:

{papers}

Provide a coherent summary that:
1. Identifies the main themes and findings
2. Notes any contradictions or debates
3. Highlights key methodologies
4. Suggests gaps or future directions"""

COMPARE_PROMPT_TEMPLATE = """Compare the following scientific papers:

{papers}

Please provide:
1. A brief summary of each paper's main contribution
2. Key similarities between the papers
3. Key differences between the papers
4. How they relate to each other in the broader field"""

FOLLOW_UP_PROMPT = """Based on the question "{question}" and the answer provided, suggest 3 related follow-up questions that would help deepen understanding of this topic.

Format as a numbered list."""

CHAT_SYSTEM_PROMPT = """You are a research assistant helping with literature review.
You have access to a corpus of scientific papers.
Answer questions about the literature, summarize findings, and help identify relevant papers.
Always cite sources when making claims based on specific papers."""


class QAService:
    """
    Service for question answering over literature.
    
    Provides:
    - RAG-based Q&A with source citations
    - Multi-turn chat conversations
    - Paper summarization
    """
    
    def __init__(self, config: LitForgeConfig | None = None):
        """
        Initialize the QA service.
        
        Args:
            config: LitForge configuration
        """
        self.config = config or get_config()
        self._llm: Any | None = None
        self._knowledge: Any | None = None
        self._chat_history: list[dict[str, str]] = []
    
    @property
    def knowledge(self) -> Any:
        """Lazy-load knowledge service."""
        if self._knowledge is None:
            from litforge.services.knowledge import KnowledgeService
            self._knowledge = KnowledgeService(self.config)
        return self._knowledge
    
    @property
    def llm(self) -> Any:
        """Lazy-load LLM provider."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm
    
    def _create_llm(self) -> Any:
        """Create LLM provider based on config."""
        provider = self.config.llm.provider
        
        if provider == "openai":
            from litforge.llm.openai import OpenAIProvider
            return OpenAIProvider(
                api_key=self.config.llm.openai_api_key,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
            )
        elif provider == "anthropic":
            from litforge.llm.anthropic import AnthropicProvider
            return AnthropicProvider(
                api_key=self.config.llm.anthropic_api_key,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
            )
        elif provider == "ollama":
            from litforge.llm.ollama import OllamaProvider
            return OllamaProvider(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def ask(
        self,
        question: str,
        context_limit: int = 5,
        max_context_tokens: int = 4000,
        generate_follow_ups: bool = False,
    ) -> QAResponse:
        """
        Ask a question about the indexed literature.
        
        Args:
            question: Question to answer
            context_limit: Maximum context chunks to retrieve
            max_context_tokens: Maximum tokens in context
            generate_follow_ups: Generate related follow-up questions
            
        Returns:
            QAResponse with answer and sources
        """
        # Retrieve context
        context = self.knowledge.get_context(
            query=question,
            limit=context_limit,
            max_tokens=max_context_tokens,
        )
        
        if not context:
            return QAResponse(
                answer="I don't have enough information in the knowledge base to answer this question. Please index some relevant papers first.",
                sources=[],
                context_used=False,
                confidence="low",
            )
        
        # Build prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )
        
        # Get LLM response
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        
        # Extract sources from context
        sources = self._extract_sources(context)
        
        # Determine confidence based on number of sources
        if len(sources) >= 3:
            confidence = "high"
        elif len(sources) >= 1:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Generate follow-up questions if requested
        related_questions = []
        if generate_follow_ups:
            related_questions = self._generate_follow_ups(question, response)
        
        return QAResponse(
            answer=response,
            sources=sources,
            context_used=True,
            confidence=confidence,
            related_questions=related_questions,
        )
    
    def _generate_follow_ups(self, question: str, answer: str) -> list[str]:
        """Generate follow-up questions based on Q&A."""
        try:
            prompt = FOLLOW_UP_PROMPT.format(question=question)
            response = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
            
            # Parse numbered list
            questions = []
            for line in response.split("\n"):
                line = line.strip()
                # Remove numbering
                if line and line[0].isdigit():
                    # Remove "1." or "1)" etc.
                    parts = line.split(".", 1) if "." in line[:3] else line.split(")", 1)
                    if len(parts) > 1:
                        questions.append(parts[1].strip())
                    else:
                        questions.append(line)
            
            return questions[:3]  # Limit to 3
        except Exception as e:
            logger.warning(f"Failed to generate follow-up questions: {e}")
            return []
    
    def compare(
        self,
        publications: Sequence[Publication],
    ) -> ComparisonResult:
        """
        Compare multiple papers to identify similarities and differences.
        
        Args:
            publications: Papers to compare (2-5 papers recommended)
            
        Returns:
            ComparisonResult with summary, similarities, and differences
        """
        if len(publications) < 2:
            return ComparisonResult(
                summary="Need at least 2 papers to compare.",
                papers=[p.title for p in publications],
            )
        
        # Build paper descriptions
        paper_descriptions = []
        paper_titles = []
        for i, pub in enumerate(publications[:5], 1):  # Limit to 5 papers
            paper_titles.append(pub.title)
            authors = ", ".join(a.name for a in pub.authors[:3])
            if len(pub.authors) > 3:
                authors += " et al."
            
            desc = f"{i}. {pub.title}"
            desc += f"\n   Authors: {authors}"
            if pub.year:
                desc += f"\n   Year: {pub.year}"
            if pub.abstract:
                desc += f"\n   Abstract: {pub.abstract[:600]}"
            
            paper_descriptions.append(desc)
        
        papers_text = "\n\n".join(paper_descriptions)
        
        # Get comparison
        prompt = COMPARE_PROMPT_TEMPLATE.format(papers=papers_text)
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        
        # Parse response to extract similarities/differences
        similarities = self._extract_list_items(response, "similarities")
        differences = self._extract_list_items(response, "differences")
        
        return ComparisonResult(
            summary=response,
            similarities=similarities,
            differences=differences,
            papers=paper_titles,
        )
    
    def _extract_list_items(self, text: str, section_name: str) -> list[str]:
        """Extract list items from a section of text."""
        items = []
        in_section = False
        
        for line in text.split("\n"):
            line_lower = line.lower()
            
            # Check if we're entering the target section
            if section_name.lower() in line_lower and (":" in line or "#" in line):
                in_section = True
                continue
            
            # Check if we're leaving the section
            if in_section and line.strip() and not line.startswith(" ") and not line.startswith("-") and not line[0].isdigit():
                if any(word in line_lower for word in ["difference", "similar", "summary", "relate"]):
                    in_section = False
                    continue
            
            # Extract list items
            if in_section and line.strip():
                item = line.strip()
                if item.startswith("-") or item.startswith("•"):
                    item = item[1:].strip()
                elif item[0].isdigit() and ("." in item[:3] or ")" in item[:3]):
                    item = item.split(".", 1)[-1].split(")", 1)[-1].strip()
                
                if item and len(item) > 10:
                    items.append(item)
        
        return items[:5]  # Limit to 5 items

    def chat(
        self,
        message: str,
        use_context: bool = True,
    ) -> ChatResponse:
        """
        Multi-turn chat about the literature.
        
        Args:
            message: User message
            use_context: Whether to retrieve context for the response
            
        Returns:
            ChatResponse with response and metadata
        """
        # Add user message to history
        self._chat_history.append({"role": "user", "content": message})
        
        # Get context if enabled
        context = ""
        if use_context:
            context = self.knowledge.get_context(query=message, limit=3)
        
        # Build messages
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT}
        ]
        
        # Add context if available
        if context:
            messages.append({
                "role": "system",
                "content": f"Relevant context from indexed papers:\n\n{context}",
            })
        
        # Add conversation history (limit to last 10 exchanges)
        messages.extend(self._chat_history[-20:])
        
        # Get response
        response = self.llm.chat(messages=messages)
        
        # Add to history
        self._chat_history.append({"role": "assistant", "content": response})
        
        return ChatResponse(
            response=response,
            context_used=bool(context),
            history_length=len(self._chat_history),
        )
    
    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        self._chat_history = []
    
    def summarize(
        self,
        publications: Sequence[Publication],
        focus: str | None = None,
    ) -> str:
        """
        Summarize a collection of papers.
        
        Args:
            publications: Papers to summarize
            focus: Optional focus area for the summary
            
        Returns:
            Summary text
        """
        # Build paper descriptions
        paper_descriptions = []
        for i, pub in enumerate(publications[:20], 1):  # Limit to 20 papers
            authors = ", ".join(a.name for a in pub.authors[:3])
            if len(pub.authors) > 3:
                authors += " et al."
            
            desc = f"{i}. {pub.title}"
            desc += f"\n   Authors: {authors}"
            if pub.year:
                desc += f"\n   Year: {pub.year}"
            if pub.abstract:
                # Truncate abstract
                abstract = pub.abstract[:500]
                if len(pub.abstract) > 500:
                    abstract += "..."
                desc += f"\n   Abstract: {abstract}"
            
            paper_descriptions.append(desc)
        
        papers_text = "\n\n".join(paper_descriptions)
        
        # Build prompt
        prompt = SUMMARIZE_PROMPT_TEMPLATE.format(papers=papers_text)
        
        if focus:
            prompt += f"\n\nFocus especially on: {focus}"
        
        # Get summary
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        
        return response
    
    def _extract_sources(self, context: str) -> list[dict[str, str]]:
        """Extract source information from context."""
        sources = []
        
        for line in context.split("\n"):
            if line.startswith("[Source:"):
                # Extract title from [Source: Title]
                title = line.replace("[Source:", "").rstrip("]").strip()
                if title and title != "Unknown":
                    sources.append({"title": title})
        
        return sources
