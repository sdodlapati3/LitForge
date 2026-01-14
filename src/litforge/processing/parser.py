"""
Section Parser for Scientific Papers.

Parses extracted text into standard scientific paper sections:
- Abstract
- Introduction
- Methods/Materials
- Results
- Discussion
- Conclusion
- References
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Common section header patterns in scientific papers
SECTION_PATTERNS = {
    "abstract": [
        r"^abstract\s*$",
        r"^summary\s*$",
    ],
    "introduction": [
        r"^(?:\d+\.?\s*)?introduction\s*$",
        r"^(?:\d+\.?\s*)?background\s*$",
        r"^(?:\d+\.?\s*)?motivation\s*$",
        r"^(?:\d+\.?\s*)?overview\s*$",
    ],
    "related_work": [
        r"^(?:\d+\.?\s*)?related\s+work\s*$",
        r"^(?:\d+\.?\s*)?literature\s+review\s*$",
        r"^(?:\d+\.?\s*)?previous\s+work\s*$",
        r"^(?:\d+\.?\s*)?prior\s+work\s*$",
    ],
    "methods": [
        r"^(?:\d+\.?\s*)?methods?\s*$",
        r"^(?:\d+\.?\s*)?materials?\s+and\s+methods?\s*$",
        r"^(?:\d+\.?\s*)?methodology\s*$",
        r"^(?:\d+\.?\s*)?experimental\s+(?:section|methods?|procedures?|setup)\s*$",
        r"^(?:\d+\.?\s*)?methods?\s+and\s+materials?\s*$",
        r"^(?:\d+\.?\s*)?approach\s*$",
        r"^(?:\d+\.?\s*)?proposed\s+(?:method|approach|model|framework)\s*$",
        r"^(?:\d+\.?\s*)?model(?:ing)?\s*$",
        r"^(?:\d+\.?\s*)?architecture\s*$",
        r"^(?:\d+\.?\s*)?framework\s*$",
        r"^(?:\d+\.?\s*)?system\s+(?:design|overview|architecture)\s*$",
        r"^(?:\d+\.?\s*)?implementation\s*$",
    ],
    "experiments": [
        r"^(?:\d+\.?\s*)?experiments?\s*$",
        r"^(?:\d+\.?\s*)?experimental\s+(?:results?|evaluation|analysis)\s*$",
        r"^(?:\d+\.?\s*)?evaluation\s*$",
        r"^(?:\d+\.?\s*)?empirical\s+(?:results?|evaluation|study)\s*$",
        r"^(?:\d+\.?\s*)?case\s+stud(?:y|ies)\s*$",
        r"^(?:\d+\.?\s*)?training\s*$",
    ],
    "results": [
        r"^(?:\d+\.?\s*)?results?\s*$",
        r"^(?:\d+\.?\s*)?results?\s+and\s+discussion\s*$",
        r"^(?:\d+\.?\s*)?findings?\s*$",
        r"^(?:\d+\.?\s*)?experimental\s+results?\s*$",
        r"^(?:\d+\.?\s*)?main\s+results?\s*$",
        r"^(?:\d+\.?\s*)?quantitative\s+results?\s*$",
    ],
    "discussion": [
        r"^(?:\d+\.?\s*)?discussion\s*$",
        r"^(?:\d+\.?\s*)?discussion\s+and\s+(?:conclusions?|results?)\s*$",
        r"^(?:\d+\.?\s*)?analysis\s*$",
        r"^(?:\d+\.?\s*)?limitations?\s*$",
        r"^(?:\d+\.?\s*)?future\s+work\s*$",
        r"^(?:\d+\.?\s*)?limitations?\s+and\s+future\s+work\s*$",
    ],
    "conclusion": [
        r"^(?:\d+\.?\s*)?conclusions?\s*$",
        r"^(?:\d+\.?\s*)?concluding\s+remarks?\s*$",
        r"^(?:\d+\.?\s*)?summary\s+and\s+conclusions?\s*$",
        r"^(?:\d+\.?\s*)?final\s+remarks?\s*$",
        r"^(?:\d+\.?\s*)?summary\s*$",
    ],
    "acknowledgments": [
        r"^acknowledgm?ents?\s*$",
        r"^acknowledgements?\s*$",
    ],
    "references": [
        r"^references?\s*$",
        r"^bibliography\s*$",
        r"^literature\s+cited\s*$",
        r"^works?\s+cited\s*$",
    ],
    "supplementary": [
        r"^supplement(?:ary|al)?\s*(?:materials?|information)?\s*$",
        r"^supporting\s+information\s*$",
        r"^appendix\s*$",
        r"^appendices\s*$",
    ],
}


@dataclass
class Section:
    """A parsed section from a scientific paper."""
    
    name: str
    title: str
    content: str
    start_pos: int
    end_pos: int
    subsections: list["Section"] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []
    
    @property
    def word_count(self) -> int:
        """Word count of this section."""
        return len(self.content.split())


class SectionParser:
    """
    Parser for extracting sections from scientific paper text.
    
    Identifies standard scientific paper sections and extracts their content.
    
    Example:
        >>> parser = SectionParser()
        >>> sections = parser.parse(paper_text)
        >>> print(sections.get("abstract"))
        >>> print(sections.get("methods"))
    """
    
    def __init__(
        self,
        custom_patterns: dict[str, list[str]] | None = None,
        min_section_length: int = 50,
    ):
        """
        Initialize the section parser.
        
        Args:
            custom_patterns: Additional section patterns to recognize
            min_section_length: Minimum characters for a valid section
        """
        self.patterns = SECTION_PATTERNS.copy()
        if custom_patterns:
            for key, patterns in custom_patterns.items():
                if key in self.patterns:
                    self.patterns[key].extend(patterns)
                else:
                    self.patterns[key] = patterns
        
        self.min_section_length = min_section_length
        
        # Compile patterns
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for section_name, patterns in self.patterns.items():
            self._compiled_patterns[section_name] = [
                re.compile(p, re.IGNORECASE | re.MULTILINE)
                for p in patterns
            ]
    
    def parse(self, text: str) -> dict[str, str]:
        """
        Parse text into sections.
        
        Args:
            text: Full text of the paper
            
        Returns:
            Dictionary mapping section names to their content
        """
        sections: dict[str, str] = {}
        
        # Find all section headers and their positions
        section_matches: list[tuple[int, str, str]] = []  # (position, section_name, header_text)
        
        lines = text.split("\n")
        current_pos = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Check if this line is a section header
            for section_name, patterns in self._compiled_patterns.items():
                for pattern in patterns:
                    if pattern.match(stripped):
                        section_matches.append((current_pos, section_name, stripped))
                        break
                else:
                    continue
                break
            
            current_pos += len(line) + 1  # +1 for newline
        
        # Extract content between section headers
        for i, (pos, name, header) in enumerate(section_matches):
            # Find end position (start of next section or end of text)
            if i + 1 < len(section_matches):
                end_pos = section_matches[i + 1][0]
            else:
                end_pos = len(text)
            
            # Extract content (skip the header line itself)
            header_end = pos + len(header) + 1
            content = text[header_end:end_pos].strip()
            
            # Only add if content is substantial
            if len(content) >= self.min_section_length:
                # If section already exists, append (handles split sections)
                if name in sections:
                    sections[name] += "\n\n" + content
                else:
                    sections[name] = content
        
        # Try to extract abstract if not found via headers
        if "abstract" not in sections:
            abstract = self._extract_abstract(text)
            if abstract:
                sections["abstract"] = abstract
        
        return sections
    
    def parse_detailed(self, text: str) -> list[Section]:
        """
        Parse text into detailed Section objects with positions.
        
        Args:
            text: Full text of the paper
            
        Returns:
            List of Section objects
        """
        sections: list[Section] = []
        
        # Find all section headers
        section_matches: list[tuple[int, str, str]] = []
        lines = text.split("\n")
        current_pos = 0
        
        for line in lines:
            stripped = line.strip()
            
            for section_name, patterns in self._compiled_patterns.items():
                for pattern in patterns:
                    if pattern.match(stripped):
                        section_matches.append((current_pos, section_name, stripped))
                        break
                else:
                    continue
                break
            
            current_pos += len(line) + 1
        
        # Create Section objects
        for i, (pos, name, header) in enumerate(section_matches):
            if i + 1 < len(section_matches):
                end_pos = section_matches[i + 1][0]
            else:
                end_pos = len(text)
            
            header_end = pos + len(header) + 1
            content = text[header_end:end_pos].strip()
            
            if len(content) >= self.min_section_length:
                section = Section(
                    name=name,
                    title=header,
                    content=content,
                    start_pos=pos,
                    end_pos=end_pos,
                )
                sections.append(section)
        
        return sections
    
    def _extract_abstract(self, text: str) -> str | None:
        """
        Try to extract abstract using heuristics when no header is found.
        
        Some papers have the abstract without a clear header.
        """
        # Look for common abstract patterns
        patterns = [
            # Abstract followed by content
            r"(?:^|\n)abstract[:\s]*\n(.+?)(?=\n\s*\n|\n(?:1\.?\s*)?introduction)",
            # Summary followed by content  
            r"(?:^|\n)summary[:\s]*\n(.+?)(?=\n\s*\n|\n(?:1\.?\s*)?introduction)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                if len(abstract) >= self.min_section_length:
                    return abstract
        
        # Heuristic: First substantial paragraph after title might be abstract
        paragraphs = text.split("\n\n")
        for i, para in enumerate(paragraphs[:5]):  # Check first 5 paragraphs
            para = para.strip()
            # Abstract is usually 100-500 words
            word_count = len(para.split())
            if 50 <= word_count <= 600:
                # Check it's not a list of authors or affiliations
                if not re.match(r"^[\d\s,]+$", para):
                    if not re.search(r"@|university|department|institute", para, re.IGNORECASE):
                        return para
        
        return None
    
    def get_main_content(self, sections: dict[str, str]) -> str:
        """
        Get the main body content (excluding references and supplementary).
        
        Args:
            sections: Parsed sections dictionary
            
        Returns:
            Combined main content
        """
        main_sections = ["introduction", "methods", "results", "discussion", "conclusion"]
        
        content_parts = []
        for section in main_sections:
            if section in sections:
                content_parts.append(sections[section])
        
        return "\n\n".join(content_parts)
