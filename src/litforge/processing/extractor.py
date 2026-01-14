"""
PDF Text Extractor.

Enhanced text extraction from scientific PDFs with support for:
- Multi-column layouts
- Tables and figures
- Equations and special characters
- Header/footer removal
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """Represents extracted content from a single PDF page."""
    
    page_num: int
    text: str
    blocks: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    

@dataclass
class ExtractedDocument:
    """Represents the full extracted content from a PDF."""
    
    pages: list[ExtractedPage]
    full_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)
    
    @property
    def num_pages(self) -> int:
        """Number of pages in the document."""
        return len(self.pages)
    
    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.full_text.split())


class PDFExtractor:
    """
    Advanced PDF text extractor for scientific papers.
    
    Features:
    - Multi-column layout detection
    - Header/footer removal
    - Clean text output
    - Metadata extraction
    - Block-level extraction for structure preservation
    
    Example:
        >>> extractor = PDFExtractor()
        >>> doc = extractor.extract("paper.pdf")
        >>> print(doc.full_text)
        >>> print(doc.sections)
    """
    
    def __init__(
        self,
        remove_headers: bool = True,
        remove_footers: bool = True,
        detect_columns: bool = True,
        extract_images: bool = False,
    ):
        """
        Initialize the PDF extractor.
        
        Args:
            remove_headers: Remove page headers
            remove_footers: Remove page footers (including page numbers)
            detect_columns: Attempt to detect and properly order multi-column text
            extract_images: Extract image metadata (not image data)
        """
        self.remove_headers = remove_headers
        self.remove_footers = remove_footers
        self.detect_columns = detect_columns
        self.extract_images = extract_images
    
    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        """
        Extract text and structure from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ExtractedDocument with full text and structure
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        try:
            import pymupdf
        except ImportError:
            raise ImportError("PyMuPDF not installed. Install with: pip install pymupdf")
        
        doc = pymupdf.open(pdf_path)
        pages: list[ExtractedPage] = []
        all_text_parts: list[str] = []
        
        try:
            # Extract metadata
            metadata = self._extract_metadata(doc)
            
            for page_num, page in enumerate(doc):
                extracted_page = self._extract_page(page, page_num)
                pages.append(extracted_page)
                all_text_parts.append(extracted_page.text)
            
            # Combine and clean full text
            full_text = "\n\n".join(all_text_parts)
            full_text = self._clean_full_text(full_text)
            
            # Parse sections
            from litforge.processing.parser import SectionParser
            parser = SectionParser()
            sections = parser.parse(full_text)
            
            # Extract references
            from litforge.processing.references import ReferenceExtractor
            ref_extractor = ReferenceExtractor()
            references = ref_extractor.extract(full_text)
            
            return ExtractedDocument(
                pages=pages,
                full_text=full_text,
                metadata=metadata,
                sections=sections,
                references=references,
            )
            
        finally:
            doc.close()
    
    def extract_text_only(self, pdf_path: str | Path) -> str:
        """
        Simple text extraction without structure parsing.
        
        Faster than full extract() for when only raw text is needed.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        pdf_path = Path(pdf_path)
        
        try:
            import pymupdf
        except ImportError:
            raise ImportError("PyMuPDF not installed. Install with: pip install pymupdf")
        
        doc = pymupdf.open(pdf_path)
        text_parts: list[str] = []
        
        try:
            for page in doc:
                text = page.get_text("text")
                text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            return self._clean_full_text(full_text)
            
        finally:
            doc.close()
    
    def _extract_metadata(self, doc: Any) -> dict[str, Any]:
        """Extract PDF metadata."""
        metadata = {}
        
        try:
            pdf_metadata = doc.metadata
            if pdf_metadata:
                metadata["title"] = pdf_metadata.get("title", "")
                metadata["author"] = pdf_metadata.get("author", "")
                metadata["subject"] = pdf_metadata.get("subject", "")
                metadata["creator"] = pdf_metadata.get("creator", "")
                metadata["producer"] = pdf_metadata.get("producer", "")
                metadata["creation_date"] = pdf_metadata.get("creationDate", "")
                metadata["modification_date"] = pdf_metadata.get("modDate", "")
            
            metadata["page_count"] = doc.page_count
            
        except Exception as e:
            logger.debug(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _extract_page(self, page: Any, page_num: int) -> ExtractedPage:
        """Extract content from a single page."""
        # Get text blocks for structure
        blocks = []
        try:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    block_text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "")
                        block_text += "\n"
                    
                    blocks.append({
                        "type": "text",
                        "bbox": block.get("bbox"),
                        "text": block_text.strip(),
                    })
                elif block.get("type") == 1:  # Image block
                    if self.extract_images:
                        blocks.append({
                            "type": "image",
                            "bbox": block.get("bbox"),
                        })
        except Exception as e:
            logger.debug(f"Error extracting blocks: {e}")
        
        # Sort blocks by position for proper reading order
        if self.detect_columns:
            blocks = self._sort_blocks_by_columns(blocks, page)
        
        # Get plain text
        if blocks:
            text = "\n".join(b["text"] for b in blocks if b.get("type") == "text" and b.get("text"))
        else:
            text = page.get_text("text")
        
        # Remove headers/footers
        if self.remove_headers or self.remove_footers:
            text = self._remove_headers_footers(text, page, page_num)
        
        # Extract images metadata
        images = []
        if self.extract_images:
            images = [b for b in blocks if b.get("type") == "image"]
        
        # Try to detect tables
        tables = self._detect_tables(text)
        
        return ExtractedPage(
            page_num=page_num,
            text=text,
            blocks=blocks,
            images=images,
            tables=tables,
        )
    
    def _sort_blocks_by_columns(self, blocks: list[dict], page: Any) -> list[dict]:
        """Sort text blocks to handle multi-column layouts."""
        if not blocks:
            return blocks
        
        # Get page dimensions
        page_width = page.rect.width
        mid_x = page_width / 2
        
        # Separate blocks into left and right columns
        left_blocks = []
        right_blocks = []
        full_width_blocks = []
        
        for block in blocks:
            if block.get("type") != "text":
                continue
            
            bbox = block.get("bbox", [0, 0, 0, 0])
            block_left = bbox[0]
            block_right = bbox[2]
            block_width = block_right - block_left
            
            # Check if block spans most of the page (full width)
            if block_width > page_width * 0.7:
                full_width_blocks.append(block)
            elif block_right < mid_x + 50:  # Left column
                left_blocks.append(block)
            else:  # Right column
                right_blocks.append(block)
        
        # Sort each column by vertical position
        left_blocks.sort(key=lambda b: b.get("bbox", [0, 0, 0, 0])[1])
        right_blocks.sort(key=lambda b: b.get("bbox", [0, 0, 0, 0])[1])
        full_width_blocks.sort(key=lambda b: b.get("bbox", [0, 0, 0, 0])[1])
        
        # Combine: full-width header blocks, then left column, then right column
        # This is a simplification - real papers may interleave columns
        result = []
        
        # Add full-width blocks that appear at the top (title, abstract)
        for block in full_width_blocks:
            if block.get("bbox", [0, 0, 0, 0])[1] < 200:  # Top of page
                result.append(block)
        
        # Add left column, then right column
        result.extend(left_blocks)
        result.extend(right_blocks)
        
        # Add remaining full-width blocks
        for block in full_width_blocks:
            if block.get("bbox", [0, 0, 0, 0])[1] >= 200:
                result.append(block)
        
        return result
    
    def _remove_headers_footers(self, text: str, page: Any, page_num: int) -> str:
        """Remove common headers and footers from page text."""
        lines = text.split("\n")
        
        if not lines:
            return text
        
        # Remove page numbers (common footer patterns)
        if self.remove_footers:
            # Check last few lines for page numbers
            cleaned_lines = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Skip if it's just a number (page number)
                if re.match(r"^\d+$", stripped):
                    continue
                
                # Skip common footer patterns
                if re.match(r"^Page \d+", stripped, re.IGNORECASE):
                    continue
                
                if re.match(r"^\d+\s+of\s+\d+$", stripped, re.IGNORECASE):
                    continue
                
                cleaned_lines.append(line)
            
            lines = cleaned_lines
        
        # Remove headers (first page usually has title, subsequent pages may have running headers)
        if self.remove_headers and page_num > 0:
            # Skip very short first lines that might be headers
            while lines and len(lines[0].strip()) < 50:
                first_line = lines[0].strip()
                # Keep if it looks like content
                if len(first_line) > 20 or first_line.endswith("."):
                    break
                lines.pop(0)
        
        return "\n".join(lines)
    
    def _detect_tables(self, text: str) -> list[str]:
        """Detect and extract table-like content."""
        tables = []
        
        # Simple heuristic: look for lines with multiple tab/space separated values
        lines = text.split("\n")
        table_lines = []
        
        for line in lines:
            # Check if line looks like a table row
            # (multiple columns separated by whitespace)
            parts = re.split(r"\s{2,}|\t", line.strip())
            if len(parts) >= 3 and all(len(p) < 50 for p in parts):
                table_lines.append(line)
            else:
                if len(table_lines) >= 3:
                    # We found a table
                    tables.append("\n".join(table_lines))
                table_lines = []
        
        if len(table_lines) >= 3:
            tables.append("\n".join(table_lines))
        
        return tables
    
    def _clean_full_text(self, text: str) -> str:
        """Clean and normalize the full extracted text."""
        # Fix common OCR/extraction issues
        
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" +", " ", text)
        
        # Fix hyphenation at line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        
        # Remove lone page numbers
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
        
        # Fix common ligature issues
        ligatures = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
        }
        for lig, replacement in ligatures.items():
            text = text.replace(lig, replacement)
        
        # Remove null bytes and other control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        
        return text.strip()
