"""
PDF Processing Module.

Provides advanced text extraction, section parsing, and reference extraction
from scientific PDFs.
"""

from litforge.processing.extractor import PDFExtractor
from litforge.processing.parser import SectionParser
from litforge.processing.references import ReferenceExtractor

__all__ = [
    "PDFExtractor",
    "SectionParser", 
    "ReferenceExtractor",
]
