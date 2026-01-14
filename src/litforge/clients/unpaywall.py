"""
Unpaywall API client.

Find open access versions of papers.
"""

from __future__ import annotations

import logging
from typing import Any

from litforge.clients.base import BaseClient

logger = logging.getLogger(__name__)


class UnpaywallClient(BaseClient):
    """
    Client for the Unpaywall API.
    
    Unpaywall finds legal open access versions of papers.
    Free, requires email.
    
    API Docs: https://unpaywall.org/products/api
    """
    
    def __init__(self, email: str | None = None):
        """
        Initialize the Unpaywall client.
        
        Args:
            email: Email (required for API access)
        """
        super().__init__(
            base_url="https://api.unpaywall.org/v2",
            rate_limit=10.0,
        )
        self.email = email or "litforge@example.com"
    
    def get_open_access(self, doi: str) -> dict[str, Any] | None:
        """
        Find open access version of a paper.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Dict with OA info (pdf_url, is_oa, etc.) or None
        """
        # Normalize DOI
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        
        try:
            response = self._get(
                f"/{doi}",
                params={"email": self.email},
            )
            
            if not response.get("is_oa"):
                return None
            
            # Find best OA location
            best_oa = response.get("best_oa_location", {})
            
            return {
                "is_oa": True,
                "oa_status": response.get("oa_status"),
                "pdf_url": best_oa.get("url_for_pdf"),
                "landing_page": best_oa.get("url"),
                "host_type": best_oa.get("host_type"),
                "license": best_oa.get("license"),
                "version": best_oa.get("version"),
                "all_locations": [
                    {
                        "url": loc.get("url"),
                        "pdf_url": loc.get("url_for_pdf"),
                        "host_type": loc.get("host_type"),
                    }
                    for loc in response.get("oa_locations", [])
                ],
            }
            
        except Exception as e:
            logger.debug(f"Unpaywall lookup failed for {doi}: {e}")
            return None
