"""
Base client class for API clients.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class BaseClient(ABC):
    """Base class for API clients with rate limiting and retries."""
    
    def __init__(
        self,
        base_url: str,
        rate_limit: float = 1.0,  # requests per second
        timeout: float = 30.0,
    ):
        """
        Initialize the base client.
        
        Args:
            base_url: Base URL for the API
            rate_limit: Maximum requests per second
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request_time = 0.0
        self._client: httpx.Client | None = None
    
    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client
    
    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limit."""
        if self.rate_limit <= 0:
            return
        
        min_interval = 1.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self._last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Make a GET request with rate limiting and retries.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            
        Returns:
            JSON response
        """
        self._rate_limit_wait()
        
        response = self.client.get(
            endpoint,
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        
        return response.json()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _post(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Make a POST request with rate limiting and retries.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            headers: Additional headers
            
        Returns:
            JSON response
        """
        self._rate_limit_wait()
        
        response = self.client.post(
            endpoint,
            data=data,
            json=json_data,
            headers=headers,
        )
        response.raise_for_status()
        
        return response.json()
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None
    
    def __enter__(self) -> "BaseClient":
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.close()
