"""
OpenAI LLM provider.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from litforge.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLM):
    """
    OpenAI LLM provider.
    
    Uses GPT-4 by default for best quality.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name
            temperature: Sampling temperature
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self._client: Any = None
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )
    
    @property
    def client(self) -> Any:
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI not installed. Install with: pip install openai"
                )
        return self._client
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1000,
    ) -> str:
        """Generate text from a prompt."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, max_tokens=max_tokens)
    
    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
    ) -> str:
        """Generate response from chat messages."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content or ""
