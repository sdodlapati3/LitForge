"""
LLM Router - Smart provider selection with free-tier cascade.

Tries free providers first (Cerebras, Groq, Google, GitHub Models),
falls back to paid (OpenAI) only when necessary.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from litforge.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class CerebrasProvider(BaseLLM):
    """Cerebras Cloud - 14,400 req/day FREE (most generous)."""
    
    def __init__(self, api_key: str | None = None, model: str = "llama-3.3-70b"):
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        self.model = model
        self._client = None
        self._available = None
        
    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                from cerebras.cloud.sdk import Cerebras
                self._available = True
            except ImportError:
                self._available = False
        return self._available
        
    @property
    def client(self):
        if self._client is None:
            from cerebras.cloud.sdk import Cerebras
            self._client = Cerebras(api_key=self.api_key)
        return self._client
    
    def generate(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 1000) -> str:
        if not self.available:
            raise ImportError("cerebras package not installed")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, max_tokens=max_tokens)
    
    def chat(self, messages: list[dict], max_tokens: int = 1000) -> str:
        if not self.available:
            raise ImportError("cerebras package not installed")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class GroqProvider(BaseLLM):
    """Groq Cloud - 1,000+ req/day FREE (fastest inference)."""
    
    def __init__(self, api_key: str | None = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self._client = None
        self._available = None
    
    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                from groq import Groq
                self._available = True
            except ImportError:
                self._available = False
        return self._available
        
    @property
    def client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client
    
    def generate(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 1000) -> str:
        if not self.available:
            raise ImportError("groq package not installed")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, max_tokens=max_tokens)
    
    def chat(self, messages: list[dict], max_tokens: int = 1000) -> str:
        if not self.available:
            raise ImportError("groq package not installed")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class GoogleProvider(BaseLLM):
    """Google Gemini - 250 req/day FREE."""
    
    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = model
        self._client = None
        self._available = None
    
    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import google.generativeai
                self._available = True
            except ImportError:
                self._available = False
        return self._available
        
    @property
    def client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        return self._client
    
    def generate(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 1000) -> str:
        if not self.available:
            raise ImportError("google-generativeai package not installed")
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self.client.generate_content(full_prompt)
        return response.text
    
    def chat(self, messages: list[dict], max_tokens: int = 1000) -> str:
        if not self.available:
            raise ImportError("google-generativeai package not installed")
        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            else:
                prompt_parts.append(content)
        return self.generate("\n\n".join(prompt_parts))


class GitHubModelsProvider(BaseLLM):
    """GitHub Models API - FREE with GitHub Copilot subscription."""
    
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("GITHUB_TOKEN")
        self.model = model
        self._client = None
        self._available = None
    
    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                from openai import OpenAI
                self._available = True
            except ImportError:
                self._available = False
        return self._available
        
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://models.inference.ai.azure.com",
                api_key=self.api_key
            )
        return self._client
    
    def generate(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 1000) -> str:
        if not self.available:
            raise ImportError("openai package not installed")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, max_tokens=max_tokens)
    
    def chat(self, messages: list[dict], max_tokens: int = 1000) -> str:
        if not self.available:
            raise ImportError("openai package not installed")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class LLMRouter(BaseLLM):
    """
    Smart LLM router with free-tier cascade.
    
    Priority order (all free first, then paid):
    1. Cerebras (14,400 req/day) - Most generous
    2. Groq (1,000+ req/day) - Fastest
    3. Google Gemini (250 req/day)
    4. GitHub Models (with Copilot) 
    5. OpenAI (paid fallback)
    """
    
    PROVIDERS = [
        ("cerebras", CerebrasProvider, "CEREBRAS_API_KEY"),
        ("groq", GroqProvider, "GROQ_API_KEY"),
        ("google", GoogleProvider, "GOOGLE_API_KEY"),
        ("github", GitHubModelsProvider, "GITHUB_TOKEN"),
        ("openai", None, "OPENAI_API_KEY"),  # Uses existing OpenAIProvider
    ]
    
    def __init__(self, preferred_provider: str | None = None):
        """
        Initialize router.
        
        Args:
            preferred_provider: Force a specific provider (e.g., "groq", "cerebras")
        """
        self.preferred_provider = preferred_provider
        self._active_provider: BaseLLM | None = None
        self._active_name: str | None = None
        self._failed_providers: set[str] = set()  # Track failed providers
    
    def _get_provider(self) -> tuple[str, BaseLLM]:
        """Get the best available provider."""
        if self._active_provider:
            return self._active_name, self._active_provider
        
        # If preferred, try that first
        if self.preferred_provider and self.preferred_provider not in self._failed_providers:
            for name, cls, env_var in self.PROVIDERS:
                if name == self.preferred_provider and os.environ.get(env_var):
                    if name == "openai":
                        from litforge.llm.openai import OpenAIProvider
                        self._active_provider = OpenAIProvider()
                    else:
                        self._active_provider = cls()
                    self._active_name = name
                    logger.info(f"Using preferred LLM provider: {name}")
                    return name, self._active_provider
        
        # Cascade through free providers, checking package availability
        for name, cls, env_var in self.PROVIDERS:
            if name in self._failed_providers:
                continue
            if os.environ.get(env_var):
                try:
                    if name == "openai":
                        from litforge.llm.openai import OpenAIProvider
                        provider = OpenAIProvider()
                    else:
                        provider = cls()
                        # Check if the package is actually installed
                        if hasattr(provider, 'available') and not provider.available:
                            logger.debug(f"Skipping {name}: package not installed")
                            continue
                    
                    self._active_provider = provider
                    self._active_name = name
                    logger.info(f"Using LLM provider: {name}")
                    return name, self._active_provider
                except Exception as e:
                    logger.warning(f"Failed to init {name}: {e}")
                    continue
        
        raise RuntimeError("No LLM provider available. Set at least one API key and install the corresponding package.")
    
    def generate(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 1000) -> str:
        """Generate with automatic provider selection and fallback."""
        max_retries = len(self.PROVIDERS)
        last_error = None
        
        for _ in range(max_retries):
            try:
                name, provider = self._get_provider()
                return provider.generate(prompt, system_prompt, max_tokens)
            except Exception as e:
                last_error = e
                if self._active_name:
                    logger.warning(f"{self._active_name} failed: {e}, trying next provider...")
                    self._failed_providers.add(self._active_name)
                self._active_provider = None
                self._active_name = None
        
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
    
    def chat(self, messages: list[dict], max_tokens: int = 1000) -> str:
        """Chat with automatic provider selection and fallback."""
        max_retries = len(self.PROVIDERS)
        last_error = None
        
        for _ in range(max_retries):
            try:
                name, provider = self._get_provider()
                return provider.chat(messages, max_tokens)
            except Exception as e:
                last_error = e
                if self._active_name:
                    logger.warning(f"{self._active_name} failed: {e}, trying next provider...")
                    self._failed_providers.add(self._active_name)
                self._active_provider = None
                self._active_name = None
        
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
    
    @property
    def provider_name(self) -> str:
        """Get the name of the active provider."""
        if self._active_name:
            return self._active_name
        name, _ = self._get_provider()
        return name


def get_llm(provider: str | None = None) -> BaseLLM:
    """
    Get the best available LLM.
    
    Args:
        provider: Optional specific provider name
        
    Returns:
        LLM instance (router if no preference, specific if requested)
    """
    return LLMRouter(preferred_provider=provider)
