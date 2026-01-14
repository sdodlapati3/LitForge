"""
Configuration management for LitForge.

Supports configuration via:
- Environment variables (LITFORGE_*)
- YAML configuration file (litforge.yaml)
- Programmatic overrides
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SourceConfig(BaseSettings):
    """Configuration for data sources."""
    
    model_config = SettingsConfigDict(env_prefix="LITFORGE_")
    
    default_sources: list[str] = Field(
        default=["openalex", "semantic_scholar"],
        description="Default sources to search"
    )
    
    # OpenAlex
    openalex_email: str | None = Field(
        default=None,
        description="Email for OpenAlex polite pool (higher rate limits)"
    )
    
    # Semantic Scholar
    semantic_scholar_api_key: str | None = Field(
        default=None,
        description="API key for Semantic Scholar (optional, for higher limits)"
    )
    
    # PubMed
    pubmed_email: str | None = Field(
        default=None,
        description="Email for PubMed/NCBI API"
    )
    pubmed_api_key: str | None = Field(
        default=None,
        description="API key for PubMed (optional)"
    )


class EmbeddingConfig(BaseSettings):
    """Configuration for embeddings."""
    
    model_config = SettingsConfigDict(env_prefix="LITFORGE_")
    
    provider: Literal["openai", "local", "custom"] = Field(
        default="openai",
        description="Embedding provider"
    )
    
    model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name"
    )
    
    # OpenAI
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key"
    )
    
    # Local (sentence-transformers)
    local_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Local sentence-transformers model"
    )
    
    # Dimensions
    dimensions: int | None = Field(
        default=None,
        description="Embedding dimensions (auto-detected if None)"
    )


class VectorStoreConfig(BaseSettings):
    """Configuration for vector store."""
    
    model_config = SettingsConfigDict(env_prefix="LITFORGE_")
    
    provider: Literal["chroma", "qdrant", "faiss", "memory"] = Field(
        default="chroma",
        description="Vector store provider"
    )
    
    persist_dir: Path = Field(
        default=Path.home() / ".litforge" / "vectors",
        description="Directory for persistent storage"
    )
    
    collection_name: str = Field(
        default="litforge_papers",
        description="Collection/index name"
    )
    
    # Qdrant
    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant server URL"
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key"
    )


class LLMConfig(BaseSettings):
    """Configuration for LLM providers."""
    
    model_config = SettingsConfigDict(env_prefix="LITFORGE_")
    
    provider: Literal["openai", "anthropic", "ollama", "none"] = Field(
        default="openai",
        description="LLM provider"
    )
    
    model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name"
    )
    
    # OpenAI
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key"
    )
    
    # Anthropic
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key"
    )
    
    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    
    # Generation settings
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1)


class CacheConfig(BaseSettings):
    """Configuration for caching."""
    
    model_config = SettingsConfigDict(env_prefix="LITFORGE_")
    
    enabled: bool = Field(default=True, description="Enable caching")
    
    directory: Path = Field(
        default=Path.home() / ".litforge" / "cache",
        description="Cache directory"
    )
    
    ttl: int = Field(
        default=86400,  # 24 hours
        description="Cache TTL in seconds"
    )
    
    max_size_gb: float = Field(
        default=5.0,
        description="Maximum cache size in GB"
    )


class LitForgeConfig(BaseSettings):
    """Main LitForge configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="LITFORGE_",
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    # Sub-configurations
    sources: SourceConfig = Field(default_factory=SourceConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
    # Global settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    data_dir: Path = Field(
        default=Path.home() / ".litforge",
        description="Base data directory"
    )
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "LitForgeConfig":
        """Load configuration from YAML file."""
        import yaml
        
        path = Path(path)
        if not path.exists():
            return cls()
        
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        
        return cls(**data)
    
    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "LitForgeConfig":
        """
        Load configuration with priority:
        1. Explicit config_path
        2. LITFORGE_CONFIG env var
        3. ./litforge.yaml
        4. ~/.litforge/config.yaml
        5. Defaults
        """
        paths_to_try = []
        
        if config_path:
            paths_to_try.append(Path(config_path))
        
        if env_path := os.getenv("LITFORGE_CONFIG"):
            paths_to_try.append(Path(env_path))
        
        paths_to_try.extend([
            Path.cwd() / "litforge.yaml",
            Path.cwd() / "litforge.yml",
            Path.home() / ".litforge" / "config.yaml",
            Path.home() / ".litforge" / "config.yml",
        ])
        
        for path in paths_to_try:
            if path.exists():
                return cls.from_yaml(path)
        
        return cls()


# Global config instance (lazy-loaded)
_config: LitForgeConfig | None = None


def get_config() -> LitForgeConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = LitForgeConfig.load()
    return _config


def set_config(config: LitForgeConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to force reload."""
    global _config
    _config = None
