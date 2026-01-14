"""Tests for configuration."""

import os
import pytest
from pathlib import Path


class TestConfig:
    """Tests for configuration system."""
    
    def test_default_config(self):
        """Test default configuration."""
        from litforge.config import LitForgeConfig
        
        config = LitForgeConfig()
        
        assert config.embedding.provider == "openai"
        assert config.vector_store.backend == "chromadb"
        assert config.llm.provider == "openai"
    
    def test_config_from_dict(self):
        """Test creating config from dict."""
        from litforge.config import LitForgeConfig
        
        data = {
            "embedding": {
                "provider": "sentence_transformers",
                "model": "all-MiniLM-L6-v2",
            },
        }
        
        config = LitForgeConfig.from_dict(data)
        
        assert config.embedding.provider == "sentence_transformers"
        assert config.embedding.model == "all-MiniLM-L6-v2"
    
    def test_config_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        from litforge.config import load_config
        
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
embedding:
  provider: sentence_transformers
  model: all-MiniLM-L6-v2

vector_store:
  backend: chromadb
  path: /tmp/litforge_test
""")
        
        config = load_config(config_path)
        
        assert config.embedding.provider == "sentence_transformers"
        assert config.vector_store.path == "/tmp/litforge_test"
    
    def test_global_config(self):
        """Test global config management."""
        from litforge.config import get_config, set_config, LitForgeConfig
        
        # Get default config
        config1 = get_config()
        assert config1 is not None
        
        # Set custom config
        custom = LitForgeConfig()
        custom.llm.model = "gpt-3.5-turbo"
        set_config(custom)
        
        config2 = get_config()
        assert config2.llm.model == "gpt-3.5-turbo"
