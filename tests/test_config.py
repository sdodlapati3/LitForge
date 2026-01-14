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
        
        assert config.embeddings.provider == "openai"
        assert config.vector_store.provider == "chroma"
        assert config.llm.provider == "openai"
    
    def test_config_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        from litforge.config import LitForgeConfig
        
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
embeddings:
  provider: local
  local_model: all-MiniLM-L6-v2

vector_store:
  provider: chroma
  persist_dir: /tmp/litforge_test
""")
        
        config = LitForgeConfig.from_yaml(config_path)
        
        assert config.embeddings.provider == "local"
        assert config.embeddings.local_model == "all-MiniLM-L6-v2"
    
    def test_global_config(self):
        """Test global config management."""
        from litforge.config import get_config, set_config, LitForgeConfig
        
        # Get default config
        config1 = get_config()
        assert config1 is not None
        
        # Set custom config
        custom = LitForgeConfig()
        set_config(custom)
        
        config2 = get_config()
        assert config2.llm.model == custom.llm.model
