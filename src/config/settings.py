"""Configuration settings for the RAG application."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    huggingface_api_token: Optional[str] = Field(None, env="HUGGINGFACE_API_TOKEN")
    langsmith_api_key: Optional[str] = Field(None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field("langchain-doc-qa", env="LANGSMITH_PROJECT")
    
    # Application Configuration
    app_env: str = Field("development", env="APP_ENV")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_upload_size: int = Field(50, env="MAX_UPLOAD_SIZE")  # MB
    
    # Text Processing
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # Embedding Configuration
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", 
        env="EMBEDDING_MODEL"
    )
    embedding_device: str = Field("cpu", env="EMBEDDING_DEVICE")
    
    # Vector Store Configuration
    vector_store_type: str = Field("faiss", env="VECTOR_STORE_TYPE")
    chroma_host: str = Field("localhost", env="CHROMA_HOST")
    chroma_port: int = Field(8000, env="CHROMA_PORT")
    
    # LLM Configuration
    default_llm: str = Field("openai", env="DEFAULT_LLM")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")
    anthropic_model: str = Field("claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    temperature: float = Field(0.1, env="TEMPERATURE")
    max_tokens: int = Field(2000, env="MAX_TOKENS")
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    vector_store_path: Path = data_dir / "vector_store"
    upload_dir: Path = data_dir / "uploads"
    logs_dir: Path = base_dir / "logs"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.upload_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"


# Global settings instance
settings = Settings()