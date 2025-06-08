"""
Configuration module for loading environment variables and settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Vector store configuration
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vectors")

# Embedding model configuration
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/home/vivek/Files/Model_Files/all-MiniLM-L6-v2")

# LLM model configuration
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "/home/vivek/Files/Model_Files/Dolphin3.0-Llama3.1-8B")

# Document paths
CSV_DOCUMENTS_DIR = Path(os.getenv("CSV_DOCUMENTS_DIR", "documents/csv"))
CSV_HEADERS_DIR = Path(os.getenv("CSV_HEADERS_DIR", "documents/headers"))

def validate_paths():
    """Validate that all necessary paths exist and are accessible."""
    # Check embedding model path
    if not os.path.exists(EMBEDDING_MODEL_PATH):
        logger.warning(f"Embedding model path not found: {EMBEDDING_MODEL_PATH}")
        
    # Check LLM model path
    if not os.path.exists(LLM_MODEL_PATH):
        logger.warning(f"LLM model path not found: {LLM_MODEL_PATH}")
    
    # Ensure document directories exist
    CSV_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_HEADERS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ensure vector store directory exists
    Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    
# Run validation on module import
validate_paths()
