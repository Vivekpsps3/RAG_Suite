"""
RobustRAG package initialization.

This module imports and exposes the main components of the RobustRAG system
and configures global settings and environment variables.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Default batch size/chunk size (4096 as requested)
DEFAULT_CHUNK_SIZE = 4096
DEFAULT_CHUNK_OVERLAP = 200

# LLM Parameters
LLM_MAX_TOKENS = 32000
LLM_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 500

# Default number of top results to return from vector database
NUM_CTX_RESULTS = 5

# Vector store configuration
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vectors")

# Embedding model configuration
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/home/vivek/Files/Model_Files/all-MiniLM-L6-v2")

# Reranker model configuration
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "/home/vivek/Files/Model_Files/ms-marco-MiniLM-L-6-v2")

# LLM model configuration
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "/home/vivek/Files/Model_Files/Dolphin3.0-Llama3.1-8B")

# Document paths
CSV_DOCUMENTS_DIR = Path(os.getenv("CSV_DOCUMENTS_DIR", "documents/csv"))
JSON_DOCUMENTS_DIR = Path(os.getenv("JSON_DOCUMENTS_DIR", "documents/json"))

# System prompts
QUERY_PROMPT = """
Using ONLY the information provided in the context below, answer the following question as accurately and completely as possible. 
If the context does not contain enough information to answer, respond with: "I cannot answer this question with the provided information."

- Use all relevant details from the context.
- Do not make assumptions or use outside knowledge.
- If the answer is ambiguous or incomplete based on the context, clearly state so.
- No matter how complex the question, if the context provides relevant information, use it to construct your answer.
- Whether you answer or not, always provide a small explanation of your reasoning.

Context:
{context}

Question: {query}

Answer:
"""

# Additional Context Prompt
ADDITIONAL_CONTEXT_PROMPT = """
Given the following context: {context}
And the question: {query}

Please generate a new query that is more specific and focused to help find the answer in the context.
Completely reformulate the question to extract NEW information from the context.

Guidelines:
- Use keywords EXACTLY as they appear in the context
- Use synonyms and terminology as they appear in the context (e.g., if the context uses "Metallic Alloy" instead of "Alloy of Metals", use "Metallic Alloy")
- Focus on relevant parts and filter out irrelevant information
- Make your query precise and targeted like an effective search query

DO NOT ANSWER THE ORIGINAL QUESTION - ONLY PROVIDE THE NEW QUERY.

New Query:
"""

# Validate paths
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
    
    # Ensure vector store directory exists
    Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)

# Run validation on module import
validate_paths()

# Import components after initializing config
from .csv_processor import CSVProcessor
from .vector_store_manager import VectorStoreManager
from .application_engine import ApplicationEngine
from .model_engine import ModelEngine, model_engine

__all__ = [
    'CSVProcessor',
    'VectorStoreManager',
    'QueryAgent',
    'ApplicationEngine',
    'ModelEngine',
    'model_engine'
]
