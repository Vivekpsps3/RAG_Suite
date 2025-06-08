"""
CSVProcessor: Component for processing CSV files into document chunks for RAG.
"""
import pandas as pd
import uuid
from pathlib import Path
from typing import List, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import configuration
from .config import EMBEDDING_MODEL_PATH

# Set up logging
logger = logging.getLogger(__name__)

class CSVProcessor:
    """Processor for converting CSV data into textified documents with embeddings."""
    
    def __init__(
        self, 
        textifier_delimiter: str = " | ",
        chunk_size: int = 512,
        embedding_model_name: str = None
    ):
        """Initialize the CSV processor."""
        self.textifier_delimiter = textifier_delimiter
        self.chunk_size = chunk_size
        
        # Use config path if no model name is provided
        if embedding_model_name is None:
            embedding_model_name = EMBEDDING_MODEL_PATH
        
        # Initialize embedding model
        try:
            logger.info(f"Loading embedding model from {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=100,
            length_function=lambda text: len(text.split())
        )
    
    def process_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a CSV file into document chunks with embeddings."""
        documents = []
        
        try:
            # Load and process the CSV file
            df = pd.read_csv(file_path)
            
            # Process each row
            for row_idx, row in df.iterrows():
                # Convert row to text and split into chunks if needed
                textified_row = self._textify_row(row)
                text_chunks = self.text_splitter.split_text(textified_row)
                
                # Create documents for each chunk
                for chunk_idx, chunk in enumerate(text_chunks):
                    doc_id = str(uuid.uuid4())
                    metadata = {
                        "source": file_path.name,
                        "row_index": int(row_idx),
                        "chunk_index": chunk_idx
                    }
                    
                    # Generate embedding
                    embedding = self._generate_embedding(chunk)
                    
                    # Only add document if embedding generation was successful
                    if embedding is not None:
                        # Create document
                        document = {
                            "id": doc_id,
                            "text": chunk,
                            "metadata": metadata,
                            "embedding": embedding
                        }
                        documents.append(document)
                    else:
                        logger.warning(f"Skipping document {doc_id} due to embedding generation failure")
            
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {str(e)}")
        
        return documents
    
    def _textify_row(self, row: pd.Series) -> str:
        """Convert a row to textified format."""
        row_items = []
        
        for col_name, value in row.items():
            if pd.notna(value):  # Skip NA values
                row_items.append(f"{col_name}: {value}")
        
        return self.textifier_delimiter.join(row_items)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text chunk."""
        try:
            # Ensure text is not empty
            if not text or len(text.strip()) == 0:
                logger.warning("Received empty text for embedding generation")
                return None
                
            # Generate embedding
            logger.debug(f"Generating embedding for text: {text[:50]}...")
            embedding = self.embedding_model.encode(text)
            
            # Convert to Python list and verify dimensions
            embedding_list = embedding.tolist()
            if len(embedding_list) == 0:
                logger.error("Generated embedding has zero dimensions")
                return None
                
            logger.debug(f"Generated embedding with {len(embedding_list)} dimensions")
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            logger.exception("Embedding generation failed")
            return None  # Return None instead of fake embeddings to surface the error
