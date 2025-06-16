"""
CSVProcessor: Component for processing CSV files into document chunks for RAG.
"""
import pandas as pd
import uuid
from pathlib import Path
from typing import List, Dict, Any
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import configuration and model engine
from .model_engine import model_engine
from . import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Set up logging
logger = logging.getLogger(__name__)

class CSVProcessor:
    """Processor for converting CSV data into textified documents with embeddings."""
    
    def __init__(
        self, 
        textifier_delimiter: str = " | ",
        chunk_size: int = None,
        chunk_overlap: int = None,
        embedding_model_name: str = None
    ):
        """Initialize the CSV processor."""
        self.textifier_delimiter = textifier_delimiter
        self.chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP
        
        # Initialize embedding model in model engine if needed
        try:
            model_engine.get_embedding_model(embedding_model_name)
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=lambda text: len(text.split())
        )
    
    def process_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a CSV file into document chunks with embeddings."""
        documents = []
        all_chunks = []
        chunk_metadata = []
        
        try:
            # Load and process the CSV file
            df = pd.read_csv(file_path)
            
            # First pass: Textify all rows and collect chunks with metadata
            for row_idx, row in df.iterrows():
                # Convert row to text and split into chunks if needed
                textified_row = self._textify_row(row)
                text_chunks = self.text_splitter.split_text(textified_row)
                
                # Collect all chunks and their metadata
                for chunk_idx, chunk in enumerate(text_chunks):
                    doc_id = str(uuid.uuid4())
                    metadata = {
                        "source": file_path.name,
                        "row_index": int(row_idx),
                        "chunk_index": chunk_idx
                    }
                    all_chunks.append(chunk)
                    chunk_metadata.append((doc_id, metadata))
            
            # Second pass: Batch generate embeddings
            if all_chunks:
                logger.info(f"Batch generating embeddings for {len(all_chunks)} chunks")
                try:
                    # Generate embeddings in batch
                    embeddings = model_engine.generate_embeddings(all_chunks)

                    # Create documents with their corresponding embeddings
                    for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
                        if embedding is not None:
                            doc_id, metadata = chunk_metadata[i]
                            document = {
                                "id": doc_id,
                                "text": chunk,
                                "metadata": metadata,
                                "embedding": embedding
                            }
                            documents.append(document)
                        else:
                            doc_id = chunk_metadata[i][0]
                            logger.warning(f"Skipping document {doc_id} due to embedding generation failure")
                except Exception as e:
                    logger.error(f"Batch embedding generation failed: {str(e)}")
            
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
