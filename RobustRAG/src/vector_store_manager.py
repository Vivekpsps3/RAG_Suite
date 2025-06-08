"""
VectorStoreManager: Component for managing document vectors in ChromaDB.
"""
import chromadb
from chromadb.config import Settings
from chromadb.api import Collection
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manager for ChromaDB vector storage and retrieval."""
    
    def __init__(self, persist_directory: str = "vectors"):
        """Initialize the vector store manager."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Default collection is created lazily
            self.collection = None
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
    
    def create_or_load_collection(self, name: str = "mlearn") -> Collection:
        """Create or load a ChromaDB collection."""
        try:
            # Get or create collection with HNSW index
            collection = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Update default collection
            self.collection = collection
            return collection
            
        except Exception as e:
            logger.error(f"Error creating/loading collection {name}: {str(e)}")
            raise
    
    def add_documents(
        self, 
        documents: List[Dict[str, Any]], 
        collection_name: str = "mlearn"
    ) -> None:
        """Add documents to a collection."""
        try:
            # Ensure collection exists
            collection = self.create_or_load_collection(collection_name)
            
            # Extract components from documents
            ids = [doc["id"] for doc in documents]
            embeddings = [doc["embedding"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            texts = [doc["text"] for doc in documents]
            
            # Add documents in a single batch operation
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
    
    def retrieve(
        self, 
        query: str, 
        collection_name: str = "mlearn", 
        filter_dict: Optional[Dict[str, Any]] = None,
        n: int = 4
    ) -> List[Dict[str, Any]]:
        """Retrieve documents similar to a query."""
        try:
            # Ensure collection exists
            collection = self.create_or_load_collection(collection_name)
            
            # Query the collection
            results = collection.query(
                query_texts=[query],
                n_results=n,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            
            if results["documents"] and len(results["documents"]) > 0:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                
                for i in range(len(documents)):
                    result = {
                        "document": documents[i],
                        "metadata": metadatas[i],
                        "score": 1.0 - distances[i]  # Convert distance to similarity score
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving: {str(e)}")
            return []
