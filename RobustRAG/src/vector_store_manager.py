"""
VectorStoreManager: Component for managing document vectors in ChromaDB.
"""
import chromadb
from chromadb.config import Settings
from chromadb.api import Collection
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer

# Import configuration
from .config import VECTOR_STORE_DIR, EMBEDDING_MODEL_PATH

# Set up logging
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manager for ChromaDB vector storage and retrieval."""
    
    def __init__(self, persist_directory: str = None, embedding_model_name: str = None):
        """Initialize the vector store manager."""
        try:
            # Use config values if not provided
            if persist_directory is None:
                persist_directory = VECTOR_STORE_DIR
            
            if embedding_model_name is None:
                embedding_model_name = EMBEDDING_MODEL_PATH
                
            # Initialize ChromaDB client
            logger.info(f"Initializing ChromaDB client at {persist_directory}")
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                )
            )
            
            # Default collection is created lazily
            self.collection = None
            
            # Initialize embedding model
            logger.info(f"Loading embedding model from {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            
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
        collection_name: str = "mlearn",
        batch_size: int = 500  # Safe batch size well below ChromaDB's limit
    ) -> None:
        """Add documents to a collection in batches."""
        try:
            # Skip if no documents
            if not documents:
                logger.warning("No documents provided to add_documents")
                return
                
            # Ensure collection exists
            collection = self.create_or_load_collection(collection_name)
            
            # Verify documents have all required components
            valid_documents = []
            for doc in documents:
                if not all(key in doc for key in ["id", "embedding", "metadata", "text"]):
                    logger.warning(f"Document {doc.get('id', 'unknown')} is missing required fields")
                    continue
                if doc["embedding"] is None:
                    logger.warning(f"Document {doc['id']} has None embedding")
                    continue
                valid_documents.append(doc)
                
            if not valid_documents:
                logger.error("No valid documents to add to collection")
                return
                
            total_docs = len(valid_documents)
            logger.info(f"Adding {total_docs} documents to collection {collection_name} in batches of {batch_size}")
            
            # Process documents in batches to avoid exceeding ChromaDB's batch size limit
            for i in range(0, total_docs, batch_size):
                batch_docs = valid_documents[i:i+batch_size]
                batch_count = len(batch_docs)
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1} with {batch_count} documents")
                
                # Extract components from current batch of documents
                ids = [doc["id"] for doc in batch_docs]
                embeddings = [doc["embedding"] for doc in batch_docs]
                metadatas = [doc["metadata"] for doc in batch_docs]
                texts = [doc["text"] for doc in batch_docs]
                
                # Log the first document's embedding dimensions for debugging (first batch only)
                if i == 0 and embeddings and len(embeddings) > 0:
                    logger.info(f"First document embedding dimensions: {len(embeddings[0])}")
                
                # Add documents in a batch operation
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=texts
                )
                
                logger.info(f"Successfully added batch of {batch_count} documents")
            
            # Verify the final count
            count = collection.count()
            logger.info(f"Collection now contains {count} documents (added {total_docs})")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            logger.exception("Document addition failed")
    
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
            
            # Check if collection has documents
            count = collection.count()
            if count == 0:
                logger.error(f"Collection '{collection_name}' is empty - no documents to retrieve from")
                return []
                
            logger.info(f"Retrieving from collection with {count} documents")
            
            # Generate embedding for the query using our model
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Query the collection with the embedding
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            
            if results["documents"] and len(results["documents"]) > 0:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                
                logger.info(f"Retrieved {len(documents)} documents with query: '{query}'")
                
                for i in range(len(documents)):
                    result = {
                        "document": documents[i],
                        "metadata": metadatas[i],
                        "score": 1.0 - distances[i]  # Convert distance to similarity score
                    }
                    formatted_results.append(result)
            else:
                logger.warning(f"No documents found for query: '{query}'")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving: {str(e)}")
            logger.exception("Retrieval failed")
            return []

    def delete_collection(self, name: str = "mlearn") -> None:
        """Delete a ChromaDB collection."""
        try:
            # Ensure collection exists
            if self.collection is None or self.collection.name != name:
                self.create_or_load_collection(name)
            
            # Delete the collection
            self.client.delete_collection(name)
            logger.info(f"Collection '{name}' deleted successfully.")
            
        except Exception as e:
            logger.error(f"Error deleting collection {name}: {str(e)}")
    
    def clear_collection(self, name: str = "mlearn") -> None:
        """Clear all documents from a collection without deleting it."""
        try:
            # Get the collection 
            collection = self.create_or_load_collection(name)
            
            # Check if collection exists and has documents
            try:
                count = collection.count()
                if count > 0:
                    logger.info(f"Clearing {count} documents from collection '{name}'")
                    # Delete the collection and recreate it
                    self.client.delete_collection(name)
                    collection = self.create_or_load_collection(name)
                    logger.info(f"Collection '{name}' cleared and recreated successfully")
                else:
                    logger.info(f"Collection '{name}' is already empty")
            except Exception as inner_e:
                # If count fails, collection might be corrupted, try recreating
                logger.warning(f"Error checking collection '{name}': {str(inner_e)}")
                # Delete and recreate
                try:
                    self.client.delete_collection(name)
                except:
                    pass
                collection = self.create_or_load_collection(name)
                logger.info(f"Collection '{name}' recreated successfully")
                
        except Exception as e:
            logger.error(f"Error clearing collection {name}: {str(e)}")
    
    def get_collection_info(self, collection_name: str = "mlearn") -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            # Get the collection
            collection = self.create_or_load_collection(collection_name)
            
            # Get collection size
            count = collection.count()
            
            # Get peek of documents (if any)
            sample_data = None
            if count > 0:
                sample_data = collection.peek(limit=2)
            
            # Return collection info
            return {
                "name": collection_name,
                "count": count,
                "sample": sample_data
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {str(e)}")
            return {
                "name": collection_name,
                "error": str(e),
                "count": 0,
                "sample": None
            }