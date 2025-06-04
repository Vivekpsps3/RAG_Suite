import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import uuid

class NaiveRAGRetriever:
    """
    A naive RAG (Retrieval-Augmented Generation) system that uses ChromaDB
    for storing and retrieving document embeddings.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        embedder: Any = None  # Default to the None, will error out
    ):
        """
        Initialize the RAG retriever with ChromaDB and embedder.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Model name for the embedder
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize the embedder using tf_tools
        self.embedder = embedder
        if self.embedder is None:
            raise ValueError("Embedder must be provided. Please initialize with a valid embedding model.")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 50
    ) -> List[str]:
        """
        Add documents to the vector store, chunked with overlap.
        
        Args:
            documents: List of document texts to embed and store
            metadatas: Optional metadata for each document
            ids: Optional custom IDs for documents
            chunk_size: Max tokens per chunk
            chunk_overlap: Overlap tokens between chunks
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []

        # Helper: simple whitespace tokenizer
        def tokenize(text):
            return text.split()

        # Helper: detokenize
        def detokenize(tokens):
            return " ".join(tokens)

        all_chunks = []
        all_chunk_metadatas = []
        all_chunk_ids = []

        for doc_idx, doc in enumerate(documents):
            tokens = tokenize(doc)
            doc_chunks = []
            doc_chunk_metadatas = []
            doc_chunk_ids = []
            start = 0
            chunk_num = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = detokenize(chunk_tokens)
                doc_chunks.append(chunk_text)

                # Prepare metadata for this chunk
                base_metadata = metadatas[doc_idx] if metadatas and doc_idx < len(metadatas) else {}
                chunk_metadata = dict(base_metadata)
                chunk_metadata["chunk_index"] = chunk_num
                chunk_metadata["chunk_start"] = start
                chunk_metadata["chunk_end"] = end
                chunk_metadata["parent_doc_index"] = doc_idx
                chunk_metadata["parent_doc_length"] = len(tokens)
                chunk_metadata["text"] = chunk_text

                doc_chunk_metadatas.append(chunk_metadata)

                # Generate chunk ID
                if ids and doc_idx < len(ids):
                    base_id = ids[doc_idx]
                else:
                    base_id = str(uuid.uuid4())
                chunk_id = f"{base_id}_chunk{chunk_num}"
                doc_chunk_ids.append(chunk_id)

                chunk_num += 1
                start += chunk_size - chunk_overlap

            all_chunks.extend(doc_chunks)
            all_chunk_metadatas.extend(doc_chunk_metadatas)
            all_chunk_ids.extend(doc_chunk_ids)

        # Generate embeddings for all chunks
        embeddings = self.embedder.encode(all_chunks, convert_to_tensor=True)

        # Add to ChromaDB collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=all_chunks,
            metadatas=all_chunk_metadatas,
            ids=all_chunk_ids
        )

        return all_chunk_ids
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        include_similarities: bool = True
    ) -> Dict[str, Any]:
        """
        Search for similar documents given a query.
        
        Args:
            query_text: The search query
            top_k: Number of top results to return
            include_similarities: Whether to include similarity scores
            
        Returns:
            Dictionary containing retrieved documents and metadata
        """
        # Generate embedding for the query
        query_embedding = self.embedder.encode(query_text, convert_to_tensor=True)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = {
            "query": query_text,
            "documents": results['documents'][0] if results['documents'] else [],
            "metadatas": results['metadatas'][0] if results['metadatas'] else [],
            "ids": results['ids'][0] if results['ids'] else []
        }
        
        if include_similarities:
            # Convert distances to similarities (ChromaDB returns distances)
            distances = results['distances'][0] if results['distances'] else []
            similarities = [1.0 - dist for dist in distances]
            formatted_results["similarities"] = similarities
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_model": self.embedder.model_name
        }
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the collection by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        if ids:
            self.collection.delete(ids=ids)
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def retrieve_and_format(
        self,
        query_text: str,
        top_k: int = 3,
        context_separator: str = "\n\n---\n\n"
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve documents and format them as context for RAG.
        
        Args:
            query_text: The search query
            top_k: Number of documents to retrieve
            context_separator: Separator between documents in context
            
        Returns:
            Tuple of (formatted_context, source_documents)
        """
        results = self.query(query_text, top_k=top_k, include_similarities=True)
        
        if not results["documents"]:
            return "", []
        
        # Format context
        context_parts = []
        source_docs = []
        
        for i, (doc, metadata, similarity) in enumerate(zip(
            results["documents"],
            results["metadatas"],
            results["similarities"]
        )):
            context_parts.append(f"Document {i+1}:\n{doc}")
            source_docs.append({
                "content": doc,
                "metadata": metadata,
                "similarity": similarity,
                "rank": i + 1
            })
        
        formatted_context = context_separator.join(context_parts)
        
        return formatted_context, source_docs
