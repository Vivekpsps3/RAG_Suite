"""
ApplicationEngine: Simplified main component for the RobustRAG system.

Automatically processes CSVs from documents/csv folder and provides query functionality.
"""
from pathlib import Path
import logging
import os
from typing import Dict, Any, List

# Import constants and configs
from . import (
    CSV_DOCUMENTS_DIR, DEFAULT_CHUNK_SIZE, NUM_CTX_RESULTS,
    QUERY_PROMPT, MAX_NEW_TOKENS, LLM_TEMPERATURE,
    ADDITIONAL_CONTEXT_PROMPT
)

# Import components
from .vector_store_manager import VectorStoreManager
from .csv_processor import CSVProcessor
from .model_engine import model_engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ApplicationEngine:
    """
    Main application engine for RobustRAG system.
    
    Automatically processes CSVs in documents/csv folder and handles queries.
    """
    
    def __init__(
        self,
        model_path: str = None,
        load_new_collection: bool = False
    ):
        """
        Initialize the application engine and process all CSVs.
        
        Args:
            model_path: Path to the LLM model
            load_new_collection: If True, clears existing vectors before processing CSVs
        """
        try:
            # Initialize components
            self.vector_store = VectorStoreManager()
            self.vector_store.create_or_load_collection()
            self.csv_processor = CSVProcessor(chunk_size=DEFAULT_CHUNK_SIZE)
            
            # Initialize the model engine (which lazily loads models)
            # This ensures the model_path is set if provided
            model_engine.get_llm(model_path)
            
            # Storage for last query information (for debugging)
            self._last_query_info = None
            
            # Optionally clear the collection before processing
            if load_new_collection:
                logger.info("Clearing existing vector store collection")
                self.vector_store.clear_collection()
                # Process all CSVs
                self._process_all_csvs()
                        
            logger.info("ApplicationEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ApplicationEngine: {str(e)}")
            raise
    
    def _process_all_csvs(self) -> None:
        """
        Process all CSV files in the documents/csv directory.
        """
        csv_dir = CSV_DOCUMENTS_DIR
        if not csv_dir.exists():
            logger.info(f"Creating CSV directory: {csv_dir}")
            csv_dir.mkdir(parents=True, exist_ok=True)
            return
            
        # Use the stored CSV processor instance
        processor = self.csv_processor
        
        # Process each CSV file
        csv_files = list(csv_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        for csv_file in csv_files:
            try:
                logger.info(f"Processing CSV: {csv_file}")
                
                # Process CSV
                documents = processor.process_csv(csv_file)
                
                # Log the number of documents and check for embeddings
                if documents:
                    logger.info(f"Generated {len(documents)} documents with embeddings")
                    # Verify first document has proper embedding
                    if documents[0].get('embedding') is None:
                        logger.error("ERROR: Documents do not have embeddings!")
                    elif len(documents[0]['embedding']) == 0:
                        logger.error("ERROR: Embedding vector is empty!")
                    else:
                        logger.info(f"Embedding dimension: {len(documents[0]['embedding'])}")
                    
                    # Add to vector store in batches
                    self.vector_store.add_documents(documents)
                    
                    # Verify the documents were added by checking collection count
                    collection_info = self.vector_store.get_collection_info()
                    logger.info(f"Collection now has {collection_info['count']} total documents")
                else:
                    logger.warning(f"No documents generated from {csv_file}")
                
                logger.info(f"Successfully processed CSV: {csv_file}")
                
            except Exception as e:
                logger.error(f"Error processing CSV {csv_file}: {str(e)}")
                logger.exception("CSV processing failed")
    
    def execute_query(self, user_query: str) -> str:
        """
        Execute a user query with the RAG system.
        
        Args:
            user_query: User's question
            
        Returns:
            Generated answer
        """
        try:
            # Retrieve relevant documents
            initial_results = self.vector_store.retrieve(
                query=user_query,
                n=NUM_CTX_RESULTS  # Get context docs
            )
            
            # Format initial context for additional query generation
            initial_context_str = "\n\n".join(["Opportunity id:" + str(r["id"]) + "\n" + str(r["document"]) for r in initial_results])

            # Additional Context Retrieval
            add_context_prompt = ADDITIONAL_CONTEXT_PROMPT.format(context=initial_context_str, query=user_query)
            additional_context_query = model_engine.generate_llm_response(add_context_prompt)
            logger.info(f"Additional context query: {additional_context_query}")

            additional_results = self.vector_store.retrieve(
                query=additional_context_query,
                n=NUM_CTX_RESULTS  # Get additional context docs
            )
            
            # Combine initial and additional results
            combined_results = initial_results + additional_results
            logger.info(f"Combined {len(initial_results)} initial and {len(additional_results)} additional results")
            
            # Basic deduplication - use set to track unique documents
            logger.info("Using basic deduplication")
            seen_content = set()
            unique_results = []
            
            # Simple deduplication based on content
            for result in combined_results:
                doc = result["document"]
                if len(doc) >= 200:
                    signature = doc[:100] + doc[-100:]
                else:
                    signature = doc
                
                if signature not in seen_content:
                    seen_content.add(signature)
                    unique_results.append(result)
            
            # Sort by original similarity score and take top 10
            sorted_results = sorted(unique_results, 
                                  key=lambda x: x.get("score", 0.0), 
                                  reverse=True)
            top_results = sorted_results[:10]
            
            # Store the query info for potential debugging
            self._last_query_info = {
                'query': user_query,
                'initial_results_count': len(initial_results),
                'additional_results_count': len(additional_results),
                'unique_results_count': len(unique_results),
                'top_results_count': len(top_results),
                'top_results': [{'score': r['score'], 'metadata': r.get('metadata', {})} for r in top_results]
            }
            
            logger.info(f"Selected top {len(top_results)} results after deduplication")
            
            # Format final context from results
            context_str = "\n\n".join(["Opportunity id:" + str(r["id"]) + "\n" + str(r["document"]) for r in top_results])
            logger.info(f"Retrieved context for query: {user_query}")
            
            # Create final prompt
            final_prompt = QUERY_PROMPT.format(
                query=user_query,
                context=context_str
            )
            logger.info(f"Generated prompt for LLM:\n{final_prompt}\n")

            # Generate answer with LLM
            answer = model_engine.generate_llm_response(final_prompt)

            logger.info(f"Generated answer for query: {user_query}")
            return answer
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def get_last_query_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the last executed query, including reranking details.
        
        Returns:
            Dictionary with statistics about the last query
        """
        if self._last_query_info is None:
            return {"error": "No query has been executed yet"}
        return self._last_query_info
