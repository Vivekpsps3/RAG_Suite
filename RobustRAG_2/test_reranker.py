#!/usr/bin/env python
"""
Test script for demonstrating the reranker functionality in the RobustRAG_2 system.
"""
import logging
from src.application_engine import ApplicationEngine
from src.model_engine import model_engine
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_reranking():
    """Test the reranker functionality with a sample query."""
    # Initialize the application engine
    logger.info("Initializing ApplicationEngine...")
    app_engine = ApplicationEngine()
    
    # Sample query
    query = "What are the main financial risks for Amazon in 2020?"
    
    # Time the execution
    start_time = time.time()
    
    # Execute query with reranking
    logger.info(f"Executing query: '{query}'")
    answer = app_engine.execute_query(query)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Print result
    print("\n" + "="*80)
    print(f"Query: {query}")
    print("="*80)
    print(f"Answer: {answer}")
    print("="*80)
    print(f"Execution time: {execution_time:.2f} seconds")
    print("="*80)
    
    # Get and display reranking stats
    stats = app_engine.get_last_query_stats()
    print("\nReranking Statistics:")
    print(f"Initial results: {stats['initial_results_count']}")
    print(f"Additional results: {stats['additional_results_count']}")
    print(f"Total after reranking: {stats['reranked_results_count']}")
    print(f"Final top results: {stats['top_results_count']}")
    
    # Display top 3 result scores if available
    if stats['top_results'] and len(stats['top_results']) > 0:
        print("\nTop 3 Results Scores:")
        for i, result in enumerate(stats['top_results'][:3]):
            print(f"  {i+1}. Score: {result['score']:.4f} - Metadata: {result.get('metadata', {})}")
    print("="*80)

if __name__ == "__main__":
    test_reranking()
