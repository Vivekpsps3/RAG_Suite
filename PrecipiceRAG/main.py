import os
import sys
import logging
import argparse
import json

# Import RobustRAG components - logging is already configured in __init__
from src import logger
from src.application_engine import ApplicationEngine

def interactive_query_mode(engine: ApplicationEngine) -> None:
    """
    Run an interactive query session.
    """
    print("\n=== RobustRAG Interactive Query Mode ===")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        query = input("\nEnter your question: ").strip()
        
        if query.lower() in ["exit", "quit"]:
            print("Exiting interactive mode.")
            break
        
        if not query:
            continue
        
        try:
            print("\nGenerating answer...")
            answer = engine.execute_query(query)
            
            print("\n=== Answer ===")
            print(answer)
            print("=============")
            
            # Get reranking statistics
            stats = engine.get_last_query_stats()
            if stats:
                print("\n=== Context Information ===")
                print(f"Initial results retrieved: {stats['initial_results_count']}")
                print(f"Additional results retrieved: {stats['additional_results_count']}")
                print(f"Top results used for context: {stats['top_results_count']}")
                print("=============================")
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            print(f"\nError: {str(e)}")

def check_vector_store(engine: ApplicationEngine) -> None:
    """Check the vector store status and print debug information."""
    print("\n=== Vector Store Status ===")
    info = engine.vector_store.get_collection_info()
    print(f"Collection name: {info['name']}")
    print(f"Document count: {info['count']}")
    print(f"Chunk size: {engine.csv_processor.chunk_size}")
    
    if info['sample'] and info['count'] > 0:
        print("\nSample documents:")
        sample = info['sample']
        for i, doc_id in enumerate(sample['ids']):
            print(f"Document {i+1} ID: {doc_id}")
            print(f"  Text: {sample['documents'][i][:100]}...")
            print(f"  Metadata: {sample['metadatas'][i]}")
            print(f"  Embedding: {len(sample['embeddings'][i])} dimensions")
        print("=========================")
    else:
        print("\nNo documents found in the vector store!")
        print("=========================")

def main() -> None:
    """Main entry point for the RobustRAG system."""
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="RobustRAG system")
        parser.add_argument("--new", action="store_true", help="Create a new collection (clears existing data)")
        parser.add_argument("--check", action="store_true", help="Check vector store status")
        args = parser.parse_args()
        
        # Initialize the application engine (will auto-process CSVs)
        logger.info("Initializing ApplicationEngine...")
        load_new = args.new
        engine = ApplicationEngine(load_new_collection=load_new)
        logger.info("ApplicationEngine initialized successfully")
        
        # Check vector store status if requested
        if args.check:
            check_vector_store(engine)
            return
                
        # Run in interactive mode
        interactive_query_mode(engine)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()