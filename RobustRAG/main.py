#!/usr/bin/env python3
"""
RobustRAG: Main entry point for the RobustRAG system.

Automatically processes CSVs from documents/csv folder and provides query functionality.
"""
import os
import sys
import logging
import argparse
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RobustRAG components
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
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            print(f"\nError: {str(e)}")

def check_vector_store(engine: ApplicationEngine) -> None:
    """Check the vector store status and print debug information."""
    print("\n=== Vector Store Status ===")
    info = engine.vector_store.get_collection_info()
    print(f"Collection name: {info['name']}")
    print(f"Document count: {info['count']}")
    
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
        parser.add_argument("--debug", action="store_true", help="Print debug information")
        parser.add_argument("--new", action="store_true", help="Create a new collection (clears existing data)")
        parser.add_argument("--reload", action="store_true", help="Force reload all CSV files into vector store")
        args = parser.parse_args()
        
        # Initialize the application engine (will auto-process CSVs)
        logger.info("Initializing ApplicationEngine...")
        load_new = args.new or args.reload
        engine = ApplicationEngine(load_new_collection=load_new)
        logger.info("ApplicationEngine initialized successfully")
        
        # Print debug information if requested
        if args.debug:
            check_vector_store(engine)
        
        # Run in interactive mode
        interactive_query_mode(engine)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()