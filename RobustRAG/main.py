#!/usr/bin/env python3
"""
RobustRAG: Main entry point for the RobustRAG system.

Automatically processes CSVs from documents/csv folder and provides query functionality.
"""
import os
import sys
import logging

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

def main() -> None:
    """Main entry point for the RobustRAG system."""
    try:
        # Initialize the application engine (will auto-process CSVs)
        logger.info("Initializing ApplicationEngine...")
        engine = ApplicationEngine()
        logger.info("ApplicationEngine initialized successfully")
        
        # Run in interactive mode
        interactive_query_mode(engine)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()