#!/usr/bin/env python3
"""
RobustRAG test script to verify system functionality.

This script runs a simple test flow to verify that all components
of the RobustRAG system are working correctly.
"""
import os
import sys
import pandas as pd
import logging
from pathlib import Path
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import RobustRAG modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.application_engine import ApplicationEngine

def create_test_csvs():
    """Create test CSV files in the documents/csv folder."""
    # Create CSV directory
    csv_dir = Path("documents/csv")
    csv_dir.mkdir(exist_ok=True, parents=True)
    
    # Clean out any existing files
    for file in csv_dir.glob("*.csv"):
        file.unlink()
    
    # Create companies test data
    companies_data = {
        "Company": ["TechCorp", "AI Solutions", "DataViz", "CloudScale", "ML Experts"],
        "Industry": ["Software", "Artificial Intelligence", "Data Analytics", "Cloud Computing", "Machine Learning"],
        "Funding": ["$5M", "$12M", "$3M", "$8M", "$15M"],
        "Employees": [50, 120, 30, 80, 150],
        "Location": ["San Francisco", "New York", "Boston", "Seattle", "Austin"]
    }
    
    companies_df = pd.DataFrame(companies_data)
    companies_file = csv_dir / "companies.csv"
    companies_df.to_csv(companies_file, index=False)
    logger.info(f"Created test file: {companies_file}")
    
    # Create products test data
    products_data = {
        "Product": ["SmartAnalytics", "CloudManager", "DataInsights", "MLFramework", "SecuritySuite"],
        "Category": ["Analytics", "Cloud", "Data", "Machine Learning", "Security"],
        "Price": ["$99/month", "$199/month", "$149/month", "$299/month", "$249/month"],
        "Rating": [4.5, 4.2, 4.8, 4.3, 4.7]
    }
    
    products_df = pd.DataFrame(products_data)
    products_file = csv_dir / "products.csv"
    products_df.to_csv(products_file, index=False)
    logger.info(f"Created test file: {products_file}")

def run_test():
    """Run a complete test of the RobustRAG system."""
    logger.info("Starting RobustRAG test")
    
    # Create test data in documents/csv folder
    create_test_csvs()
    
    try:
        # Initialize the application engine
        # This will automatically process the CSV files we just created
        logger.info("Initializing ApplicationEngine...")
        engine = ApplicationEngine()
        logger.info("ApplicationEngine initialized successfully")
        
        # Test queries
        test_queries = [
            "What companies have the highest funding?",
            "Which products are in the machine learning category?",
            "What are the top rated products?",
            "Which company is in the artificial intelligence industry?"
        ]
        
        # Run test queries
        for query in test_queries:
            logger.info(f"Testing query: {query}")
            answer = engine.execute_query(query)
            print(f"\nQuery: {query}\nAnswer: {answer}\n")
            print("-" * 50)
        
        logger.info("RobustRAG test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    run_test()
