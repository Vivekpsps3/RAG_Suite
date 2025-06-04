import os
import csv
from typing import List, Dict, Tuple, Any # Updated Tuple to Any for Dict return
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVProcessor:
    """
    A basic CSV processor that extracts text from CSV files.
    """
    
    def __init__(self, csv_directory: str = "documents/csv"):
        """
        Initialize the CSV processor.
        
        Args:
            csv_directory: Directory containing CSV files to process
        """
        self.csv_directory = csv_directory
        
        # Create directory if it doesn't exist
        if not os.path.exists(csv_directory):
            os.makedirs(csv_directory)
            logger.info(f"Created directory: {csv_directory}")
    
    def _extract_text_from_csv(self, csv_path: str) -> str:
        """
        Extract text from a single CSV file.
        Each row is concatenated into a single string.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Extracted text content
        """
        try:
            logger.info(f"Processing CSV: {csv_path}")
            extracted_text = ""
            with open(csv_path, mode='r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                for row_num, row in enumerate(reader, 1):
                    row_text = ", ".join(cell.strip() for cell in row if cell)
                    if row_text: # Add non-empty rows
                        extracted_text += f"Row {row_num}: {row_text}\n"
            
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Error processing {csv_path}: {str(e)}")
            return ""
    
    def _get_csv_files(self) -> List[str]:
        """
        Get all CSV files in the specified directory.
        
        Returns:
            List of CSV file paths
        """
        csv_files = []
        
        if not os.path.exists(self.csv_directory):
            logger.warning(f"Directory does not exist: {self.csv_directory}")
            return csv_files
        
        for filename in os.listdir(self.csv_directory):
            if filename.lower().endswith('.csv'):
                csv_path = os.path.join(self.csv_directory, filename)
                csv_files.append(csv_path)
        
        logger.info(f"Found {len(csv_files)} CSV files")
        return csv_files
    
    def _process_all_csvs(self) -> Dict[str, str]:
        """
        Process all CSV files in the directory and extract text.
        
        Returns:
            Dictionary mapping CSV filenames to extracted text
        """
        csv_files = self._get_csv_files()
        results = {}
        
        if not csv_files:
            logger.warning("No CSV files found to process")
            return results
        
        for csv_path in csv_files:
            filename = os.path.basename(csv_path)
            logger.info(f"Starting extraction for: {filename}")
            
            extracted_text = self._extract_text_from_csv(csv_path)
            
            if extracted_text:
                results[filename] = extracted_text
                logger.info(f"Successfully extracted text from: {filename}")
            else:
                logger.warning(f"No text extracted from: {filename}")
        
        logger.info(f"Completed processing {len(results)} CSV files")
        return results
    
    def get_documents_for_rag(self) -> List[Dict[str, Any]]:
        """
        Process CSVs and return documents in a format suitable for RAG.
        
        Returns:
            List of dictionaries, each with "id", "text", and "metadata" keys.
        """
        csv_texts = self._process_all_csvs()
        documents = []
        
        for filename, text in csv_texts.items():
            doc_id = os.path.splitext(filename)[0]
            documents.append({
                "id": f"csv_{doc_id}",
                "text": text,
                "metadata": {"source_type": "csv"}
            })
        
        return documents

def main():
    """
    Main function to demonstrate CSV processing.
    """
    # Create a dummy CSV file for testing
    dummy_csv_dir = "documents/csv"
    if not os.path.exists(dummy_csv_dir):
        os.makedirs(dummy_csv_dir)
    
    dummy_csv_path = os.path.join(dummy_csv_dir, "sample.csv")
    with open(dummy_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Age", "City"])
        writer.writerow(["Alice", "30", "New York"])
        writer.writerow(["Bob", "24", "Paris"])
        writer.writerow(["Charlie", "35", "London"])

    processor = CSVProcessor()
    
    # Process all CSVs and get results
    results = processor.get_documents_for_rag()
    
    # Print summary
    print(f"\nProcessed {len(results)} CSV files:")
    for doc_info in results: # Updated variable name
        print(f"- {doc_info['id']}: {len(doc_info['text'])} characters extracted")
        print(f"  Preview: {doc_info['text'][:150]}...")
        print(f"  Metadata: {doc_info['metadata']}")
        print()

    # Clean up dummy file
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)
    # Clean up dummy dir if empty
    if os.path.exists(dummy_csv_dir) and not os.listdir(dummy_csv_dir):
        os.rmdir(dummy_csv_dir)


if __name__ == "__main__":
    main()
