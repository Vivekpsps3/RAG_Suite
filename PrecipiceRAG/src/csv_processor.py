"""
CSVProcessor: Component for processing CSV files into document chunks for RAG.
"""
import pandas as pd
import uuid
from pathlib import Path
from typing import List, Dict, Any
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import configuration and model engine
from .model_engine import model_engine
from . import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, CSV_DOCUMENTS_DIR, JSON_DOCUMENTS_DIR
import json

# Set up logging
logger = logging.getLogger(__name__)

class CSVProcessor:
    """Processor for converting CSV data into textified documents with embeddings."""
    
    def __init__(
        self, 
        textifier_delimiter: str = " | ",
        chunk_size: int = None,
        chunk_overlap: int = None,
        embedding_model_name: str = None
    ):
        """Initialize the CSV processor."""
        self.textifier_delimiter = textifier_delimiter
        self.chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP
        
        # Initialize embedding model in model engine if needed
        try:
            model_engine.get_embedding_model(embedding_model_name)
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=lambda text: len(text.split())
        )


    def process_csv(self, file_path: Path) -> List[Dict[str, Any]]:

        #First convert the CSV to JSON
        self.convert_to_json(csv_path=file_path, json_path=JSON_DOCUMENTS_DIR)
        
        # Process the JSON file to create documents
        documents = self.json_to_documents(json_path=JSON_DOCUMENTS_DIR / "output.json")
        return documents

    def convert_to_json(self, csv_path: str, json_path: str) -> None:
        """
        Generate a JSON-like structure from all given CSV files.
        Ensures that each unique 'id' appears only once in the results.
        """
        results = []
        id_to_entry = {}  # Maps id to its entry in results

        csv_dir = Path(csv_path)
        # Get all CSV files in the directory or the single file if path is a file
        csv_files = [csv_dir] if csv_dir.is_file() else list(csv_dir.glob("*.csv"))

        for file in csv_files:
            df = pd.read_csv(file)
            # Iterate over each row in the DataFrame
            for _, row in df.iterrows():
                row_id = row['id']
                row_dict = row.to_dict()
                row_dict.pop('id', None)  # Remove 'id' from the row dict

                # If this id is not yet in id_to_entry, create a new entry
                if row_id not in id_to_entry:
                    id_to_entry[row_id] = {
                        'id': row_id,
                        'contents': []
                    }
                    results.append(id_to_entry[row_id])  # Add to results

                # Append the row's data to the contents list for this id
                id_to_entry[row_id]['contents'].append(row_dict)

        # Save results to JSON
        output_dir = Path(json_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "output.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def json_to_documents(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Convert JSON data into a list of documents for RAG.
        """
        self.convert_to_json(csv_path=CSV_DOCUMENTS_DIR, json_path=JSON_DOCUMENTS_DIR)  # Ensure JSON is generated
        documents = []
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                doc_id = entry['id']
                contents = entry['contents']
                text = self.textifier_delimiter.join(
                    [json.dumps(content, ensure_ascii=False) for content in contents]
                )
                metadata = {"source_type": "csv", "source_id": doc_id}
                documents.append({
                    "id": doc_id,
                    "text": text,
                    "embedding": model_engine.generate_embeddings(text),
                    "metadata": metadata
                })
        return documents
    

if __name__ == "__main__":
    # Example usage
    processor = CSVProcessor()
    csv_path = "../documents/csv/"  # Replace with your CSV file path
    json_path = "../documents/json/"  # Replace with your desired JSON output path
    processed_data = processor.convert_to_json(csv_path, json_path)
    print(f"Processed {len(processed_data)} entries from {csv_path}")