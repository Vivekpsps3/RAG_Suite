"""
HeaderRepository: Component for managing CSV column headers in a structured format.
"""
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import logging

# Set up logging
logger = logging.getLogger(__name__)

class HeaderRepository:
    """
    Repository for managing CSV column headers as metadata.
    """
    
    def __init__(self, header_dir: Path = Path("documents/headers")):
        """Initialize the header repository."""
        self.header_dir = header_dir
        
        # Create directory if it doesn't exist
        if not self.header_dir.exists():
            self.header_dir.mkdir(parents=True)
            logger.info(f"Created header directory: {self.header_dir}")
    
    def save_headers(self, file_path: Path) -> None:
        """Extract headers from a CSV file and save them as JSON."""
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return
            
            # Read only the header row
            df = pd.read_csv(file_path, nrows=0)
            headers = df.columns.tolist()
            
            # Create and save header data
            header_data = {
                "headers": headers,
                "source": file_path.name
            }
            
            output_file = self.header_dir / f"{file_path.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(header_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving headers for {file_path}: {str(e)}")
    
    def load_headers(self) -> Dict[str, List[str]]:
        """Load all header files from the repository."""
        header_mapping = {}
        
        try:
            # Load all JSON files in the header directory
            for json_file in self.header_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    source = data.get("source", json_file.stem)
                    headers = data.get("headers", [])
                    header_mapping[source] = headers
                    
                except Exception as e:
                    logger.error(f"Error loading header file {json_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading headers: {str(e)}")
        
        return header_mapping
