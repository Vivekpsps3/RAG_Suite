import os
import pytesseract
from pdf2image import convert_from_path
from typing import List, Dict, Tuple, Any # Updated Tuple to Any for Dict return
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    A basic PDF processor that uses OCR to extract text from PDF files.
    """
    
    def __init__(self, pdf_directory: str = "documents/pdf"):
        """
        Initialize the PDF processor.
        
        Args:
            pdf_directory: Directory containing PDF files to process
        """
        self.pdf_directory = pdf_directory
        
        # Create directory if it doesn't exist
        if not os.path.exists(pdf_directory):
            os.makedirs(pdf_directory)
            logger.info(f"Created directory: {pdf_directory}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a single PDF file using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            extracted_text = ""
            
            # Process each page
            for page_num, image in enumerate(images, 1):
                logger.info(f"Processing page {page_num}/{len(images)}")
                
                # Use OCR to extract text from image
                page_text = pytesseract.image_to_string(image)
                
                # Add page separator
                extracted_text += f"\n--- Page {page_num} ---\n"
                extracted_text += page_text + "\n"
            
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return ""
    
    def _get_pdf_files(self) -> List[str]:
        """
        Get all PDF files in the specified directory.
        
        Returns:
            List of PDF file paths
        """
        pdf_files = []
        
        if not os.path.exists(self.pdf_directory):
            logger.warning(f"Directory does not exist: {self.pdf_directory}")
            return pdf_files
        
        for filename in os.listdir(self.pdf_directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_directory, filename)
                pdf_files.append(pdf_path)
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    
    def _process_all_pdfs(self) -> Dict[str, str]:
        """
        Process all PDF files in the directory and extract text.
        
        Returns:
            Dictionary mapping PDF filenames to extracted text
        """
        pdf_files = self._get_pdf_files()
        results = {}
        
        if not pdf_files:
            logger.warning("No PDF files found to process")
            return results
        
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            logger.info(f"Starting extraction for: {filename}")
            
            extracted_text = self._extract_text_from_pdf(pdf_path)
            
            if extracted_text:
                results[filename] = extracted_text
                logger.info(f"Successfully extracted text from: {filename}")
            else:
                logger.warning(f"No text extracted from: {filename}")
        
        logger.info(f"Completed processing {len(results)} PDF files")
        return results
    
    def get_documents_for_rag(self) -> List[Dict[str, Any]]:
        """
        Process PDFs and return documents in a format suitable for RAG.
        
        Returns:
            List of dictionaries, each with "id", "text", and "metadata" keys.
        """
        pdf_texts = self._process_all_pdfs()
        documents = []
        
        for filename, text in pdf_texts.items():
            doc_id = os.path.splitext(filename)[0]
            documents.append({
                "id": f"pdf_{doc_id}",
                "text": text,
                "metadata": {"source_type": "pdf"}
            })
        
        return documents


def main():
    """
    Main function to demonstrate PDF processing.
    """
    processor = PDFProcessor()
    
    # Process all PDFs and get results
    results = processor.get_documents_for_rag()
    
    # Print summary
    print(f"\nProcessed {len(results)} PDF files:")
    for doc_info in results: # Updated variable name
        print(f"- {doc_info['id']}: {len(doc_info['text'])} characters extracted")
        print(f"  Preview: {doc_info['text'][:100]}...")
        print(f"  Metadata: {doc_info['metadata']}")
        print()


if __name__ == "__main__":
    main()
