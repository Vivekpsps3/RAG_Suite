"""
ApplicationEngine: Simplified main component for the RobustRAG system.

Automatically processes CSVs from documents/csv folder and provides query functionality.
"""
from pathlib import Path
import logging
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from .header_repository import HeaderRepository
from .vector_store_manager import VectorStoreManager
from .query_agent import QueryAgent
from .csv_processor import CSVProcessor

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
        model_path: str = "/home/vivek/Programming/FUN_PROJ/AI/Model_Library/Dolphin3.0-Llama3.2-1B",
        load_new_collection: bool = False
    ):
        """
        Initialize the application engine and process all CSVs.
        """
        try:
            # Initialize components
            self.header_repo = HeaderRepository()
            self.vector_store = VectorStoreManager()
            self.vector_store.create_or_load_collection()
            
            # Initialize the LLM
            logger.info(f"Loading LLM from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            if load_new_collection:
                # Process all CSVs in the documents/csv folder
                self._process_all_csvs()
                logger.info("Loading new collection, clearing existing vector store")
                self.vector_store.clear_collection()
            else:
                logger.info("Loading existing vector store collection")
            
            # Load header database after processing CSVs
            header_db = self.header_repo.load_headers()
            self.query_agent = QueryAgent(header_db)
            
            logger.info("ApplicationEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ApplicationEngine: {str(e)}")
            raise
    
    def _process_all_csvs(self) -> None:
        """
        Process all CSV files in the documents/csv directory.
        """
        csv_dir = Path("documents/csv")
        if not csv_dir.exists():
            logger.info(f"Creating CSV directory: {csv_dir}")
            csv_dir.mkdir(parents=True, exist_ok=True)
            return
            
        processor = CSVProcessor()
        
        # Process each CSV file
        for csv_file in csv_dir.glob("*.csv"):
            try:
                logger.info(f"Processing CSV: {csv_file}")
                
                # Save headers
                self.header_repo.save_headers(csv_file)
                
                # Process CSV
                documents = processor.process_csv(csv_file)
                
                # Add to vector store
                self.vector_store.add_documents(documents)
                
                logger.info(f"Successfully processed CSV: {csv_file}")
                
            except Exception as e:
                logger.error(f"Error processing CSV {csv_file}: {str(e)}")
    
    def execute_query(self, user_query: str) -> str:
        """
        Execute a user query with the RAG system.
        
        Args:
            user_query: User's question
            
        Returns:
            Generated answer
        """
        try:
            # 1. Get relevant sources from LLM
            source_files = self.query_agent.get_relevant_sources(user_query)
            
            # If no sources found, use empty filter
            filter_dict = None
            if source_files:
                filter_dict = {"source": {"$in": source_files}}
            
            # 2. Reformulate query with header keywords
            reformulated_query = self.query_agent.reformulate_query(user_query, source_files)
            
            # 3. Retrieve relevant documents
            results = self.vector_store.retrieve(
                query=reformulated_query,
                filter_dict=filter_dict,
                n=8  # Get more context docs
            )
            
            # 4. Format context
            context_str = "\n\n".join([r["document"] for r in results])
            
            # 5. Create final prompt
            final_prompt = f"""
            Answer the following question based ONLY on the provided context.
            If the context doesn't contain enough information, say "I don't have enough information to answer this question."
            
            CONTEXT:
            {context_str}
            
            QUESTION:
            {user_query}
            
            ANSWER:
            """
            
            # 6. Generate answer with LLM
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.llm.device)
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7
                )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part (after the prompt)
            answer_start = answer.find("ANSWER:")
            if answer_start != -1:
                answer = answer[answer_start + 7:].strip()
            
            logger.info(f"Generated answer for query: {user_query}")
            return answer
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return f"Error generating answer: {str(e)}"
