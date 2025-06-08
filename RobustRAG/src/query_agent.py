"""
QueryAgent: Component for intelligent query processing with LLM-based source selection.
"""
import json
from typing import List, Dict
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logger = logging.getLogger(__name__)

class QueryAgent:
    """Agent for intelligent query processing using LLM."""
    
    def __init__(
        self, 
        header_db: Dict[str, List[str]],
        model_name: str = "/home/vivek/Files/Model_Files/Dolphin3.0-Llama3.1-8B"
    ):
        """Initialize the query agent."""
        self.header_db = header_db
        
        # Static prompt template
        self.metadata_prompt = """
        Which files would best answer: "{query}"?

        Available file schemas:
        {header_db}

        RESPOND EXCLUSIVELY IN JSON:
        {{ "sources": ["file1.csv", ...], "extra_context": "..." }}
        """
        
        try:
            # Initialize LLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load the model with appropriate settings
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}")
            raise
    
    def get_relevant_sources(self, query: str) -> List[str]:
        """Use LLM to determine relevant sources for a query."""
        try:
            # Format prompt with query and header database
            prompt = self.metadata_prompt.format(
                query=query,
                header_db=str(self.header_db)
            )
            
            # Generate LLM response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                # Extract JSON part
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_text = response[start_idx:end_idx]
                    data = json.loads(json_text)
                    return data.get("sources", [])
                else:
                    return []
                
            except json.JSONDecodeError:
                return []
            
        except Exception as e:
            logger.error(f"Error getting relevant sources: {str(e)}")
            return []
    
    def reformulate_query(self, query: str, source_keywords: List[str]) -> str:
        """Reformulate a query with source-specific keywords."""
        try:
            # Get all keywords from sources
            all_keywords = []
            for source in source_keywords:
                if source in self.header_db:
                    all_keywords.extend(self.header_db[source])
            
            # Get unique keywords (limited to 10)
            unique_keywords = list(set(all_keywords))[:10]
            
            if not unique_keywords:
                return query
            
            # Create reformulated query
            keyword_context = " | ".join(unique_keywords)
            return f"{query} ({keyword_context})"
            
        except Exception as e:
            logger.error(f"Error reformulating query: {str(e)}")
            return query
