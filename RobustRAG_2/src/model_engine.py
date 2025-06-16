"""
ModelEngine: Centralized component for model initialization across the RobustRAG system.

This ensures models are only loaded once and shared across components.
"""
import logging
import torch
from typing import Dict, Any, List, Union, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np

from . import LLM_MODEL_PATH, EMBEDDING_MODEL_PATH, MAX_NEW_TOKENS, LLM_TEMPERATURE

# Set up logging
logger = logging.getLogger(__name__)

class ModelEngine:
    """
    Singleton class for managing ML models across the application
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating ModelEngine instance")
            cls._instance = super(ModelEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the model engine if n ot already initialized."""
        if self._initialized:
            return
            
        try:
            # Initialize attributes
            self._llm = None
            self._tokenizer = None
            self._embedding_model = None
            
            # Flag as initialized
            self._initialized = True
            logger.info("ModelEngine initialized (models will be loaded on demand)")
            
        except Exception as e:
            logger.error(f"Error initializing ModelEngine: {str(e)}")
            raise
    
    def get_llm(self, model_path: str = None) -> Dict[str, Any]:
        """Get the LLM model and tokenizer, loading if necessary."""
        if self._llm is None:
            try:
                # Use config value if not provided
                if model_path is None:
                    model_path = LLM_MODEL_PATH
                    
                # Initialize the LLM
                logger.info(f"Loading LLM from {model_path}")
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                self._llm = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="cuda",
                    torch_dtype=torch.bfloat16,
                )
                logger.info("LLM loaded successfully")
            except Exception as e:
                logger.error(f"Error loading LLM: {str(e)}")
                raise
                
        return {
            "model": self._llm,
            "tokenizer": self._tokenizer
        }
    
    def get_embedding_model(self, model_path: str = None) -> SentenceTransformer:
        """Get the embedding model, loading if necessary."""
        if self._embedding_model is None:
            try:
                # Use config value if not provided
                if model_path is None:
                    model_path = EMBEDDING_MODEL_PATH
                
                # Initialize embedding model
                logger.info(f"Loading embedding model from {model_path}")

                # Load the model
                self._embedding_model = SentenceTransformer(model_path)
                # Move to GPU
                self._embedding_model.to('cuda')
                # Set to 16-bit precision
                self._embedding_model.half()

                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                raise
                
        return self._embedding_model

    def generate_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for a single text or a list of texts.
        
        Args:
            texts: Single text string or a list of text strings
            
        Returns:
            For a single text: A single embedding as a list of floats
            For a list of texts: A list of embeddings (each as a list of floats)
        """
        try:
            # Check for empty input
            if not texts:
                logger.warning("Empty input for embedding generation")
                return []
                
            # Handle single text vs list of texts
            is_single_text = isinstance(texts, str)
            texts_list = [texts] if is_single_text else texts
            
            # Filter empty strings
            valid_texts = [text for text in texts_list if text and text.strip()]
            if not valid_texts:
                logger.warning("No valid texts for embedding generation")
                return [] if not is_single_text else []
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            embedding_model = self.get_embedding_model()
            embeddings = embedding_model.encode(valid_texts)
            
            # Convert to Python lists if needed
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
                
            # Return appropriate format based on input type
            return embeddings[0] if is_single_text else embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            logger.exception("Embedding generation failed")
            return [] if not is_single_text else []

    def generate_llm_response(
        self, 
        prompt: str, 
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a response from the LLM model based on a prompt.
        
        Args:
            prompt: The prompt text to send to the LLM
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature value for text generation
            
        Returns:
            The generated text response
        """
        try:
            # Ensure LLM is loaded
            llm_dict = self.get_llm()
            llm = llm_dict["model"]
            tokenizer = llm_dict["tokenizer"]
            
            # Use default values if not provided
            if max_new_tokens is None:
                max_new_tokens = MAX_NEW_TOKENS
            if temperature is None:
                temperature = LLM_TEMPERATURE
                
            # Tokenize the input prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
            input_length = inputs.input_ids.shape[1]
            
            # Generate response
            logger.info(f"Generating LLM response with max_new_tokens={max_new_tokens}, temperature={temperature}")
            with torch.no_grad():
                outputs = llm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature
                )
            
            # Decode only the newly generated tokens
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            logger.exception("LLM response generation failed")
            return f"Error generating response: {str(e)}"

# Create a global instance for easy importing
model_engine = ModelEngine()
