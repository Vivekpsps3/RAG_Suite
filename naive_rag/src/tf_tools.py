from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import os
from openai import OpenAI
import torch._dynamo
torch._dynamo.config.suppress_errors = True


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings :)

SYSTEM_PROMPT = """
You are a helpful assistant that provides answers based on the provided context.
If you do not know the answer, say "I don't know".
You will be given a question and a context. Use the context to answer the question.
If the context does not provide enough information, say "I don't know".
"""

class LMAnswerModel:
    """
    A base class for language model answer generation.
    """
    def __init__(self, model_name: str = "/home/vivek/Programming/FUN_PROJ/AI/Model_Library/Dolphin3.0-Llama3.2-1B", 
                 embedding_model_name: str = "/home/vivek/Programming/FUN_PROJ/AI/Model_Library/all-MiniLM-L6-v2"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.embedder = SentenceTransformer(embedding_model_name)
        self.model.generation_config.cache_implementation = "static"
        self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)

    # def generate(self, prompt: str, max_length: int = 100):
    #     """
    #     Generate a response using an external LM Studio API through OpenAI client.
        
    #     Args:
    #         prompt (str): The input prompt.
    #         max_length (int): Maximum length of the generated text.
        
    #     Returns:
    #         str: The generated response.
    #     """

    #     client = OpenAI(
    #         base_url="http://10.2.0.2:1234/v1",
    #         api_key="lm-studio"  # The API key is not strictly needed for LM Studio, but must be set
    #     )

    #     messages = [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": prompt}
    #     ]

    #     response = client.chat.completions.create(
    #         model="google/gemma-3-1b",
    #         messages=messages,
    #         temperature=0.7,
    #         top_p=0.9
    #     )

    #     return response.choices[0].message.content

    def generate(self, prompt: str, max_length: int = 100):
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("assistant")[-1].strip()

    def embed(self, text: str):
        """
        Generate embeddings for the given text using the embedding model.

        Args:
            text (str): The input text to embed.

        Returns:
            torch.Tensor: The embeddings for the input text.
        """
        return self.embedder.encode(text, convert_to_tensor=True)



# if __name__ == "__main__":
#     model = LMAnswerModel()
#     prompt = "What is the capital of France?"
#     response = model.generate(prompt)
#     print(response)