import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, AutoModelForCausalLM
from llm.llm_wrapper import LLMWrapper


class GPT2LLM(LLMWrapper):
    def __init__(self, model_name: str, device='cpu'):
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        # Initialize LLMWrapper
        super().__init__(model_name, self.tokenizer, llm_model, device)

    def get_layers(self):
        """Return the transformer layers of the GPT-2 model."""
        return self.model.base_model.h
