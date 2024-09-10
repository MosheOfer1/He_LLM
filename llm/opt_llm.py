import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, OPTForCausalLM
from llm.llm_wrapper import LLMWrapper


class OptLLM(LLMWrapper):
    def __init__(self, model_name: str, device: str = 'cpu'):
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_model = OPTForCausalLM.from_pretrained(model_name).to(device)

        # Initialize LLMWrapper
        super().__init__(model_name, self.tokenizer, llm_model, device)

    def get_layers(self):
        """Return the decoder layers of the OPT model."""
        return self.model.base_model.decoder.layers
