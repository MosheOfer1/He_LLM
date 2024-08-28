import sys
import os
from transformers import AutoTokenizer, OPTForCausalLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm.llm_integration import LLMWrapper


class FacebookLLM(LLMWrapper):

    def __init__(self, model_name):
        self.llm_model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = OPTForCausalLM.from_pretrained(model_name)

        super(FacebookLLM, self).__init__(model_name, self.tokenizer, self.llm_model)
