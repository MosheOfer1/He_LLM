import torch.nn as nn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, AutoModelForCausalLM
from llm.llm_integration import LLMWrapper


class FacebookLLM(nn.Module ,LLMWrapper):

    def __init__(self, 
                 model_name: str,
                 device: str = 'cpu'):
        
        print(f"FacebookLLM.__init__ - uses: {device}")
        
        self.device = device
        
        # Initialize nn.Module
        nn.Module.__init__(self)
        
        self.llm_model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        # Initialize LLMWrapper
        LLMWrapper.__init__(self, 
                            model_name, 
                            self.tokenizer, 
                            self.llm_model,
                            device=device)
