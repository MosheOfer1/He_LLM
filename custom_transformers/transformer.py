
import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Transformers
from custom_transformers.short_rnn_transformer1 import Transformer1
from custom_transformers.transformer_2 import Transformer2

# Translators
from translation.translator import Translator

# LLM
from llm.llm_integration import LLMWrapper


# TODO - allow loading pretrained transformers

class Transformer(nn.Module):
    
    def __init__(self,
                translator: Translator = None,
                llm: LLMWrapper = None,
                pretrained_transformer1_path: str = None,
                pretrained_transformer2_path: str = None):
        
        nn.Module.__init__(self)
        
        # Obtain transformer1
        if pretrained_transformer1_path:
            # self.transformer1: Transformer1 = torch.load(pretrained_transformer1_path)
            pass

        else:
            self.transformer1 = Transformer1(translator=translator, llm=llm)

        # Obtain transformer2
        if pretrained_transformer2_path:
            self.transformer2: Transformer2 = torch.load(pretrained_transformer2_path)
        else:
            self.transformer2 = Transformer2(translator=translator, llm=llm)
