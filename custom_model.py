import torch
import torch.nn as nn
from typing import Union, Tuple
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Transformers
from custom_transformers.transformer_1 import Transformer1
from custom_transformers.transformer_2 import Transformer2
from custom_transformers.transformer import Transformer



# Translators
from translation.translator import Translator
from translation.helsinki_translator import HelsinkiTranslator



from llm.facebook_llm import facebookLLM


class MyCustomModel(nn.Module):

    def __init__(self, 
                  src_to_target_translator_model_name, 
                  target_to_src_translator_model_name,
                  llm_model_name):
        
        super(MyCustomModel, self).__init__()
        
        # Custom Translator
        self.translator = HelsinkiTranslator(src_to_target_translator_model_name,
                                             target_to_src_translator_model_name)
        # Custom LLM
        self.llm = facebookLLM(llm_model_name)
        
        
        self.transformer = Transformer(translator=self.translator,
                                        llm=self.llm)
        
        # Freeze Translator1 parameters
        self.translator.set_requires_grad(False)

        # Freeze LLM parameters
        self.llm.set_requires_grad(False)
        
        
        
    def forward(self):
        pass

