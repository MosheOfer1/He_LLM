
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Transformers
from custom_transformers.transformer_1 import Transformer1
from custom_transformers.transformer_2 import Transformer2

# Translators
from translation.translator import Translator

# LLM
from llm.llm_integration import LLMWrapper


class Transformer:
    
    def __init__(self,
                translator: Translator,
                llm: LLMWrapper):
        
        self.transformer1 = Transformer1(translator=translator,
                                         llm=llm)
    
        self.transformer2 = Transformer2(translator=translator,
                                         llm=llm)
    
    

    def get_transformer(self, first: bool):
        if first:
            return self.transformer1
        
        return self.transformer2
    
