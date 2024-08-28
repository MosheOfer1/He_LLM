from transformers import MarianMTModel, MarianTokenizer
import torch.nn as nn

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from translation.translator import Translator

class HelsinkiTranslator(nn.Module, Translator):
    def __init__(self, src_to_target_translator_model_name,
                 target_to_src_translator_model_name):
        """
        Initializes the HelsinkiTranslator with pretrained MarianMT models and tokenizers
        for translating between Hebrew and English.

        Models and tokenizers are loaded for both directions of translation:
        from Hebrew to English and from English to Hebrew.
        """
        
        # Initialize nn.Module
        nn.Module.__init__(self)
        
        # Initialize Translator
        Translator.__init__(self,
            src_to_target_translator_model_name,
            target_to_src_translator_model_name,
            MarianMTModel.from_pretrained(src_to_target_translator_model_name, output_hidden_states=True),
            MarianMTModel.from_pretrained(target_to_src_translator_model_name, output_hidden_states=True),
            MarianTokenizer.from_pretrained(src_to_target_translator_model_name),
            MarianTokenizer.from_pretrained(target_to_src_translator_model_name)
        )
