import torch
import torch.nn as nn
from typing import Union, Tuple
import sys
import os

from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Transformers
from custom_transformers.transformer_1 import Transformer1
from custom_transformers.transformer_2 import Transformer2
from custom_transformers.transformer import Transformer

# Translators
from translation.translator import Translator
from translation.helsinki_translator import HelsinkiTranslator

from llm.llm_integration import LLMWrapper


class FacebookLLM(LLMWrapper):

    def __init__(self, model_name):
        self.llm_model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = OPTForCausalLM.from_pretrained(model_name)

        super(FacebookLLM, self).__init__(model_name, self.tokenizer, self.llm_model)
