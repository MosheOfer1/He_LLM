import torch
import torch.nn as nn

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Transformers
from custom_transformers.transformer_1 import Transformer1
from custom_transformers.transformer_2 import Transformer2

# Translators
from translation.translator import Translator

# LLM
from llm.llm_wrapper import LLMWrapper


# TODO - allow loading pretrained transformers

class Transformer(nn.Module):

    def __init__(self,
                 translator: Translator = None,
                 llm: LLMWrapper = None,
                 pretrained_transformer1_path: str = None,
                 pretrained_transformer2_path: str = None,
                 device: str = 'cpu'):

        print(f"Transformer.__init__ - uses: {device}")

        self.device = device

        nn.Module.__init__(self)
        # Obtain transformer1
        if pretrained_transformer1_path:
            self.transformer1 = Transformer1.load_model(
                model_name=pretrained_transformer1_path,
                translator=translator,
                llm=llm,
                device=device
            )
        else:
            self.transformer1 = Transformer1(translator=translator, llm=llm, device=device)

        # Obtain transformer2
        if pretrained_transformer2_path:
            self.transformer2: Transformer2 = torch.load(pretrained_transformer2_path)
        else:
            self.transformer2 = Transformer2(translator=translator, llm=llm, device=device)
