import torch
import torch.nn as nn
from abc import ABC

from llm.llm_integration import LLMWrapper
from translation.translator import Translator


class BaseTransformer(nn.Module, ABC):
    def __init__(self, model_name: str,
                 translator=None, llm=None):

        super(BaseTransformer, self).__init__()
        self.model_name = model_name
        if "transformer_1" in model_name:
            self.dataset_path = '../my_datasets/transformer1_dataset.pt'
        else:
            self.dataset_path = '../my_datasets/transformer2_dataset.pt'

        self.model_path = f'../models/{model_name}.pth'
        self.translator = translator
        self.llm = llm

