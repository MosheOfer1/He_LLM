import torch.nn as nn
from abc import ABC


class BaseTransformer(nn.Module, ABC):
    def __init__(self, model_name: str,
                 translator=None, llm=None):

        super(BaseTransformer, self).__init__()
        self.model_name = model_name
        self.model_path = f'../models/{model_name}.pth'
        self.translator = translator
        self.llm = llm

