import torch.nn as nn
from custom_transformers.base_transformer import BaseTransformer


class Transformer1(BaseTransformer):
    def __init__(self):
        super(Transformer1, self).__init__(model_name="transformer_1")

        # TODO: Define layers for the transformer model

        self.transform_layer = nn.Linear(512, 512)

    def forward(self, hidden_states):
        """
        Define the forward pass for Transformer1.
        """
        return self.transform_layer(hidden_states)

    def train_model(self):
        """
        Implement the training process for Transformer1.
        This method should train the model and save the state dict.
        """
        # TODO: implement the training
        pass
