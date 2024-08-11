import torch.nn as nn
from custom_transformers.base_transformer import BaseTransformer


class Transformer2(BaseTransformer):
    def __init__(self):
        super(Transformer2, self).__init__(model_name="transformer_2")

        # TODO: Define layers for the transformer model

        self.transform_layer = nn.Linear(512, 512)

    def forward(self, hidden_states):
        """
        Define the forward pass for Transformer2.
        """
        return self.transform_layer(hidden_states)

    def train_model(self):
        """
        Implement the training process for Transformer2.
        This method should train the model and save the state dict.
        """
        # TODO: implement the training
        pass
