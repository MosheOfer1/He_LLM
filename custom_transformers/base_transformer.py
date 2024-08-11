import torch
import torch.nn as nn
import os
from abc import ABC
from custom_transformers.transformer_strategy import TransformerStrategy


class BaseTransformer(nn.Module, TransformerStrategy, ABC):
    def __init__(self, model_name: str):
        super(BaseTransformer, self).__init__()
        self.model_name = model_name
        self.model_path = f'models/{model_name}.pth'

    def load_or_train_model(self):
        """
        Load the model if it exists; otherwise, train and save it.
        """
        if os.path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))
            self.eval()  # Set the model to evaluation mode
        else:
            self.train_model()
            torch.save(self.state_dict(), self.model_path)

    def train_model(self):
        """
        Trigger the training process for the transformer.
        This should be overridden by specific transformer implementations.
        """
        raise NotImplementedError("Subclasses should implement this method to train the model.")

    def transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Transform the hidden states using the model, ensuring the model is trained.

        :param hidden_states: The hidden states from the previous layer or model.
        :return: The transformed hidden states.
        """
        self.load_or_train_model()
        return self.forward(hidden_states)
