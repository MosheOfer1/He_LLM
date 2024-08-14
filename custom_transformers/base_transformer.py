import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC

from custom_transformers.transformer_strategy import TransformerStrategy
from datasets.create_datasets import create_transformer1_dataset, create_transformer2_dataset


def load_dataset(path, batch_size=32, shuffle=True):
    """
    Utility function to load a dataset from a given path.
    """
    dataset = torch.load(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class BaseTransformer(nn.Module, TransformerStrategy, ABC):
    def __init__(self, model_name: str, translator=None, llm=None):
        super(BaseTransformer, self).__init__()
        self.model_name = model_name
        self.model_path = f'models/{model_name}.pth'
        self.dataset_path = f'datasets/generated_datasets/{model_name}_dataset.pth'
        self.translator = translator
        self.llm = llm

    def load_or_train_model(self):
        """
        Load the model if it exists; otherwise, train and save it.
        """
        if os.path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))
            self.eval()  # Set the model to evaluation mode
        else:
            # Check if the dataset exists, if not create it
            if not os.path.exists(self.dataset_path):
                file_name = "hebrew_sentences.csv"
                if "transformer_1" in self.model_name:
                    dataset = create_transformer1_dataset(self.translator, self.llm, file_name)
                elif "transformer_2" in self.model_name:
                    dataset = create_transformer2_dataset(self.translator, self.llm, file_name)
                else:
                    raise ValueError(f"Unknown transformer model name: {self.model_name}")
                torch.save(dataset, self.dataset_path)

            # Load the DataLoader of the right model
            train_loader = load_dataset(self.dataset_path)

            self.train_model(train_loader)
            torch.save(self.state_dict(), self.model_path)

    def train_model(self, train_loader, num_epochs=10, learning_rate=1e-4):
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
