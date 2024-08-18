import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC
from my_datasets.create_datasets import create_transformer1_dataset, create_transformer2_dataset


class BaseTransformer(nn.Module, ABC):
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
                if "transformer_1" in self.model_name:
                    dataset = create_transformer1_dataset(self.translator, self.llm)
                elif "transformer_2" in self.model_name:
                    dataset = create_transformer2_dataset(self.translator, self.llm)
                else:
                    raise ValueError(f"Unknown transformer model name: {self.model_name}")
                torch.save(dataset, self.dataset_path)

            # Load the DataLoader of the right model
            train_loader = BaseTransformer.load_dataset(self.dataset_path)

            self.train_model(train_loader)
            torch.save(self.state_dict(), self.model_path)

    def train_model(self, train_loader, num_epochs=10, learning_rate=1e-4):
        """
        Implement the training process for Transformer1.
        This method should train the model and save the state dict.

        :param train_loader: DataLoader providing training data (input-output pairs).
        :param num_epochs: Number of epochs to train the model.
        :param learning_rate: Learning rate for the optimizer.
        """
        criterion = nn.MSELoss()  # Loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            for batch in train_loader:
                inputs, targets = batch

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs)

                # Compute loss
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Save the trained model with a name reflecting the translator and LLM used
        torch.save(self.state_dict(), f"{self.model_name}_state_dict.pth")

    def transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Transform the hidden states using the model, ensuring the model is trained.

        :param hidden_states: The hidden states from the previous layer or model.
        :return: The transformed hidden states.
        """
        self.load_or_train_model()
        return self.forward(hidden_states)

    def forward(self, hidden_states):
        """
        Define the forward pass for Transformer1/2.
        """
        x = self.layer1(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    @staticmethod
    def load_dataset(path, batch_size=32, shuffle=True):
        """
        Utility function to load a dataset from a given path.
        """
        dataset = torch.load(path)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
