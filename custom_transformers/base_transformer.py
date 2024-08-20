import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC
from my_datasets.create_datasets import create_transformer1_dataset, create_transformer2_dataset
from torch.utils.data import Dataset
import torch.nn.functional as F


MAX_TANSOR_LENGTH = 15
BATCH_SIZE = 64
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.15
EPOCHS = 3
DROPOUT = 0.1
INPUT_SIZE = 1024
OUTPUT_SIZE = 512


class BaseTransformer(nn.Module, ABC):
    def __init__(self, model_name: str, 
                 translator=None, llm=None,
                 input_size=INPUT_SIZE,
                 output_size=OUTPUT_SIZE, 
                 num_layers=2, 
                 num_heads=1, 
                 dim_feedforward=128, 
                 dropout=DROPOUT, 
                 activation=F.relu):
        
        super(BaseTransformer, self).__init__()
        self.model_name = model_name
        if "transformer_1" in model_name:
            self.dataset_path = '../my_datasets/transformer1_dataset.pt'
        else:
            self.dataset_path = '../my_datasets/transformer2_dataset.pt'

        self.model_path = f'../models/{model_name}.pth'
        self.translator = translator
        self.llm = llm


        # TODO - Check it
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.encoder_layers.self_attn.batch_first = True
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)

        # Linear layer to map to the target hidden size
        self.fc = nn.Linear(input_size, output_size)
    
    
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
                    create_transformer1_dataset(self.translator, self.llm, self.dataset_path)
                elif "transformer_2" in self.model_name:
                    create_transformer2_dataset(self.translator, self.llm, self.dataset_path)
                else:
                    raise ValueError(f"Unknown transformer model name: {self.model_name}")

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
        torch.save(self.state_dict(), f"../models/{self.model_name}_state_dict.pth")


    def forward(self, hidden_states, src_key_padding_mask=None):
        encoded = self.transformer_encoder(hidden_states, src_key_padding_mask=src_key_padding_mask)
        output = self.fc(encoded)
        return output

    @staticmethod
    def load_dataset(path, batch_size=32, shuffle=True):
        loaded_data = torch.load(path)
        dataset = CustomHiddenStateDataset(loaded_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class CustomHiddenStateDataset(Dataset):
    def __init__(self, data):
        self.input_data = data['input']
        self.target_data = data['target']

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        target_sample = self.target_data[idx]
        return input_sample, target_sample
