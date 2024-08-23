import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from custom_transformers.base_transformer import BaseTransformer
from my_datasets.create_datasets import create_transformer1_dataset


class Transformer1(BaseTransformer):
    def __init__(self, translator, llm, model_name=None, hidden_dim=256, num_layers=1):
        """
        Initialize the Transformer1 RNN-based model.

        :param translator: The translator instance used.
        :param llm: The LLM instance used.
        """
        # Determine input and output dimensions based on the translator and LLM
        self.input_dim = translator.src_to_target_model.config.hidden_size
        self.output_dim = llm.model.config.hidden_size

        # Generate a model name if not provided
        if not model_name:
            model_name = f"rnn_transformer1_{translator.src_to_target_translator_model_name.replace('/', '_')}_to_{llm.model.config.name_or_path.replace('/', '_')}"

        super(Transformer1, self).__init__(model_name=model_name, translator=translator, llm=llm)

        # Define the encoder and decoder
        self.encoder = RNNEncoder(input_dim=self.input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.decoder = RNNDecoder(output_dim=self.output_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def train_model(self, train_dataset=None, test_dataset=None, epochs=8):
        if not train_dataset:
            train_dataset, test_dataset = create_transformer1_dataset(self.translator, self.llm, '../my_datasets/')

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0

            for batch in train_loader:
                src = batch['input_ids']  # Assuming batch is a dictionary with 'inputs' and 'targets'
                tgt = batch['labels']

                self.optimizer.zero_grad()

                # Forward pass
                output = self(src, tgt)

                # Calculate loss
                loss = self.criterion(output, tgt)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Print epoch loss
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")

            # Evaluate on test dataset
            self.evaluate(test_loader)

        # Optionally save the trained model
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        torch.save(self.state_dict(), self.model_path)

    def evaluate(self, test_loader):
        self.eval()
        test_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                src = batch['input_ids']  # Assuming batch is a dictionary with 'inputs' and 'targets'
                tgt = batch['labels']

                output = self(src, tgt)
                loss = self.criterion(output, tgt)
                test_loss += loss.item()

        print(f"Test Loss: {test_loss / len(test_loader)}")

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, src_len, _ = src.size()
        tgt_len = tgt.size(1)
        outputs = torch.zeros(batch_size, tgt_len, self.output_dim).to(src.device)

        encoder_outputs, hidden = self.encoder(src)
        input = tgt[:, 0, :]  # Initial decoder input

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input.unsqueeze(1), hidden)
            outputs[:, t, :] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = tgt[:, t, :] if teacher_force else output.squeeze(1)

        return outputs

    def infer(self, src, max_len=50, start_token=None):
        """
        Generate a sequence based on the source sequence.

        :param src: The input source sequence (tensor of shape [batch_size, src_len, input_dim]).
        :param max_len: Maximum length of the generated sequence.
        :param start_token: Initial token for the decoder (optional).
        :return: The generated sequence.
        """
        self.eval()
        batch_size = src.size(0)
        inputs = torch.zeros(batch_size, 1, self.output_dim).to(src.device)  # Initialize with start token if available
        if start_token is not None:
            inputs[:, 0, :] = start_token

        # Pass the source sequence through the encoder
        encoder_outputs, hidden = self.encoder(src)
        generated_seq = torch.zeros(batch_size, max_len, self.output_dim).to(src.device)

        for t in range(max_len):
            output, hidden = self.decoder(inputs, hidden)
            generated_seq[:, t, :] = output.squeeze(1)

            # Use the output as input for the next timestep
            inputs = output

        return generated_seq


class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        outputs, hidden = self.rnn(x)
        return outputs, hidden


class RNNDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(RNNDecoder, self).__init__()
        self.rnn = nn.GRU(output_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        outputs, hidden = self.rnn(x, hidden)
        predictions = self.fc(outputs)
        return predictions, hidden
