import os

import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    def train_model(self, train_dataset=None, test_dataset=None, epochs=8):
        if not train_dataset:
            train_dataset, test_dataset = create_transformer1_dataset(self.translator, self.llm, 'my_datasets/')
            # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',  # output directory
            num_train_epochs=5,  # total number of training epochs
            per_device_train_batch_size=8,  # batch size per device during training
            save_steps=10_000,  # number of updates steps before saving checkpoint
            save_total_limit=2,  # limit the total amount of checkpoints
        )

        # Initialize the Trainer
        trainer = CustomTrainer(
            model=self,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=test_dataset
        )

        # Train the model
        trainer.train()

        # Optionally save the trained model
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        torch.save(self.state_dict(), self.model_path)

    def forward(self, input_ids, labels=None, teacher_forcing_ratio=0.5):
        batch_size, src_len, _ = input_ids.size()
        tgt_len = labels.size(1) if labels is not None else src_len
        outputs = torch.zeros(batch_size, tgt_len, self.output_dim).to(input_ids.device)

        encoder_outputs, hidden = self.encoder(input_ids)
        input = labels[:, 0, :] if labels is not None else torch.zeros(batch_size, self.output_dim).to(input_ids.device)

        for t in range(1, tgt_len):
            input = input.clone().detach()

            output, hidden = self.decoder(input.unsqueeze(1), hidden)
            outputs[:, t, :] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = labels[:, t, :] if teacher_force and labels is not None else outputs[:, t, :]

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


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        outputs = model(input_ids)

        # Calculate the loss
        loss_fct = nn.MSELoss()  # Assuming regression task, modify if needed
        loss = loss_fct(outputs, labels)

        return (loss, outputs) if return_outputs else loss
