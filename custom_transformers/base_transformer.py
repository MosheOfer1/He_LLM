import os

import torch
import torch.nn as nn
from abc import ABC
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from my_datasets.create_datasets import create_transformer1_dataset, create_transformer2_dataset
from torch.utils.data import Dataset


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

    def train_model(self, train_dataset: 'Seq2SeqDataset' = None):
        if not train_dataset:
            train_dataset = create_transformer1_dataset(self.translator, self.llm, self.dataset_path)

        training_args = Seq2SeqTrainingArguments(
            output_dir='../my_datasets',
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=False,  # Not generating text, so disable generation
            logging_dir='../my_datasets/logs',
        )

        # Initialize the Seq2SeqTrainer
        trainer = Seq2SeqTrainer(
            model=self,  # Pass the current model instance
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            # Use the same dataset for evaluation, can be replaced with an actual eval dataset
            tokenizer=None,  # No tokenizer since we're working with raw vectors
            data_collator=None,  # Custom data collator if needed, else can be left as None
        )

        # Train the model
        trainer.train()

        # Optionally save the trained model
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        torch.save(self.state_dict(), self.model_path)

        print(f"Model saved to {self.model_path}")


class Seq2SeqDataset(Dataset):
    def __init__(self, inputs, targets):
        """
        inputs: Tensor of shape (num_samples, seq_len, hidden_state_dim)
        targets: Tensor of shape (num_samples, seq_len, hidden_state_dim)
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'labels': self.targets[idx]
        }
