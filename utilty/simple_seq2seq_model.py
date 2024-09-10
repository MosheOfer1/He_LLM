import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_datasets.create_datasets import load_and_create_dataset


class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768, hidden_dim=512, num_layers=2):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, labels=None):
        encoder_output, _ = self.encoder(input_ids)
        decoder_output, _ = self.decoder(encoder_output)
        output = self.fc(decoder_output)
        return output

    @classmethod
    def load_model(cls, model_name: str):
        """
        :param model_name: Name of the model file (without the .pth extension).
        :return: The loaded Transformer model (Transformer1 or Transformer2).
        """
        if not model_name.endswith('.pth') and not model_name.endswith('.pt'):
            model_name += '.pth'
        # Construct the full path to the model file
        model_path = f"models/{model_name}"

        # Initialize the appropriate Transformer model
        model = Seq2SeqModel()

        # Load the model state dictionary from the saved file
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        # Set the model to evaluation mode
        model.eval()

        print(f"Model '{model_name}' loaded from {model_path}")
        return model


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        outputs = model(input_ids)

        # Calculate the loss
        loss_fct = nn.MSELoss()
        loss = loss_fct(outputs, labels)

        return (loss, outputs) if return_outputs else loss


file_path = 'my_datasets/'

train_dataset_path = file_path + input("Enter train dataset name")
test_dataset_path = file_path + input("Enter test dataset name")
while True:
    try:
        # Try to load the datasets
        train_ds = load_and_create_dataset(train_dataset_path)
        test_ds = load_and_create_dataset(test_dataset_path)
        print(f"Datasets loaded from {train_dataset_path}")
        break
    except FileNotFoundError as e:
        print(e)


# Initialize the model
model = Seq2SeqModel()

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
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_ds,               # training dataset
    eval_dataset=test_ds
)

# Train the model
trainer.train()

model_path = f'models/try_model.pth'

# Optionally save the trained model
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")

