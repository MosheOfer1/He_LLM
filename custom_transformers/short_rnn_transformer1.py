import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_transformers.base_transformer import BaseTransformer
from llm.llm_integration import LLMWrapper
from translation.translator import Translator


class Transformer1(BaseTransformer):
    def __init__(self, 
                 translator, 
                 llm, 
                 model_name=None, 
                 input_dim=1024, 
                 output_dim=768, 
                 hidden_dim=512, 
                 num_layers=2,
                 device='cpu'):
        
        self.device = device
        
        translator = translator.to(device)
        llm = llm.to(device)
        
        # Determine input and output dimensions based on the translator and LLM
        self.input_dim = translator.src_to_target_model.config.hidden_size
        self.output_dim = llm.model.config.hidden_size

        # Generate a model name if not provided
        if not model_name:
            model_name = f"long_short_rnn_transformer1_{translator.src_to_target_translator_model_name.replace('/', '_')}_to_{llm.model.config.name_or_path.replace('/', '_')}"

        super(Transformer1, self).__init__(model_name=model_name)

        self.encoder = RNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        self.decoder = RNNDecoder(output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        self.output_dim = output_dim

        # Set model name and path for saving
        self.model_name = model_name if model_name else "seq2seq_model"
        self.model_path = f"models/{self.model_name}.pth"

    def forward(self, input_ids, labels=None, teacher_forcing_ratio=0.01):
        
        input_ids = input_ids.to(self.device)
        
        batch_size, src_len, _ = input_ids.size()
        tgt_len = labels.size(1) if labels is not None else src_len
        outputs = torch.zeros(batch_size, tgt_len, self.output_dim).to(self.device)

        encoder_outputs, hidden = self.encoder(input_ids)
        input = labels[:, 0, :] if labels is not None else torch.zeros(batch_size, self.output_dim).to(self.device)

        for t in range(1, tgt_len):
            input = input.clone().detach()
            output, hidden = self.decoder(input.unsqueeze(1), hidden)
            outputs[:, t, :] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = labels[:, t, :] if teacher_force and labels is not None else outputs[:, t, :]

        return outputs

    def train_model(self, train_dataset=None, test_dataset=None, epochs=8):
        training_args = TrainingArguments(
            output_dir='./results',  # output directory
            num_train_epochs=epochs,  # total number of training epochs
            per_device_train_batch_size=8,  # batch size per device during training
            save_steps=10_000,  # number of updates steps before saving checkpoint
            save_total_limit=2,  # limit the total amount of checkpoints
        )

        trainer = CustomTrainer(
            model=self,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        trainer.train()

        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        torch.save(self.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    @classmethod
    def load_model(cls, model_name: str, translator: Translator, llm: LLMWrapper, device='cpu'):
        """
        Load a Transformer model (either Transformer1 or Transformer2) from the ../models directory using the model name.

        :param model_name: Name of the model file (without the .pth extension).
        :param translator: The translator instance used in the Transformer model.
        :param llm: The LLM instance used in the Transformer model.
        :param device: The device to load the model onto (cpu or cuda).
        :return: The loaded Transformer model (Transformer1 or Transformer2).
        """
        if not model_name.endswith('.pth') and not model_name.endswith('.pt'):
            model_name += '.pth'
        # Construct the full path to the model file
        model_path = f"models/{model_name}"

        # Initialize the appropriate Transformer model
        model = Transformer1(model_name=model_name, translator=translator, llm=llm, device=device)

        try:
            # Load the model state dictionary from the saved file
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
            print(f"Model '{model_name}' loaded from {model_path}")
        except:
            print(f"Model '{model_name}' wasn't found from {model_path}, created new one")

        # Set the model to evaluation mode
        model.eval()

        return model

    def infer(self, src, max_len=50, start_token=None):
        self.eval()
        src = src.to(self.device)
        
        batch_size = src.size(0)
        inputs = torch.zeros(batch_size, 1, self.output_dim).to(self.device)
        if start_token is not None:
            inputs[:, 0, :] = start_token.to(self.device)

        encoder_outputs, hidden = self.encoder(src)
        generated_seq = torch.zeros(batch_size, max_len, self.output_dim).to(self.device)

        for t in range(max_len):
            output, hidden = self.decoder(inputs, hidden)
            generated_seq[:, t, :] = output.squeeze(1)
            inputs = output

        return generated_seq


class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        outputs, hidden = self.rnn(x)
        return outputs, hidden


class RNNDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=2):
        super(RNNDecoder, self).__init__()
        self.rnn = nn.LSTM(output_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        outputs, hidden = self.rnn(x, hidden)
        predictions = self.fc(outputs)
        return predictions, hidden


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get("input_ids").to(model.device)
        labels = inputs.get("labels").to(model.device)
        outputs = model(input_ids, labels)

        loss_fct = nn.MSELoss()
        loss = loss_fct(outputs, labels)

        return (loss, outputs) if return_outputs else loss
