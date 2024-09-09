import torch.nn as nn
import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

from my_datasets.create_datasets import create_transformer1_dataset
from custom_transformers.base_transformer import BaseTransformer
from llm.llm_integration import LLMWrapper
from translation.translator import Translator


class Transformer1(BaseTransformer):
    def __init__(self, translator: Translator,
                 llm: LLMWrapper, 
                 model_name=None, 
                 nhead=8, 
                 num_layers=6,
                 max_seq_len=512,
                 device='cpu'):
        """
        Initialize the Transformer1 model.

        :param translator: The translator instance used.
        :param llm: The LLM instance used.
        """
        
        print(f"Transformer1.__init__ - uses: {device}")
        
        self.device = device
        
        # Determine input and output dimensions based on the translator and LLM
        self.input_dim = translator.src_to_target_model.config.hidden_size
        self.output_dim = llm.model.config.hidden_size
        hidden_dim = self.output_dim
        # Generate a model name that includes the translator and LLM names
        if not model_name:
            model_name = f"transformer_1_{translator.src_to_target_translator_model_name.replace('/', '_')}_to_{llm.model.config.name_or_path.replace('/', '_')}"

        super(Transformer1, self).__init__(model_name=model_name, 
                                           translator=translator, 
                                           llm=llm)

        """ Define the layers of the transformer model  """
        # Input projection to align translator's hidden states to the model's hidden dimension
        self.input_projection = nn.Linear(self.input_dim, hidden_dim).to(device)
        # Initializing the positional encoding as a learnable parameter in the model max seq len is 512
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim)).to(device)
        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead).to(device)
        self.encoder = TransformerEncoder(encoder_layers, num_layers=num_layers).to(device)
        # Transformer Decoder Layer
        decoder_layers = TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True).to(device)
        self.decoder = TransformerDecoder(decoder_layers, num_layers=num_layers).to(device)
        # Output projection to align model's output to the LLM's required hidden dimension
        self.output_projection = nn.Linear(hidden_dim, self.output_dim).to(device)

        # Define the EOS vector (e.g., a vector of zeros or a specific learned vector)
        self.eos_vector_input = torch.zeros(translator.src_to_target_model.config.hidden_size).to(device)
        self.eos_vector_output = torch.zeros(llm.model.config.hidden_size).to(device)

    def encode(self, input_seq):
        input_seq = self.input_projection(input_seq)
        input_seq = input_seq + self.positional_encoding[:, :input_seq.size(1), :]
        memory = self.encoder(input_seq)
        return memory

    def decode(self, target_seq, memory):
        """
        Decoding step using the transformer decoder.

        :param target_seq: Target sequence tensor of shape (batch_size, seq_len, hidden_dim).
        :param memory: Memory tensor from the encoder of shape (batch_size, seq_len, hidden_dim).
        :return: The decoded output.
        """
        # Add positional encoding to the target sequence
        target_seq = target_seq + self.positional_encoding[:, :target_seq.size(1), :]
        # Decode using the Transformer Decoder
        output = self.decoder(tgt=target_seq, memory=memory)
        return output

    def forward(self, input_ids, labels=None, mse_threshold=1e-4):
        """
        Forward pass through the Transformer1 model.

        :param mse_threshold:
        :param input_ids: Input tensor of shape (batch_size, seq_len, input_dim).
        :param labels: Target tensor of shape (batch_size, seq_len, output_dim), optional.
        :return: The output of the model.
        """
        input_ids = input_ids.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
            
        max_length = input_ids.size(1) + 5

        # Encode the input sequence using the encoder
        memory = self.encode(input_ids)

        if labels is not None:
            # Training mode
            decoder_input = labels[:, :-1]  # Shifted target sequence
            decoder_output = self.decode(decoder_input, memory)
            logits = self.output_projection(decoder_output)

            # Compute Mean Squared Error Loss (MSELoss)
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(logits, labels[:, 1:])
            # Return a dictionary with the loss
            return {"loss": loss}
        else:
            # Inference mode with autoregressive decoding
            batch_size = input_ids.size(0)
            # Initialize the decoder input with a start token (or a tensor of zeros)
            # Assuming the first token in the sequence as a start token.
            start_tokens = torch.zeros((batch_size, 1, self.output_dim))#, device=self.device)
            generated_seq = start_tokens.to(self.device)

            for _ in range(max_length):
                # Decode the current sequence
                decoder_output = self.decode(generated_seq, memory)
                # Project the decoder output to get the logits
                logits = self.output_projection(decoder_output)
                # Get the predicted token by taking the argmax of the logits (greedy decoding)
                next_token = logits[:, -1, :].to(self.device)  # Take the last time step's output
                # Calculate MSE with the EOS vector
                mse_loss = torch.nn.MSELoss()
                mse = mse_loss(next_token, self.eos_vector_output)

                # If MSE is below the threshold, stop decoding
                if mse < mse_threshold:
                    break

                # Append the predicted token to the sequence
                generated_seq = torch.cat([generated_seq, next_token.unsqueeze(1)], dim=1)

            # The generated sequence is now the output
            output = generated_seq

        return output

    def train_model(self, train_dataset=None, test_dataset=None, epochs=8):
        if not train_dataset:
            train_dataset, test_dataset = create_transformer1_dataset(self.translator, self.llm, '../my_datasets/')

        training_args = Seq2SeqTrainingArguments(
            output_dir='../my_datasets',
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs,
            predict_with_generate=False,  # Not generating text, so disable generation
            logging_dir='../my_datasets/logs',
        )

        # Print trainable layers and parameters
        print("Trainable Layers and Parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")
            else:
                print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad} (Not trainable)")

        # Initialize the Seq2SeqTrainer
        trainer = Seq2SeqTrainer(
            model=self.to(self.device),  # Pass the current model instance
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
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
        self.evaluate_model(trainer, test_dataset)
        self.plot_loss(trainer)

    @staticmethod
    def plot_loss(trainer):
        # Extract the logs from the trainer's state
        training_loss = trainer.state.log_history

        # Extract loss values for training and evaluation
        train_loss = [entry['loss'] for entry in training_loss if 'loss' in entry.keys()]
        eval_loss = [entry['eval_loss'] for entry in training_loss if 'eval_loss' in entry.keys()]

        # Plotting the loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(eval_loss, label='Evaluation Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def evaluate_model(trainer, test_dataset):
        # Evaluate the model on the test dataset
        eval_results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Evaluation Results: {eval_results}")
        return eval_results

    @classmethod
    def load_model(cls, model_name: str, translator: Translator, llm: LLMWrapper):
        """
        Load a Transformer model (either Transformer1 or Transformer2) from the ../models directory using the model name.

        :param model_name: Name of the model file (without the .pth extension).
        :param translator: The translator instance used in the Transformer model.
        :param llm: The LLM instance used in the Transformer model.
        :return: The loaded Transformer model (Transformer1 or Transformer2).
        """
        if not model_name.endswith('.pth') and not model_name.endswith('.pt'):
            model_name += '.pth'
        # Construct the full path to the model file
        model_path = f"../models/{model_name}"

        # Initialize the appropriate Transformer model
        model = Transformer1(translator=translator, llm=llm)

        # Load the model state dictionary from the saved file
        model.load_state_dict(torch.load(model_path, map_location=torch.device(model.device)))

        # Set the model to evaluation mode
        model.eval()

        print(f"Model '{model_name}' loaded from {model_path}")
        return model
