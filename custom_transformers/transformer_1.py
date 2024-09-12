import torch.nn as nn
import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch.nn.functional as F

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

from custom_transformers.base_transformer import BaseTransformer
from llm.llm_wrapper import LLMWrapper
from translation.translator import Translator


def create_positional_encoding(max_seq_len, hidden_dim):
    # Initialize the positional encoding matrix
    position = torch.arange(0, max_seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-torch.log(torch.tensor(10000.0)) / hidden_dim))

    positional_encoding = torch.zeros(max_seq_len, hidden_dim)
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)

    return positional_encoding.unsqueeze(0)  # Shape (1, max_seq_len, hidden_dim)


class BlackBox:
    def __init__(self, llm: LLMWrapper):
        self.llm = llm


class Transformer1(BaseTransformer):
    def __init__(self, translator: Translator,
                 llm: LLMWrapper, 
                 model_name=None, 
                 nhead=2,
                 num_layers=2,
                 max_seq_len=128,
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

        super(Transformer1, self).__init__(model_name=model_name)

        """ Define the layers of the transformer model  """
        # Input projection to align translator's hidden states to the model's hidden dimension
        self.input_projection = nn.Linear(self.input_dim, hidden_dim).to(device)
        # Initializing the positional encoding as a learnable parameter in the model max seq len is 512
        self.positional_encoding = create_positional_encoding(max_seq_len, hidden_dim).to(device)

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
        self.eos_vector_output = torch.ones(llm.model.config.hidden_size).to(device)

        self.black_box = BlackBox(llm)

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

    def forward(self, input_ids, labels=None):
        """
        Forward pass through the Transformer1 model.

        :param input_ids: Tensor of shape (batch_size, seq_len, input_dim), representing the input sequence.
        :param labels: Optional tensor of shape (batch_size, seq_len, output_dim), representing the target sequence.
        :param mse_threshold: MSE loss threshold for early stopping if the error becomes too small.
        :return: The output sequence tensor.
        """

        # Move input_ids to the correct device
        input_ids = input_ids.to(self.device)

        # Encode the input sequence to get memory
        memory = self.encode(input_ids)

        # Initialize the output tensor
        batch_size, tgt_len = input_ids.size(0), labels.size(1) if labels is not None else input_ids.size(1)
        outputs = torch.zeros(batch_size, tgt_len, self.output_dim).to(self.device)

        # Initialize the target sequence with the EOS vector for the first token
        input_token = self.eos_vector_output.unsqueeze(0).expand(batch_size, -1).to(self.device)

        # Iterate over the sequence to generate each token step by step
        for t in range(tgt_len):
            # Create a sequence tensor for the current target token
            target_seq = input_token.unsqueeze(1)  # Shape: (batch_size, 1, output_dim)

            # Decode using the transformer decoder
            decoder_output = self.decode(target_seq, memory)

            # Project the decoder output to the desired output dimension
            output_token = self.output_projection(decoder_output[:, -1, :])  # Shape: (batch_size, output_dim)

            # Store the generated token in the output sequence
            outputs[:, t, :] = output_token

            # If labels are provided, calculate the loss between predicted and true tokens
            if labels is not None:
                true_token = labels[:, t, :].to(self.device)

                # Update the input token for the next time step with teacher forcing
                input_token = true_token.clone().detach()
            else:
                # In the absence of labels, use the predicted output as the next input token
                input_token = output_token.clone().detach()

        return outputs

    def train_model(self, train_dataset, test_dataset, epochs=5):
        training_args = Seq2SeqTrainingArguments(
            output_dir='../my_datasets',
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=1,
            save_strategy="epoch",
            num_train_epochs=epochs,
            predict_with_generate=False,  # Not generating text, so disable generation
            logging_dir='../my_datasets/logs',
        )

        # Print trainable layers and parameters
        print_model_parameters(self)

        # Initialize the Seq2SeqTrainer
        trainer = CustomTrainer(
            model=self.to(self.device),  # Pass the current model instance
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=None,  # No tokenizer since we're working with raw vectors
            data_collator=lambda x: collate_fn(x, max_seq_len=128)
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
    def plot_loss(trainer, save_path='images/loss_plot.png'):
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

        # Save the plot to the specified file path
        plt.savefig(save_path)
        plt.close()  # Close the figure to avoid displaying it

    @staticmethod
    def evaluate_model(trainer, test_dataset):
        # Evaluate the model on the test dataset
        eval_results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Evaluation Results: {eval_results}")
        return eval_results

    @classmethod
    def load_model(cls, model_name: str, translator: Translator, llm: LLMWrapper, device="cpu"):
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
        model_path = model_name

        # Initialize the appropriate Transformer model
        model = Transformer1(translator=translator, llm=llm, device=device)

        # Load the model state dictionary from the saved file
        model.load_state_dict(torch.load(model_path, map_location=torch.device(model.device)))

        # Set the model to evaluation mode
        model.eval()

        print(f"Model '{model_name}' loaded from {model_path}")
        return model


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


def logits_from_first_layer(llm, hidden_states):
    # Inject the noised hidden states back into the model
    llm.inject_hidden_states(hidden_states)
    # Get the logits after injecting the hidden states
    batch_size = hidden_states.shape[0]
    token_num = hidden_states.shape[1]
    llm_outputs = llm.get_output_by_using_dummy(token_num=token_num, batch_size=batch_size)
    predicted_logits = llm_outputs.logits
    return predicted_logits


def _calculate_KL_div(llm, outputs, label):
    predicted_logits = logits_from_first_layer(llm, outputs)
    true_logits = logits_from_first_layer(llm, label)

    # Convert logits to probabilities using softmax
    predicted_probs = F.softmax(predicted_logits, dim=-1)
    original_probs = F.softmax(true_logits, dim=-1)

    # Calculate KL divergence between the original and predicted logits
    kl_div = F.kl_div(original_probs.log(), predicted_probs, reduction='batchmean')

    return kl_div


class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get("input_ids").to(model.device)
        labels = inputs.get("labels").to(model.device)
        outputs = model(input_ids, labels)
        llm = model.black_box.llm
        loss = _calculate_KL_div(llm, outputs, labels)

        return (loss, outputs) if return_outputs else loss


def pad_hidden_states(hidden_states, max_len):
    """Pad hidden states to a fixed length with ones."""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    if seq_len < max_len:
        pad_size = max_len - seq_len
        padding = torch.ones((batch_size, pad_size, hidden_dim))  # Pad with ones
        return torch.cat([hidden_states, padding], dim=1)  # Concatenate along the seq_len dimension
    hidden_states = hidden_states[:, :max_len - 1, :]  # Truncate if necessary
    padding = torch.ones((batch_size, 1, hidden_dim))  # Add ones for EOS
    return torch.cat([hidden_states, padding], dim=1)


def collate_fn(batch, max_seq_len=None):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Determine the maximum sequence length in the batch
    max_input_len = max([x.size(0) for x in input_ids])
    max_label_len = max([x.size(0) for x in labels])

    # Set the max sequence length to pad if provided, otherwise use the batch max
    max_input_len = min(max_seq_len, max_input_len) if max_seq_len is not None else max_input_len
    max_label_len = min(max_seq_len, max_label_len) if max_seq_len is not None else max_label_len

    # Pad input_ids and labels using your custom pad_hidden_states function
    padded_input_ids = torch.stack([pad_hidden_states(x.unsqueeze(0), max_input_len).squeeze(0) for x in input_ids])
    padded_labels = torch.stack([pad_hidden_states(x.unsqueeze(0), max_label_len).squeeze(0) for x in labels])

    return {
        'input_ids': padded_input_ids,
        'labels': padded_labels
    }
