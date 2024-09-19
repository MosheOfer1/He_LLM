import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
import matplotlib.pyplot as plt
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_transformers.custom_trainer_trans1 import *
from transformers import Seq2SeqTrainingArguments
from custom_transformers.base_transformer import BaseTransformer
from llm.llm_wrapper import LLMWrapper
from translation.translator import Translator


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
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True).to(device)
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

    def encode(self, input_seq, input_mask=None):
        input_seq = self.input_projection(input_seq)
        input_seq = input_seq + self.positional_encoding[:, :input_seq.size(1), :].to(input_seq.device)

        # Pass the attention mask to the encoder
        memory = self.encoder(input_seq, src_key_padding_mask=input_mask)
        return memory

    def decode(self, target_seq, memory, target_mask=None, memory_mask=None):
        """
        Decoding step using the transformer decoder.

        :param target_seq: Target sequence tensor of shape (batch_size, seq_len, hidden_dim).
        :param memory: Memory tensor from the encoder of shape (batch_size, seq_len, hidden_dim).
        :param target_mask: Attention mask for the target sequence.
        :param memory_mask: Attention mask for the memory (source sequence).
        :return: The decoded output.
        """
        # Add positional encoding to the target sequence
        target_seq = target_seq + self.positional_encoding[:, :target_seq.size(1), :].to(target_seq.device)

        # Decode using the Transformer Decoder with attention masks
        output = self.decoder(tgt=target_seq, memory=memory, tgt_key_padding_mask=target_mask,
                              memory_key_padding_mask=memory_mask)
        return output

    def forward(self, input_ids, labels=None, input_mask=None, label_mask=None):
        """
        Forward pass through the Transformer1 model.

        :param input_ids: Tensor of shape (batch_size, seq_len, input_dim), representing the input sequence.
        :param labels: Optional tensor of shape (batch_size, seq_len, output_dim), representing the target sequence.
        :param input_mask: Attention mask for the input sequence.
        :param label_mask: Attention mask for the labels (target sequence).
        :return: The output sequence tensor.
        """

        # Move input_ids and masks to the correct device
        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device) if input_mask is not None else None
        label_mask = label_mask.to(self.device) if label_mask is not None else None

        # Encode the input sequence to get memory
        memory = self.encode(input_ids, input_mask=input_mask)

        # Initialize the output tensor
        batch_size, tgt_len = input_ids.size(0), labels.size(1) if labels is not None else input_ids.size(1)
        outputs = torch.zeros(batch_size, tgt_len, self.output_dim).to(self.device)

        # Initialize the target sequence with the EOS vector for the first token
        input_token = self.eos_vector_output.unsqueeze(0).expand(batch_size, -1).to(self.device)

        # Iterate over the sequence to generate each token step by step
        for t in range(tgt_len):
            # Create a sequence tensor for the current target token
            target_seq = input_token.unsqueeze(1)  # Shape: (batch_size, 1, output_dim)

            # Slice the label_mask for the current token (t-th step)
            target_mask_step = label_mask[:, t:t + 1] if label_mask is not None else None

            # Decode using the transformer decoder
            decoder_output = self.decode(target_seq, memory, target_mask=target_mask_step, memory_mask=input_mask)

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
            output_dir='my_datasets/transformer1_training',
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=32,
            weight_decay=0.01,
            save_total_limit=1,
            save_strategy="epoch",
            num_train_epochs=epochs,
            predict_with_generate=False,  # Not generating text, so disable generation
            logging_dir='my_datasets/logs',
        )

        # Initialize the Seq2SeqTrainer
        trainer = CustomTrainer(
            model=self.to(self.device),  # Pass the current model instance
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=None,  # No tokenizer since we're working with raw vectors
            data_collator=lambda x: collate_fn(x, max_seq_len=128, device=self.device)
        )

        self.printTrainableParams()

        # Train the model
        trainer.train()

        # Optionally save the trained model
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        torch.save(self.state_dict(), self.model_path)

        print(f"Model saved to {self.model_path}")
        self.evaluate_model(trainer, test_dataset)
        self.plot_loss(trainer)

    def printTrainableParams(self):
        total_params = sum(p.numel() for p in self.parameters())  # Total number of parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)  # Trainable parameters

        # Print the parameter names for the trainable parameters
        print("Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

        # Print the total number of parameters and trainable parameters
        print(f"\nTotal parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

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

        device = model.module.device if hasattr(model, 'module') else model.device

        # Load the model state dictionary from the saved file
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

        # Set the model to evaluation mode
        model.eval()

        print(f"Model '{model_name}' loaded from {model_path}")
        return model
