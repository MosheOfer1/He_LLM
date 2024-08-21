import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
import torch.nn.functional as F

from custom_transformers.base_transformer import BaseTransformer
from llm.llm_integration import LLMWrapper
from translation.translator import Translator


class Transformer1(BaseTransformer):
    def __init__(self, translator: Translator, llm: LLMWrapper, nhead=8, num_layers=6, max_seq_len=512):
        """
        Initialize the Transformer1 model.

        :param translator: The translator instance used.
        :param llm: The LLM instance used.
        """
        # Determine input and output dimensions based on the translator and LLM
        self.input_dim = translator.src_to_target_model.config.hidden_size
        self.output_dim = llm.model.config.hidden_size
        hidden_dim = self.output_dim
        # Generate a model name that includes the translator and LLM names
        model_name = f"transformer_1_{translator.src_to_target_translator_model_name.replace('/','_')}_to_{llm.model.config.name_or_path.replace('/','_')}"

        super(Transformer1, self).__init__(model_name=model_name, translator=translator, llm=llm)

        """ Define the layers of the transformer model  """
        # Input projection to align translator's hidden states to the model's hidden dimension
        self.input_projection = nn.Linear(self.input_dim, hidden_dim)
        # Initializing the positional encoding as a learnable parameter in the model max seq len is 512
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        # Transformer Decoder Layer
        decoder_layers = TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layers, num_layers=num_layers)
        # Output projection to align model's output to the LLM's required hidden dimension
        self.output_projection = nn.Linear(hidden_dim, self.output_dim)

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

    def forward(self, input_ids, labels=None, eos_token_id=2):
        """
        Forward pass through the Transformer1 model.

        :param input_ids: Input tensor of shape (batch_size, seq_len, input_dim).
        :param labels: Target tensor of shape (batch_size, seq_len, output_dim), optional.
        :param eos_token_id: ID of the EOS token to stop decoding.
        :return: The output of the model.
        """
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
            start_tokens = torch.zeros((batch_size, 1, self.output_dim), device=input_ids.device)
            generated_seq = start_tokens

            for _ in range(max_length):
                # Decode the current sequence
                decoder_output = self.decode(generated_seq, memory)
                # Project the decoder output to get the logits
                logits = self.output_projection(decoder_output)
                # Get the predicted token by taking the argmax of the logits (greedy decoding)
                next_token = logits[:, -1, :]  # Take the last time step's output
                next_token_id = next_token.argmax(dim=-1).unsqueeze(1)

                # If EOS token is predicted, stop decoding
                if eos_token_id is not None and (next_token_id == eos_token_id).all():
                    break

                # Append the predicted token to the sequence
                next_token_embedding = F.one_hot(next_token_id, num_classes=self.output_dim).float()
                generated_seq = torch.cat([generated_seq, next_token_embedding], dim=1)

            # The generated sequence is now the output
            output = generated_seq

        return output
