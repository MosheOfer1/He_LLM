import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_transformers.base_transformer import BaseTransformer
from llm.llm_wrapper import LLMWrapper
from translation.translator import Translator


class Transformer2(BaseTransformer):
    def __init__(self, 
                 translator: Translator, 
                 llm: LLMWrapper, 
                 hidden_dim=1024,
                 device='cpu'):
        """
        Initialize the Transformer2 model.

        :param translator: The translator instance used in the pipeline.
        :param llm: The LLM instance used.
        :param hidden_dim: Dimension of the hidden layer(s) in Transformer2.
        """
        
        print(f"Transformer2.__init__ - uses: {device}")
        
        self.device = device

        # Determine input and output dimensions based on the LLM and translator
        if hasattr(llm.model.config, 'word_embed_proj_dim'):
            # For models that have word_embed_proj_dim (like BART, T5)
            input_dim = llm.model.config.word_embed_proj_dim
        elif hasattr(llm.model.config, 'n_embd'):
            # For GPT-2 or models that use n_embd
            input_dim = llm.model.config.n_embd
        else:
            raise AttributeError(f"Unsupported model architecture: {llm.model.config.__class__.__name__}")

        output_dim = translator.target_to_src_model.config.hidden_size

        # Generate a model name that includes the translator and LLM names
        model_name = f"transformer_2_{llm.model.config.name_or_path.replace('/','_')}_to_{translator.target_to_src_translator_model_name.replace('/','_')}"

        super(Transformer2, self).__init__(model_name=model_name)

        # Define the layers of the transformer model
        self.layer1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.layer2 = nn.Linear(hidden_dim, output_dim).to(device)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_attention_mask, output_attention_mask):
        """
        Define the forward pass for Transformer2.
        :param hidden_states: Input hidden states (batch_size, input_seq_len, input_dim).
        :param input_attention_mask: Input attention mask (batch_size, input_seq_len), where 1 indicates valid positions
         and 0 indicates padded positions.
        :param output_attention_mask: Output attention mask (batch_size, output_seq_len), where 1 indicates valid
         positions and 0 indicates padded positions.
        """
        hidden_states = hidden_states.to(self.device)
        input_mask = input_attention_mask.to(self.device)
        output_mask = output_attention_mask.to(self.device)

        batch_size, input_seq_len, input_dim = hidden_states.shape
        output_seq_len = output_mask.shape[1]

        # Apply input mask
        input_mask = input_mask.unsqueeze(-1)  # Shape: (batch_size, input_seq_len, 1)
        hidden_states = hidden_states * input_mask

        # Layer 1 processing
        x = self.layer1(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)

        # Prepare for projection
        input_lengths = input_mask.sum(dim=1).squeeze(-1)  # Shape: (batch_size,)
        output_lengths = output_mask.sum(dim=1)  # Shape: (batch_size,)

        # Project from input sequence length to output sequence length
        projected_x = []
        for i in range(batch_size):
            valid_input_len = int(input_lengths[i].item())  # Convert to integer
            valid_output_len = int(output_lengths[i].item())  # Convert to integer

            valid_input = x[i, :valid_input_len]

            # Interpolate to match output length
            if valid_input.shape[0] > 1:
                interpolated = F.interpolate(valid_input.unsqueeze(0).transpose(1, 2),
                                             size=valid_output_len,
                                             mode='linear',
                                             align_corners=False)
                interpolated = interpolated.transpose(1, 2).squeeze(0)
            else:
                # If there's only one valid input, repeat it
                interpolated = valid_input.repeat(valid_output_len, 1)

            # Pad to full output length if necessary
            if valid_output_len < output_seq_len:
                padding = torch.zeros(output_seq_len - valid_output_len, interpolated.shape[1], device=self.device)
                interpolated = torch.cat([interpolated, padding], dim=0)

            projected_x.append(interpolated)

        x = torch.stack(projected_x)

        # Apply output mask
        output_mask = output_mask.unsqueeze(-1)  # Shape: (batch_size, output_seq_len, 1)
        x = x * output_mask

        # Layer 2 processing
        x = self.layer2(x)

        return x
