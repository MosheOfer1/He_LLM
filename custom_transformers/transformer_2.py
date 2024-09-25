import torch.nn as nn

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

    def forward(self, hidden_states, attention_mask=None):
        """
        Define the forward pass for Transformer2.
        :param hidden_states: Input hidden states (batch_size, seq_len, hidden_dim).
        :param attention_mask: Attention mask (batch_size, seq_len), where 1 indicates valid positions and 0 indicates padded positions.
        """
        hidden_states = hidden_states.to(self.device)

        if attention_mask is not None:
            # Expand attention_mask to match the hidden_states shape if needed
            attention_mask = attention_mask.unsqueeze(-1).to(self.device)  # Shape: (batch_size, seq_len, 1)

        # Layer 1 processing
        x = self.layer1(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)

        # Layer 2 processing
        x = self.layer2(x)

        # Apply attention mask if provided
        if attention_mask is not None:
            x = x * attention_mask  # Mask out the padded positions

        return x
