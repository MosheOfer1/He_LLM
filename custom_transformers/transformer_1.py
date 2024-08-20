import torch.nn as nn
from custom_transformers.base_transformer import BaseTransformer
from llm.llm_integration import LLMWrapper
from translation.translator import Translator


class Transformer1(BaseTransformer):
    def __init__(self, translator: Translator, llm: LLMWrapper, hidden_dim=1024, output_size=1024):
        """
        Initialize the Transformer1 model.

        :param translator: The translator instance used.
        :param llm: The LLM instance used.
        :param hidden_dim: Dimension of the hidden layer(s) in Transformer1.
        """
        # Determine input and output dimensions based on the translator and LLM
        input_dim = translator.src_to_target_model.config.hidden_size
        output_dim = llm.model.config.hidden_size

        # Generate a model name that includes the translator and LLM names
        model_name = f"transformer_1_{translator.src_to_target_translator_model_name.replace('/','_')}_to_{llm.model.config.name_or_path.replace('/','_')}"

        super(Transformer1, self).__init__(model_name=model_name, translator=translator, llm=llm, output_size=output_size)

        # Define the layers of the transformer model
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
