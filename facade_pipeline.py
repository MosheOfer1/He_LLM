from llm.llm_integration import LLMIntegration
from custom_transformers.transformer_1 import Transformer1
from custom_transformers.transformer_2 import Transformer2
from translation.translator import Translator


class Pipeline:
    def __init__(self, translator: Translator, llm: LLMIntegration, use_transformer_1: bool = True, use_transformer_2: bool = True):
        """
        Initialize the Pipeline with optional transformers and LLM integration.

        :param translator: The translator component for handling language translation.
        :param llm: The LLM integration component.
        :param use_transformer_1: Whether to use the first transformer in the pipeline.
        :param use_transformer_2: Whether to use the second transformer in the pipeline.
        """
        # Initialize the translator
        self.translator = translator

        # Initialize the LLMIntegration
        self.llm = llm

        # Initialize the transformers based on user preference
        self.transformer_1 = Transformer1(self.translator, self.llm) if use_transformer_1 else None
        self.transformer_2 = Transformer2(self.translator, self.llm) if use_transformer_2 else None

    def process_text(self, text: str) -> str:
        """
        Processes the text through the pipeline, transforming it into hidden states and back,
        depending on the state of transformers.
        """
        # Step 1: Translate to target language
        if self.transformer_1:
            data = self.translator.translate_to_en_returns_hidden_states(text)

            # Step 2: Pass through Transformer 1 (if enabled)
            data = self.transformer_1.transform(data)
        else:
            data = self.translator.translate_to_target(text)

        # Step 3: Pass through the LLM
        if self.transformer_1:
            data = self.llm.inject_input_embeddings(data)  # Data is a Tensor, returns Tensor
        else:
            data = self.llm.process_text_input(data)  # Data is text, returns Tensor

        if self.transformer_2:
            # Step 4: Pass through Transformer 2 (if enabled)
            data = self.transformer_2.transform(data)  # Data remains as a Tensor

            # Step 5: Translate back to source language
            translated_text = self.translator.translate_hidden_to_source(data)
        else:
            # Step 5: Decode the data into text
            decoded_text = self.llm.decode_logits(data)
            translated_text = self.translator.translate_to_source(decoded_text)

        return translated_text
