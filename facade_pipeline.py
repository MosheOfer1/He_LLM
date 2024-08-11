from llm.llm_integration import LLMIntegration
from custom_transformers.transformer_1 import Transformer1
from custom_transformers.transformer_2 import Transformer2
from translation.helsinki_translator import HelsinkiTranslator


class Pipeline:
    def __init__(self, use_transformer_1: bool = True, use_transformer_2: bool = True):
        """
        Initialize the TranslationFacade with optional transformers.

        :param use_transformer_1: Whether to use the first transformer in the pipeline.
        :param use_transformer_2: Whether to use the second transformer in the pipeline.
        """
        # Initialize the HelsinkiTranslator to handle translation between Hebrew and English
        self.translator = HelsinkiTranslator()

        # Initialize the LLMIntegration to interface with a Language Model (e.g., OPT)
        self.llm = LLMIntegration()

        # Initialize the transformers based on user preference
        self.transformer_1 = Transformer1() if use_transformer_1 else None
        self.transformer_2 = Transformer2() if use_transformer_2 else None

    def process_text(self, text: str) -> str:
        """
        Orchestrates the entire translation process, passing the hidden states between components.

        :param text: The original Hebrew text input by the user.
        :return: The final translated Hebrew text after processing through the pipeline.
        """
        # Step 1: Translate Hebrew to English and retrieve hidden states
        hidden_states = self.translator.he_to_en_hidden_states(text)

        # Step 2: Apply the first transformer if it exists
        if self.transformer_1:
            hidden_states = self.transformer_1.transform(hidden_states)
            # Inject hidden states into the LLM
            hidden_states = self.llm.inject_hidden_states(hidden_states)
        else:
            # Pass the original text to the LLM if the first transformer is not used
            hidden_states = self.llm.process_text_input(text)

        # Step 3: Apply the second transformer if it exists
        if self.transformer_2:
            hidden_states = self.transformer_2.transform(hidden_states)

        # Step 4: Translate the hidden states back to Hebrew
        translated_text = self.translator.en_to_he_hidden_states(hidden_states)

        return translated_text
