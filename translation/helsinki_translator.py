from transformers import MarianMTModel, MarianTokenizer
from translation.translator import Translator


class HelsinkiTranslator(Translator):
    def __init__(self, src_to_target_translator_model_name,
                 target_to_src_translator_model_name):
        """
        Initializes the HelsinkiTranslator with pretrained MarianMT models and tokenizers
        for translating between Hebrew and English.

        Models and tokenizers are loaded for both directions of translation:
        from Hebrew to English and from English to Hebrew.
        """
        self.src_to_target_translator_model_name = src_to_target_translator_model_name
        self.target_to_src_translator_model_name = target_to_src_translator_model_name
        self.src_to_target_tokenizer = MarianTokenizer.from_pretrained(self.src_to_target_translator_model_name)
        self.src_to_target_model = MarianMTModel.from_pretrained(self.src_to_target_translator_model_name,
                                                                    output_hidden_states=True)

        self.target_to_src_tokenizer = MarianTokenizer.from_pretrained(self.target_to_src_translator_model_name)
        self.target_to_src_model = MarianMTModel.from_pretrained(self.target_to_src_translator_model_name,
                                                                    output_hidden_states=True)

        super().__init__(
            self.src_to_target_translator_model_name,
            self.target_to_src_translator_model_name,
            self.src_to_target_model,
            self.target_to_src_model,
            self.src_to_target_tokenizer,
            self.target_to_src_tokenizer
        )
