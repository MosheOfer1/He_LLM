import torch
from transformers import MarianMTModel, MarianTokenizer
from translation.translator import Translator


class HelsinkiTranslator(Translator):
    """
    The HelsinkiTranslator class is responsible for translating text and hidden states
    between Hebrew (source) and English (target) using the Helsinki-NLP MarianMT models.

    This class supports both direct text-to-text translation and the more advanced use case
    of translating hidden states generated by transformers or other models.

    Attributes:
        source_to_target_model_name (str): The model name for translating from source to target language.
        target_to_source_model_name (str): The model name for translating from target to source language.
        source_to_target_tokenizer (MarianTokenizer): Tokenizer for source to target translation.
        source_to_target_model (MarianMTModel): Pretrained MarianMT model for source to target translation.
        target_to_source_tokenizer (MarianTokenizer): Tokenizer for target to source translation.
        target_to_source_model (MarianMTModel): Pretrained MarianMT model for target to source translation.
    """

    def __init__(self):
        """
        Initializes the HelsinkiTranslator with pretrained MarianMT models and tokenizers
        for translating between Hebrew and English.

        Models and tokenizers are loaded for both directions of translation:
        from Hebrew to English and from English to Hebrew.
        """
        self.source_to_target_model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
        self.target_to_source_model_name = 'Helsinki-NLP/opus-mt-en-he'

        self.source_to_target_tokenizer = MarianTokenizer.from_pretrained(self.source_to_target_model_name)
        self.source_to_target_model = MarianMTModel.from_pretrained(self.source_to_target_model_name)

        self.target_to_source_tokenizer = MarianTokenizer.from_pretrained(self.target_to_source_model_name)
        self.target_to_source_model = MarianMTModel.from_pretrained(self.target_to_source_model_name,
                                                                    output_hidden_states=True)

    def translate_to_target(self, text: str) -> str:
        """
        Translates text from the source language (Hebrew) to the target language (English).

        :param text: The text in Hebrew to be translated.
        :return: The translated text in English.
        """
        tokens = self.source_to_target_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        translated_tokens = self.source_to_target_model.generate(**tokens)
        translated_text = self.source_to_target_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text

    def translate_to_source(self, text: str) -> str:
        """
        Translates text from the target language (English) back to the source language (Hebrew).

        :param text: The text in English to be translated.
        :return: The translated text in Hebrew.
        """
        tokens = self.target_to_source_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        translated_tokens = self.target_to_source_model.generate(**tokens)
        translated_text = self.target_to_source_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text

    def translate_hidden_to_source(self, hidden_states: torch.Tensor) -> str:
        """
        Translates hidden states directly into the source language (Hebrew).

        This method is useful when working with transformers or other models that generate
        hidden states instead of text.

        :param hidden_states: The hidden states tensor to be translated back to Hebrew.
        :return: The translated text in Hebrew.
        """
        outputs = self.target_to_source_model.generate(inputs_embeds=hidden_states)
        translated_text = self.target_to_source_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    def translate_to_en_returns_tensor(self, text: str) -> torch.Tensor:
        tokens = self.source_to_target_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        translated_tokens = self.source_to_target_model.generate(**tokens)
        return translated_tokens
