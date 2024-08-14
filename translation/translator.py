from abc import ABC, abstractmethod
import torch


class Translator(ABC):
    @abstractmethod
    def translate_to_target(self, text: str) -> str:
        """
        Translates the input text to the target language.

        :param text: The text to be translated.
        :return: Translated text in the target language.
        """
        pass

    @abstractmethod
    def translate_to_en_returns_hidden_states(self, text: str) -> torch.Tensor:
        """
        Translates the input text to the target language.

        :param text: The text to be translated.
        :return: Translated text in the target language.
        """
        pass

    @abstractmethod
    def translate_to_source(self, text: str) -> str:
        """
        Translates the input text back to the source language.

        :param text: The text to be translated back.
        :return: Translated text in the source language.
        """
        pass

    @abstractmethod
    def translate_hidden_to_source(self, hidden_states: torch.Tensor) -> str:
        """
        Translates hidden states back to the source language.

        :param hidden_states: The hidden states to be translated back.
        :return: Translated text in the source language.
        """
        pass
