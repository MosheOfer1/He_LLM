import torch
from transformers import MarianMTModel, MarianTokenizer


class HelsinkiTranslator:
    def __init__(self):
        self.he_to_en_model_name = 'Helsinki-NLP/opus-mt-he-en'
        self.en_to_he_model_name = 'Helsinki-NLP/opus-mt-en-he'

        self.he_to_en_tokenizer = MarianTokenizer.from_pretrained(self.he_to_en_model_name)
        self.he_to_en_model = MarianMTModel.from_pretrained(self.he_to_en_model_name, output_hidden_states=True)

        self.en_to_he_tokenizer = MarianTokenizer.from_pretrained(self.en_to_he_model_name)
        self.en_to_he_model = MarianMTModel.from_pretrained(self.en_to_he_model_name, output_hidden_states=True)

    def he_to_en_hidden_states(self, text: str) -> torch.Tensor:
        tokens = self.he_to_en_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.he_to_en_model(**tokens)
        hidden_states = outputs.hidden_states[-1]
        return hidden_states

    def en_to_he_hidden_states(self, hidden_states: torch.Tensor) -> str:
        # TODO: inject the first hidden state layer into the translator and return Hebrew text
        pass
