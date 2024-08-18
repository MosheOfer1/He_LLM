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
        self.source_to_target_tokenizer = MarianTokenizer.from_pretrained(self.src_to_target_translator_model_name)
        self.source_to_target_model = MarianMTModel.from_pretrained(self.src_to_target_translator_model_name,
                                                                    output_hidden_states=True)

        self.target_to_source_tokenizer = MarianTokenizer.from_pretrained(self.target_to_src_translator_model_name)
        self.target_to_source_model = MarianMTModel.from_pretrained(self.target_to_src_translator_model_name,
                                                                    output_hidden_states=True)

        super().__init__(
            self.src_to_target_translator_model_name,
            self.target_to_src_translator_model_name,
            self.source_to_target_model,
            self.target_to_source_model,
            self.source_to_target_tokenizer,
            self.target_to_source_tokenizer
        )

    @staticmethod
    def text_to_hidden_states(text, layer_num, model_name, from_encoder=True):
        """
        Extracts hidden states from the specified layer in either the encoder or decoder.

        :param text: The input text to be tokenized and passed through the model.
        :param layer_num: The layer number from which to extract hidden states.
        :param model_name: The name of the MarianMTModel to load.
        :param from_encoder: If True, return hidden states from the encoder; otherwise, return from the decoder.
        :return: The hidden states from the specified layer.
        """
        # Load the tokenizer and model
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name, output_hidden_states=True)

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt")

        # Forward pass through the model, providing decoder input ids
        outputs = Translator.process_outputs(inputs, model, tokenizer)

        # Return the hidden states of the specified layer
        if from_encoder:
            return outputs.encoder_hidden_states[layer_num]
        else:
            return outputs.decoder_hidden_states[layer_num]
