import torch
from transformers import MarianTokenizer, MarianMTModel

from utilty.injectable import Injectable, CustomLayerWrapper


class Translator(Injectable):
    def __init__(self,
                 src_to_target_translator_model_name,
                 target_to_src_translator_model_name,
                 source_to_target_model,
                 target_to_source_model,
                 source_to_target_tokenizer,
                 target_to_source_tokenizer):
        self.src_to_target_translator_model_name = src_to_target_translator_model_name
        self.target_to_src_translator_model_name = target_to_src_translator_model_name
        self.source_to_target_model = source_to_target_model
        self.target_to_source_model = target_to_source_model
        self.source_to_target_tokenizer = source_to_target_tokenizer
        self.target_to_source_tokenizer = target_to_source_tokenizer

        self.outputs = None

        # Let the Second translator be Injectable by replacing his first block
        self.injected_layer_num = 0
        original_layer = self.target_to_source_model.base_model.encoder.layers[self.injected_layer_num]
        wrapped_layer = CustomLayerWrapper(original_layer, None)
        self.target_to_source_model.base_model.encoder.layers[self.injected_layer_num] = wrapped_layer

    def inject_hidden_states(self, injected_hidden_state: torch.Tensor):
        """
        Inject hidden states into the Second translator

        :param injected_hidden_state: The injected hidden states
        """
        self.target_to_source_model.base_model.encoder.layers[self.injected_layer_num].injected_hidden_state = injected_hidden_state

    def get_output_by_using_dummy(self, token_num):
        """
        Receive the output from the Second translator
        :param token_num:
        :return:
        """
        # Generate a dummy input for letting the model output the desired result of the injected layer
        inputs = self.target_to_source_tokenizer("`" * (token_num - 2), return_tensors="pt")

        # Generate decoder input ids using the start token
        decoder_input_ids = torch.tensor([[self.target_to_source_tokenizer.pad_token_id]])

        # Forward pass through the model, providing decoder input ids
        self.outputs = self.target_to_source_model(**inputs, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

        return self.outputs

    def get_output(self, from_first, text):
        if from_first:
            # Regular insertion
            inputs = self.source_to_target_tokenizer(text, return_tensors="pt")
            self.outputs = self.source_to_target_model(**inputs, output_hidden_states=True)
        else:  # From second translator which his first block in custom and need to be injected
            # Injection
            second_trans_first_hs = self.text_to_hidden_states(text, 0, self.target_to_src_translator_model_name)
            self.inject_hidden_states(second_trans_first_hs)

            # Dummy insertion
            token_num = second_trans_first_hs.shape[1]
            inputs = self.target_to_source_tokenizer("`" * (token_num - 2), return_tensors="pt")
            self.outputs = self.target_to_source_model(**inputs, output_hidden_states=True)

        return self.outputs

    def decode_logits(self, from_first, logits: torch.Tensor) -> str:
        """
        Decodes the logits back into text.

        :param from_first: Indicates whether to use the source-to-target or target-to-source tokenizer.
        :param logits: The logits tensor output from the LLM.
        :return: The decoded text.
        """
        # Get the token IDs by taking the argmax over the vocabulary dimension (dim=-1)
        token_ids = torch.argmax(logits, dim=-1)

        # If logits contain multiple sequences (batch size > 1), process each separately
        if token_ids.dim() > 1:
            # Concatenate token IDs along the sequence length dimension (dim=1)
            token_ids = token_ids.squeeze()

        # Use the appropriate tokenizer to decode the token IDs
        tokenizer = self.source_to_target_tokenizer if from_first else self.target_to_source_tokenizer

        # Decode the token IDs to a full sentence, skipping special tokens like <pad>, <eos>, etc.
        generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)

        return generated_text

    @staticmethod
    def text_to_hidden_states(text, layer_num, model_name):
        # Load the tokenizer and model
        target_to_source_tokenizer = MarianTokenizer.from_pretrained(model_name)
        target_to_source_model = MarianMTModel.from_pretrained(model_name,
                                                               output_hidden_states=True)

        # Tokenize the input text
        inputs = target_to_source_tokenizer(text, return_tensors="pt")

        # Generate decoder input ids using the start token
        decoder_input_ids = torch.tensor([[target_to_source_tokenizer.pad_token_id]])

        # Forward pass through the model, providing decoder input ids
        outputs = target_to_source_model(**inputs, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

        # Return the hidden states of the specified layer in the encoder
        return outputs.encoder_hidden_states[layer_num]

