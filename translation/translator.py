import torch
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

        self.inputs = None
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
        :param token_num: The number of tokens to create the dummy
        :return: The outputs of the model after passing it through
        """
        # Generate a dummy input for letting the model output the desired result of the injected layer
        self.inputs = self.target_to_source_tokenizer("`" * (token_num - 2), return_tensors="pt")

        # Generate the full sentence to get the all necessary layers of hidden states of the decoder in the outputs
        self.generate_sentence_from_outputs(use_first_translator=False)

        return self.outputs

    def get_output(self, from_first, text):
        if from_first:
            tokenizer = self.source_to_target_tokenizer
            use_first_translator = True
        else:
            # Set the costume block to be not in injected mode
            self.target_to_source_model.base_model.encoder.layers[self.injected_layer_num].set_injection_state(False)

            tokenizer = self.target_to_source_tokenizer
            use_first_translator = False

        # Regular insertion
        self.inputs = tokenizer(text, return_tensors="pt")
        # Generate the full sentence to get the all necessary layers of hidden states of the decoder in the outputs
        self.generate_sentence_from_outputs(use_first_translator=use_first_translator)
        # Put it back as injectable
        self.target_to_source_model.base_model.encoder.layers[self.injected_layer_num].set_injection_state(True)

        return self.outputs

    def translate(self, from_first, text):
        output = self.get_output(
            from_first=from_first,
            text=text
        )
        tokenizer = self.source_to_target_tokenizer if from_first else self.target_to_source_tokenizer
        translated_text = self.decode_logits(
            tokenizer=tokenizer,
            logits=output.logits
        )
        return translated_text

    def generate_sentence_from_outputs(self, use_first_translator=True):
        """
        Generate a full sentence without using the `generate` method.

        :param use_first_translator: Boolean flag indicating whether to use the first translator (True) or the second (False).
        :return: The generated sentence as a string.
        """

        # Choose the appropriate tokenizer and model based on the flag
        if use_first_translator:
            tokenizer = self.source_to_target_tokenizer
            model = self.source_to_target_model
        else:
            tokenizer = self.target_to_source_tokenizer
            model = self.target_to_source_model

        # Use the static method to process inputs and get the final outputs
        self.outputs = self.process_outputs(self.inputs, model, tokenizer)

        # Extract the logits from the outputs
        final_logits = self.outputs.logits

        # Decode the logits into a sentence
        generated_sentence = self.decode_logits(tokenizer=tokenizer, logits=final_logits)

        return generated_sentence

    @staticmethod
    def process_outputs(inputs, model, tokenizer):
        """
        Processes the model to generate outputs, including logits and hidden states.

        :param inputs: The inputs to the model (e.g., encoder inputs).
        :param model: The MarianMTModel to use for generating outputs.
        :param tokenizer: The tokenizer to use for decoding.
        :return: The final outputs after processing all tokens.
        """
        # Initialize decoder input IDs with the start token ID
        decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])

        while True:
            # Run the model with the current decoder input IDs to get the outputs
            outputs = model(
                **inputs,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True
            )

            # Get the token ID for the current timestep (take argmax over the vocabulary dimension)
            token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()

            # Check if the token is an end-of-sequence token
            if token_id == tokenizer.eos_token_id:
                break

            # Update the decoder input IDs with the newly generated token
            decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[token_id]])], dim=-1)

        return outputs

    @staticmethod
    def decode_logits(tokenizer, logits: torch.Tensor) -> str:
        """
        Decodes the logits back into text.

        :param logits: The logits tensor output from the model.
        :param tokenizer: The tokenizer to use for decoding.
        :return: The decoded text.
        """
        # Get the token IDs by taking the argmax over the vocabulary dimension (dim=-1)
        token_ids = torch.argmax(logits, dim=-1)

        # If logits contain multiple sequences (batch size > 1), process each separately
        if token_ids.dim() > 1:
            # Concatenate token IDs along the sequence length dimension (dim=1)
            token_ids = token_ids.squeeze()

        # Decode the token IDs to a full sentence, skipping special tokens like <pad>, <eos>, etc.
        generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)

        return generated_text

    @staticmethod
    def text_to_hidden_states(text, layer_num, tokenizer, model, from_encoder=True):
        """
        Extracts hidden states from the specified layer in either the encoder or decoder.

        :param text: The input text to be tokenized and passed through the model.
        :param layer_num: The layer number from which to extract hidden states.
        :param tokenizer: The specific tokenizer.
        :param model: The specific model.
        :param from_encoder: If True, return hidden states from the encoder; otherwise, return from the decoder.
        :return: The hidden states from the specified layer.
        """
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt")

        # Forward pass through the model, providing decoder input ids
        outputs = Translator.process_outputs(inputs, model, tokenizer)

        # Return the hidden states of the specified layer
        if from_encoder:
            return outputs.encoder_hidden_states[layer_num]
        else:
            return outputs.decoder_hidden_states[layer_num]
