from contextlib import contextmanager

import torch

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilty.injectable import Injectable, CustomLayerWrapper


class Translator(Injectable):
    def __init__(self,
                 src_to_target_translator_model_name,
                 target_to_src_translator_model_name,
                 src_to_target_model,
                 target_to_src_model,
                 src_to_target_tokenizer,
                 target_to_src_tokenizer,
                 device='cpu'):

        print(f"Translator.__init__ - uses: {device}")

        self.device = device

        self.src_to_target_translator_model_name = src_to_target_translator_model_name
        self.target_to_src_translator_model_name = target_to_src_translator_model_name

        self.src_to_target_model = src_to_target_model.to(self.device)
        self.target_to_src_model = target_to_src_model.to(self.device)
        self.src_to_target_tokenizer = src_to_target_tokenizer
        self.target_to_src_tokenizer = target_to_src_tokenizer

        self.inputs = None
        self.outputs = None

        # Let the Second translator be Injectable by replacing his first block
        self.injected_layer_num = 0
        original_layer = self.target_to_src_model.base_model.encoder.layers[self.injected_layer_num].to(self.device)
        wrapped_layer = CustomLayerWrapper(original_layer, None).to(self.device)
        self.target_to_src_model.base_model.encoder.layers[self.injected_layer_num] = wrapped_layer

    def set_requires_grad(self, requires_grad: bool):
        """
        If requires_grad = True the parameters (weights) will freeze meaning they will not change during training.
        """
        # Translator1 parameters
        for param in self.src_to_target_model.parameters():
            param.requires_grad = requires_grad

        # Translator2 parameters
        for param in self.target_to_src_model.parameters():
            param.requires_grad = requires_grad

    def inject_hidden_states(self, injected_hidden_state: torch.Tensor):
        """
        Inject hidden states into the Second translator

        :param injected_hidden_state: The injected hidden states
        """
        self.target_to_src_model.base_model.encoder.layers[
            self.injected_layer_num].injected_hidden_state = injected_hidden_state

    def get_output_by_using_dummy(self, token_num, batch_size=1, attention_mask=None):
        """
        Receive the output from the second translator for a batch of inputs.

        :param token_num: The number of tokens to create the dummy inputs for each sequence.
        :param batch_size: The number of sequences in the batch.
        :param attention_mask: The attention mask to be used (optional).
        :return: The outputs of the model after passing the batch through.
        """
        # Create a dummy input tensor of shape (batch_size, token_num)
        dummy_input = torch.zeros((batch_size, token_num), dtype=torch.long).to(
            self.device)  # dtype=torch.long for token IDs

        # Set self.inputs to the dummy input tensor for the batch
        self.inputs = {"input_ids": dummy_input}

        # Add attention_mask to inputs if it's provided
        if attention_mask is not None:
            self.inputs['attention_mask'] = attention_mask

        # Initialize decoder input IDs for the entire batch (batch_size, 1) with the pad token ID
        decoder_input_ids = torch.full((batch_size, 1), self.target_to_src_tokenizer.pad_token_id, dtype=torch.long).to(
            self.device)

        # Pass the batch through the model
        self.outputs = self.target_to_src_model(
            **self.inputs,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True
        )

        return self.outputs

    def get_output(self, from_first, text):
        if from_first:
            tokenizer = self.src_to_target_tokenizer
            use_first_translator = True
        else:
            # Set the costume block to be not in injected mode
            self.target_to_src_model.base_model.encoder.layers[self.injected_layer_num].set_injection_state(False)

            tokenizer = self.target_to_src_tokenizer
            use_first_translator = False

        # Regular insertion
        self.inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # Generate the full sentence to get the all necessary layers of hidden states of the decoder in the outputs
        self.generate_sentence_from_outputs(use_first_translator=use_first_translator)
        # Put it back as injectable
        self.target_to_src_model.base_model.encoder.layers[self.injected_layer_num].set_injection_state(True)

        return self.outputs

    def translate(self, from_first, text):
        output = self.get_output(
            from_first=from_first,
            text=text
        ).to(self.device)

        tokenizer = self.src_to_target_tokenizer if from_first else self.target_to_src_tokenizer
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
            tokenizer = self.src_to_target_tokenizer
            model = self.src_to_target_model.to(self.device)
        else:
            tokenizer = self.target_to_src_tokenizer
            model = self.target_to_src_model.to(self.device)

        # Use the static method to process inputs and get the final outputs
        self.outputs = self.process_outputs(inputs=self.inputs, model=model, tokenizer=tokenizer)

        # Extract the logits from the outputs
        final_logits = self.outputs.logits

        # Decode the logits into a sentence
        generated_sentence = self.decode_logits(tokenizer=tokenizer, logits=final_logits)

        return generated_sentence

    @staticmethod
    def process_outputs(inputs, model, tokenizer, max_len=50):
        """
        Processes the model to generate outputs, including logits and hidden states.
        Handles a batch of inputs, such as (batch_size, seq_len).
        """
        input_ids = inputs["input_ids"]
        input_shape = input_ids.shape

        batch_size = input_shape[0]

        # Get the start token ID (<bos> or <cls>)
        if tokenizer.bos_token_id is not None:
            start_token_id = tokenizer.bos_token_id
        elif tokenizer.cls_token_id is not None:
            start_token_id = tokenizer.cls_token_id
        else:
            start_token_id = tokenizer.pad_token_id

        device = model.module.device if hasattr(model, 'module') else model.device

        # Initialize decoder input IDs with the start token ID for all sentences in the batch
        decoder_input_ids = torch.full(
            (batch_size, 1), tokenizer.pad_token_id, dtype=torch.long, device=device
        )

        # Initialize the attention mask with ones (attending to all tokens initially)
        attention_mask = torch.ones((batch_size, 1), device=device)

        counter = 0
        while True:
            # Run the model with the current decoder input IDs to get the outputs
            outputs = model(
                **inputs,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True
            )

            # Get the token IDs for the current timestep (take argmax over the vocabulary dimension)
            token_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1)  # Shape: [batch_size]

            # Check if all sentences in the batch have generated the end-of-sequence token
            if (token_ids == tokenizer.eos_token_id).all() or counter == max_len:
                break

            # Update the attention mask: append 0 if eos_token_id is encountered, otherwise append 1
            new_attention_mask_values = (token_ids != tokenizer.eos_token_id).unsqueeze(-1)  # Shape: [batch_size, 1]
            attention_mask = torch.cat([attention_mask, new_attention_mask_values], dim=-1)

            # Update the decoder input IDs with the newly generated tokens (append new tokens to decoder_input_ids)
            new_token_tensor = token_ids.unsqueeze(-1)  # Shape: [batch_size, 1]
            decoder_input_ids = torch.cat([decoder_input_ids, new_token_tensor],
                                          dim=-1)  # Shape: [batch_size, seq_len + 1]

            counter += 1

        # Add the attention mask to the inputs for future passes
        inputs['attention_mask'] = attention_mask

        return outputs

    @staticmethod
    def decode_logits(tokenizer, logits: torch.Tensor) -> str:
        """
        Decodes the logits back into text for each sentence in the batch.

        :param logits: The logits tensor output from the model (batch_size, seq_len, vocab_size).
        :param tokenizer: The tokenizer to use for decoding.
        :return: A list of decoded texts, one for each sentence in the batch.
        """
        # Get the token IDs by taking the argmax over the vocabulary dimension (dim=-1)
        token_ids = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_len)

        # Decode each sequence in the batch individually
        generated_texts = []
        for seq in token_ids:
            # Decode the token IDs to a sentence, skipping special tokens like <pad>, <eos>, etc.
            generated_text = tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            generated_texts.append(generated_text)

        generated_texts = "\n".join(generated_texts)
        return generated_texts

    @staticmethod
    def text_to_hidden_states(text, layer_num, tokenizer, model, from_encoder=True):
        """
        Extracts hidden states from the specified layer in either the encoder or decoder for a batch of sentences.

        :param text: List of input sentences to be tokenized and passed through the model.
        :param layer_num: The layer number from which to extract hidden states.
        :param tokenizer: The specific tokenizer.
        :param model: The specific model.
        :param from_encoder: If True, return hidden states from the encoder; otherwise, return from the decoder.
        :return: The hidden states from the specified layer.
        """
        device = model.module.device if hasattr(model, 'module') else model.device

        # Tokenize the input text as a batch
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

        # Forward pass through the model, providing decoder input ids
        outputs = Translator.process_outputs(inputs=inputs, model=model, tokenizer=tokenizer)

        # Return the hidden states of the specified layer for all sentences in the batch
        if from_encoder:
            return outputs.encoder_hidden_states[layer_num]
        else:
            return outputs.decoder_hidden_states[layer_num]

    @staticmethod
    def input_ids_to_hidden_states(input_ids, layer_num, tokenizer, model, from_encoder=True, attention_mask=None):
        device = model.module.device if hasattr(model, 'module') else model.device

        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask
        }

        # Forward pass through the model, providing decoder input ids
        outputs = Translator.process_outputs(inputs=inputs, model=model, tokenizer=tokenizer)
        attention_mask = inputs.get('attention_mask')

        # Return the hidden states of the specified layer
        if from_encoder:
            return outputs.encoder_hidden_states[layer_num], attention_mask
        else:
            return outputs.decoder_hidden_states[layer_num], attention_mask

    @contextmanager
    def injection_state(self):
        """Context manager to set the injection state for a specific layer with default values."""
        # Use default values: injected_layer_num and False for the state
        layer_num = self.injected_layer_num
        state = False

        # Set the injection state to the desired value
        self.target_to_src_model.base_model.encoder.layers[layer_num].set_injection_state(state)
        try:
            # Yield control to the block inside the 'with' statement
            yield
        finally:
            # Revert the injection state when exiting the 'with' block
            self.target_to_src_model.base_model.encoder.layers[layer_num].set_injection_state(not state)
