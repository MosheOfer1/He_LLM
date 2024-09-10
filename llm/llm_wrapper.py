from contextlib import contextmanager

import torch
from torch import nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilty.injectable import CustomLayerWrapper, Injectable


class LLMWrapper(nn.Module, Injectable):
    def __init__(self, model_name, tokenizer, llm_model, device='cpu', *args, **kwargs):
        """
        Common initialization logic for all models.
        """
        super().__init__(*args, **kwargs)
        self.device = device
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model = llm_model.to(device)
        self.injected_layer_num = 0  # Layer to be injected
        self.inject_layer()

        self.outputs = None

    def get_layers(self):
        """
        Abstract method: Each subclass must implement this method to return the model's layers.
        """
        raise NotImplementedError("Subclasses must implement 'get_layers' method")

    def inject_layer(self):
        """
        Inject custom layer into the model by replacing the specified layer.
        """
        # Access the specific layer from the subclass implementation of get_layers()
        layers = self.get_layers()
        original_layer = layers[self.injected_layer_num]

        # Wrap the layer using a custom layer wrapper
        wrapped_layer = CustomLayerWrapper(original_layer, None)

        # Replace the original layer with the wrapped layer
        layers[self.injected_layer_num] = wrapped_layer

    def set_requires_grad(self, requires_grad: bool):
        """
            If requires_grad = True the parameters (weights) will freeze meaning they will not change during training.
        """

        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def inject_hidden_states(self, injected_hidden_state: torch.Tensor):
        """
        Inject hidden states into the LLM by using a custom layer that wrappers the origin first layer
        of the LLM: CustomLayerWrapper.

        :param injected_hidden_state: The injected hidden states
        """
        injected_hidden_state = injected_hidden_state
        self.get_layers()[self.injected_layer_num].injected_hidden_state = injected_hidden_state

    def get_output_by_using_dummy(self, token_num, batch_size=1):
        # Generate a dummy input for letting the model output the desired result of the injected layer
        dummy_input = torch.zeros((batch_size, token_num), dtype=torch.long).to(
            self.device)

        self.outputs = self.model(input_ids=dummy_input, output_hidden_states=True)

        return self.outputs

    def process_text_input_to_logits(self, text: str) -> torch.Tensor:
        """
        Process a regular text input through the LLM and return the logits layer.

        :param text: The input text to be processed by the LLM.
        :return: The last hidden state layer from the LLM.
        """
        # Tokenize the text input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(inputs.input_ids)
        return outputs.logits

    def decode_logits(self, logits: torch.Tensor) -> str:
        """
        Decodes the logits back into text.

        :param logits: The logits tensor output from the LLM.
        :return: The decoded text.
        """
        logits = logits.to(self.device) 
        
        # Get the token IDs by taking the argmax over the vocabulary dimension (dim=-1)
        token_ids = torch.argmax(logits, dim=-1)

        # Decode the token IDs to text
        generated_text = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)

        return generated_text

    @staticmethod
    def text_to_hidden_states(tokenizer, model, text, layer_num):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, output_hidden_states=True)

        return outputs.hidden_states[layer_num]

    @contextmanager
    def injection_state(self, state=False):
        """
        Context manager to set the injection state for a specific layer with default values.
        This will work with any model architecture by relying on the subclass to return the correct layers.
        """
        layer_num = self.injected_layer_num

        # Access the specific layer from the subclass implementation of get_layers()
        layers = self.get_layers()
        layer = layers[layer_num]

        # Set the injection state to the desired value
        layer.set_injection_state(state)
        try:
            yield  # Yield control to the block inside the 'with' statement
        finally:
            # Revert the injection state when exiting the 'with' block
            layer.set_injection_state(not state)
