from transformers import OPTForCausalLM
import torch
from utilty.injectable import CustomLayerWrapper, Injectable


class LLMWrapper(Injectable):
    def __init__(self, model_name, tokenizer, llm_model):
        """
        Initialize the LLMIntegration with a specific OPT model.0
        """
        self.outputs = None
        self.tokenizer = tokenizer
        self.model: OPTForCausalLM = llm_model
        self.model_name = model_name

        # Let the LLM be Injectable by replacing the first block of the LLM
        self.injected_layer_num = 0
        original_layer = self.model.base_model.decoder.layers[self.injected_layer_num]
        wrapped_layer = CustomLayerWrapper(original_layer, None)
        self.model.base_model.decoder.layers[self.injected_layer_num] = wrapped_layer

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
        self.model.base_model.decoder.layers[self.injected_layer_num].injected_hidden_state = injected_hidden_state

    def get_output_by_using_dummy(self, token_num):
        # Generate a dummy input for letting the model output the desired result of the injected layer
        dummy_input = torch.zeros((1, token_num), dtype=torch.long)  # dtype=torch.long for token IDs

        self.outputs = self.model(input_ids=dummy_input, output_hidden_states=True)

        return self.outputs

    def process_text_input_to_logits(self, text: str) -> torch.Tensor:
        """
        Process a regular text input through the LLM and return the logits layer.

        :param text: The input text to be processed by the LLM.
        :return: The last hidden state layer from the LLM.
        """
        # Tokenize the text input
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(inputs.input_ids)
        return outputs.logits

    def decode_logits(self, logits: torch.Tensor) -> str:
        """
        Decodes the logits back into text.

        :param logits: The logits tensor output from the LLM.
        :return: The decoded text.
        """
        # Get the token IDs by taking the argmax over the vocabulary dimension (dim=-1)
        token_ids = torch.argmax(logits, dim=-1)

        # Decode the token IDs to text
        generated_text = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)

        return generated_text

    @staticmethod
    def text_to_hidden_states(tokenizer, model, text, layer_num):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)

        return outputs.hidden_states[layer_num]
