from transformers import AutoTokenizer, OPTForCausalLM
import torch
import torch.nn as nn


class CustomLayerWrapper(nn.Module):
    def __init__(self, layer, hidden_states):
        super().__init__()
        self.layer = layer
        self.hs = hidden_states  # The injected hidden state layer

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None,
                past_key_value=None, output_attentions=None, use_cache=None):
        # Apply modifications to hidden_states here

        # Pass modified_hidden_states to the original layer
        return self.layer(self.hs, attention_mask, layer_head_mask,
                          past_key_value, output_attentions, use_cache)


class LLMIntegration:
    def __init__(self, model_name):
        """
        Initialize the LLMIntegration with a specific OPT model.0
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = OPTForCausalLM.from_pretrained(model_name)
        self.model_name = model_name

        # Let the LLM be Injectable by replacing the first layer of the LLM
        original_layer = self.model.base_model.decoder.layers[0]
        wrapped_layer = CustomLayerWrapper(original_layer, None)
        self.model.base_model.decoder.layers[0] = wrapped_layer

    def inject_hs(self, layer_num, llm_first_hs: torch.Tensor):
        """
        Inject hidden states into the LLM by using a custom layer that wrappers the origin first layer of the LLM: CustomLayerWrapper.

        :param inputs_embeds: The input embeddings to be injected into the LLM.
        :return: The logits layer from the LLM.
        """
        self.model.base_model.decoder.layers[layer_num].hs = llm_first_hs

    def get_output(self, token_num=5):
        # Generate a dummy input for letting the model output the desired result of the injected layer
        inputs = self.tokenizer(" " * (token_num - 1), return_tensors="pt")

        outputs = self.model(**inputs, output_hidden_states=True)

        return self.decode_logits(outputs.logits)

    def process_text_input_to_logits(self, text: str) -> torch.Tensor:
        """
        Process a regular text input through the LLM and return the logits layer.

        :param text: The input text to be processed by the LLM.
        :return: The last hidden state layer from the LLM.
        """
        # Tokenize the text input
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
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
    def text_to_first_hs(text, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = OPTForCausalLM.from_pretrained(model_name)

        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)

        return outputs.hidden_states[0]

