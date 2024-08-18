from abc import ABC, abstractmethod
import torch
from utilty.injectable import Injectable
from utilty.custom_layer_wrapper import CustomLayerWrapper



class Translator(Injectable):
    
    """
        src: The source language
        target: The target language
    """
    def __init__(self,src_to_target_translator_model_name,
                target_to_src_translator_model_name,
                src_to_target_tokenizer,
                src_to_target_model,
                target_to_src_tokenizer,
                target_to_src_model):
        
        self.src_to_target_translator_model_name = src_to_target_translator_model_name
        self.target_to_src_translator_model_name = target_to_src_translator_model_name
        self.src_to_target_tokenizer = src_to_target_tokenizer
        self.src_to_target_model = src_to_target_model
        self.target_to_src_tokenizer = target_to_src_tokenizer
        self.target_to_src_model = target_to_src_model
        
     # Let the LLM be Injectable by replacing the first layer of the LLM
        original_layer = self.target_to_src_model.base_model.encoder.layers[0]
        wrapped_layer = CustomLayerWrapper(original_layer, None)
        self.target_to_src_model.base_model.encoder.layers[0] = wrapped_layer

    
    def inject_hidden_states(self, layer_num, hidden_states: torch.Tensor):
        """
        Method to inject hidden states into the model.

        :param hidden_states: The hidden states tensor to be injected.
        """
        self.target_to_src_model.model.encoder.layers[layer_num].hs = hidden_states

    def get_output_using_dummy(self, token_num: int):
        """
        Method to retrieve the output after injection.

        :return: The output tensor or processed result.
        """
        # Generate a dummy input for letting the model output the desired result of the injected layer
        inputs = self.tokenizer(" " * (token_num - 1), return_tensors="pt")

        outputs = self.model(**inputs, output_hidden_states=True)

        return self.decode_logits(outputs.logits)

    
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
