from transformers import AutoTokenizer, OPTForCausalLM
import torch
import torch.nn as nn


class CustomLayerWrapper(nn.Module):
    def __init__(self, layer, hidden_states):
        super().__init__()
        self.layer = layer
        self.hs = hidden_states #transformer's result


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
        
        
        # Injectable
        original_layer = self.model.base_model.decoder.layers[1]
        wrapped_layer = CustomLayerWrapper(original_layer, None)
        self.model.base_model.decoder.layers[1] = wrapped_layer

    """
    
    """
    def inject_hs(self, layer_num, llm_first_hs: torch.Tensor): #-> torch.Tensor:
        """
        Inject hidden states into the LLM and return the logits layer.

        :param inputs_embeds: The input embeddings to be injected into the LLM.
        :return: The logits layer from the LLM.
        """
        self.model.base_model.decoder.layers[layer_num].hs = llm_first_hs
        
    
    def get_output(self, token_num = 15):
        # Generate the response based on hidden states
        inputs = self.llm_tokenizer(" " * (token_num - 1), return_tensors="pt")
        
        # decoder_input_ids = torch.tensor([[self.llm_tokenizer.pad_token_id]])

        
        outputs = self.model(**inputs, 
                           output_hidden_states=True
                           )
        
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
        """
        # Decode the predicted token IDs to text
        generated_text = self.tokenizer.decode(logits[0], skip_special_tokens=True)
        return generated_text

    def inject_input_embeddings(self, inputs_embeds: torch.Tensor, max_length: int = 50) -> torch.Tensor:
        """
        Inject input embeddings into the LLM, generate text, and return the generated outputs.

        :param inputs_embeds: The input embeddings to be injected into the LLM.
        :param max_length: Maximum length of the generated sequence.
        :return: The logits from the LLM.
        """
        # Generate the output using inputs_embeds
        generated_outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        return generated_outputs

    def process_text_input(self, text: str) -> torch.Tensor:
        """
        Process a regular text input through the LLM and return the generated outputs.

        :param text: The input text to be processed by the LLM.
        :return: The generated outputs from the LLM.
        """
        # Tokenize the text input
        inputs = self.tokenizer(text, return_tensors="pt")

        # Get the input embeddings
        inputs_embeds = self.model.model.decoder.embed_tokens(inputs['input_ids'])

        # Inject the embeddings and get the logits
        generated_outputs = self.inject_input_embeddings(inputs_embeds)

        return generated_outputs
    
    @staticmethod
    def text_to_first_hs(text, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = OPTForCausalLM.from_pretrained(model_name)
        
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        
        return outputs.hidden_states[0]
        

        


# # Example usage:
# llm = LLMIntegration()
# input_text = "The capital city of France is"
# logits = llm.process_text_input(input_text)
# generated_text = llm.decode_logits(logits)
# print(generated_text)
