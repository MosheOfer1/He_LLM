from transformers import AutoTokenizer, OPTForCausalLM
import torch


class LLMIntegration:
    def __init__(self, model_to_use="350m"):
        """
        Initialize the LLMIntegration with a specific OPT model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use)
        self.model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use, output_hidden_states=True)

    def inject_input_embeddings_to_logits(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Inject hidden states into the LLM and return the logits layer.

        :param inputs_embeds: The input embeddings to be injected into the LLM.
        :return: The logits layer from the LLM.
        """
        # Generate the response based on hidden states
        outputs = self.model(inputs_embeds=inputs_embeds)
        return outputs.logits

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


# Example usage:
llm = LLMIntegration()
input_text = "The capital city of France is"
logits = llm.process_text_input(input_text)
generated_text = llm.decode_logits(logits)
print(generated_text)
