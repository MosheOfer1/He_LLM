from transformers import AutoTokenizer, OPTForCausalLM
import torch


class LLMIntegration:
    def __init__(self, model_to_use="350m"):
        """
        Initialize the LLMIntegration with a specific OPT model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use)
        self.model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use, output_hidden_states=True)

    def inject_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Inject hidden states into the LLM and return the last hidden state layer.

        :param hidden_states: The hidden states to be injected into the LLM.
        :return: The last hidden state layer from the LLM.
        """
        # Generate the response based on hidden states
        outputs = self.model(inputs_embeds=hidden_states)
        last_hidden_state = outputs.hidden_states[-1]
        return last_hidden_state

    def process_text_input(self, text: str) -> torch.Tensor:
        """
        Process a regular text input through the LLM and return the last hidden state layer.

        :param text: The input text to be processed by the LLM.
        :return: The last hidden state layer from the LLM.
        """
        # Tokenize the text input
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        last_hidden_state = outputs.hidden_states[-1]
        return last_hidden_state

    def decode_hidden_states(self, hidden_states: torch.Tensor) -> str:
        """
        Decodes the hidden states back into text.
        """
        tokens = hidden_states.argmax(dim=-1)
        decoded_text = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        return decoded_text
