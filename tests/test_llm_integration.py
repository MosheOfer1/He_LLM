import unittest
import sys
import os
import torch
from transformers import AutoTokenizer, OPTForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.llm_integration import LLMIntegration


class TestLLMIntegration(unittest.TestCase):

    def setUp(self):
        """Set up the LLMIntegration instance for testing."""

        self.model_name = "facebook/opt-125m"
        self.llm_integration = LLMIntegration(self.model_name)
        self.sample_text = "to be or not to"

    def test_injection(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = OPTForCausalLM.from_pretrained(self.model_name)

        llm_first_hs = self.llm_integration.text_to_hidden_states(
            tokenizer=tokenizer,
            model=model,
            text=self.sample_text,
            layer_num=0
        )

        self.llm_integration.inject_hidden_states(llm_first_hs)

        llm_output = self.llm_integration.get_output_by_using_dummy(llm_first_hs.shape[1])
        llm_output = self.llm_integration.decode_logits(llm_output.logits)
        self.assertIsInstance(llm_output, str)
        self.assertGreater(len(llm_output), 0)  # Ensure the decoded text is not empty
        print(llm_output.split(" ")[-1])

    def test_retrieval(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = OPTForCausalLM.from_pretrained(self.model_name)

        llm_last_hs = self.llm_integration.text_to_hidden_states(
            tokenizer=tokenizer,
            model=model,
            text=self.sample_text,
            layer_num=-1
        )
        # Ensure the model is in evaluation mode
        self.llm_integration.model.eval()

        # Pass the hidden states through the model's head (typically a linear layer)
        with torch.no_grad():
            logits = self.llm_integration.model.lm_head(llm_last_hs)

        llm_output = self.llm_integration.decode_logits(logits)
        self.assertIsInstance(llm_output, str)
        self.assertGreater(len(llm_output), 0)  # Ensure the decoded text is not empty
        print(llm_output.split(" ")[-1])


if __name__ == '__main__':
    unittest.main()
