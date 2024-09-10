import unittest
import sys
import os
import torch
from transformers import AutoTokenizer, OPTForCausalLM, GPT2LMHeadModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.opt_llm import OptLLM


class TestLLMIntegration(unittest.TestCase):

    def setUp(self):
        """Set up the LLMIntegration instance for testing."""

        self.model_name = "facebook/opt-125m"
        self.llm_integration = OptLLM(self.model_name)
        self.sample_text = "to be or not to"

    def test_injection(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = OPTForCausalLM.from_pretrained(self.model_name)

        first_hs = self.llm_integration.text_to_hidden_states(
            tokenizer=tokenizer,
            model=model,
            text=self.sample_text,
            layer_num=0
        )

        last_hs = self.llm_integration.text_to_hidden_states(
            tokenizer=tokenizer,
            model=model,
            text=self.sample_text,
            layer_num=-1
        )

        self.llm_integration.inject_hidden_states(first_hs)
        outputs = self.llm_integration.get_output_by_using_dummy(first_hs.shape[1])
        self.assertTrue(torch.equal(outputs.hidden_states[-1], last_hs))

        text = self.llm_integration.decode_logits(outputs.logits)
        print(text)

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

    def test_set_requires_grad(self):
        for param in self.llm_integration.model.parameters():
            self.assertEqual(param.requires_grad, True)

        self.llm_integration.set_requires_grad(False)

        for param in self.llm_integration.model.parameters():
            self.assertEqual(param.requires_grad, False)


from llm.gpt2_llm import GPT2LLM


class TestGPT2LLMIntegration(unittest.TestCase):

    def setUp(self):
        """Set up the GPT2LLM instance for testing."""
        self.model_name = "gpt2"  # Small, locally running GPT-2 model
        self.gpt2_integration = GPT2LLM(self.model_name)
        self.sample_text = "to be or not to"

    def test_injection(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = GPT2LMHeadModel.from_pretrained(self.model_name)

        # Get hidden states from the first layer
        first_hs = self.gpt2_integration.text_to_hidden_states(
            tokenizer=tokenizer,
            model=model,
            text=self.sample_text,
            layer_num=0
        )

        # Get hidden states from the last layer
        last_hs = self.gpt2_integration.text_to_hidden_states(
            tokenizer=tokenizer,
            model=model,
            text=self.sample_text,
            layer_num=-1
        )

        # Inject hidden states
        self.gpt2_integration.inject_hidden_states(first_hs)

        # Get output using dummy hidden states shape
        outputs = self.gpt2_integration.get_output_by_using_dummy(first_hs.shape[1])

        # Verify that outputs are not None and check if they're similar to last hidden states
        self.assertTrue(torch.is_tensor(outputs.hidden_states[-1]))
        self.assertEqual(outputs.hidden_states[-1].shape, last_hs.shape)

    def test_retrieval(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = GPT2LMHeadModel.from_pretrained(self.model_name)

        # Retrieve the last hidden state
        gpt2_last_hs = self.gpt2_integration.text_to_hidden_states(
            tokenizer=tokenizer,
            model=model,
            text=self.sample_text,
            layer_num=-1
        )

        # Ensure the model is in evaluation mode
        self.gpt2_integration.model.eval()

        # Pass the hidden states through the model's head (typically a linear layer)
        with torch.no_grad():
            logits = self.gpt2_integration.model.lm_head(gpt2_last_hs)

        # Decode the logits to text
        gpt2_output = self.gpt2_integration.decode_logits(logits)
        self.assertIsInstance(gpt2_output, str)
        self.assertGreater(len(gpt2_output), 0)  # Ensure the decoded text is not empty
        print(gpt2_output.split(" ")[-1])  # Print the last word

    def test_set_requires_grad(self):
        # Ensure requires_grad is True by default
        for param in self.gpt2_integration.model.parameters():
            self.assertTrue(param.requires_grad)

        # Disable gradient calculation
        self.gpt2_integration.set_requires_grad(False)

        # Ensure requires_grad is set to False
        for param in self.gpt2_integration.model.parameters():
            self.assertFalse(param.requires_grad)


if __name__ == '__main__':
    unittest.main()
