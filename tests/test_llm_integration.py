import unittest
import torch
from llm.llm_integration import LLMIntegration


class TestLLMIntegration(unittest.TestCase):

    def setUp(self):
        """Set up the LLMIntegration instance for testing."""
        self.llm_integration = LLMIntegration(model_to_use="125m")
        self.sample_text = "The capital city of France is"
        # Tokenize the sample text
        inputs = self.llm_integration.tokenizer(self.sample_text, return_tensors="pt")

        # Pass the inputs through the model to get the hidden states
        with torch.no_grad():  # Disable gradient calculation since this is a test
            outputs = self.llm_integration.model(**inputs)
            self.input_embeddings = self.llm_integration.model.model.decoder.embed_tokens(inputs['input_ids'])

        self.logits = outputs.logits

    def test_initialization(self):
        """Test that the LLMIntegration is initialized correctly."""
        self.assertIsNotNone(self.llm_integration.model)
        self.assertIsNotNone(self.llm_integration.tokenizer)

    def test_inject_embeddings(self):
        """Test injecting input embeddings into the LLM."""
        logits = self.llm_integration.inject_input_embeddings_to_logits(self.input_embeddings)
        self.assertIsInstance(logits, torch.Tensor)
        # Ensure the hidden state has injected correctly
        self.assertEqual(self.llm_integration.decode_logits(logits),
                         self.llm_integration.decode_logits(self.llm_integration.process_text_input_to_logits(self.sample_text)))
        print(self.llm_integration.decode_logits(logits))

    def test_decode_logits(self):
        """Test decoding logits back into text."""
        # Ensure the hidden states' tensor can be decoded into a string
        decoded_text = self.llm_integration.decode_logits(self.logits)
        self.assertIsInstance(decoded_text, str)
        self.assertGreater(len(decoded_text), 0)  # Ensure the decoded text is not empty
        print(decoded_text)


if __name__ == '__main__':
    unittest.main()
