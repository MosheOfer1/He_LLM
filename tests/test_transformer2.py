import unittest
import torch

from custom_transformers.transformer_2 import Transformer2


class TestTransformer2(unittest.TestCase):
    def setUp(self):
        # Create a mock LLMWrapper and Translator with the necessary attributes
        class MockLLM:
            class Config:
                def __init__(self):
                    self.n_embd = 512
                    self.name_or_path = 'mock-llm'

            def __init__(self):
                self.model = self  # Mimic the structure where LLMWrapper contains a model
                self.config = self.Config()

        class MockTranslator:
            class Config:
                def __init__(self):
                    self.hidden_size = 768

            def __init__(self):
                self.target_to_src_model = self
                self.config = self.Config()
                self.target_to_src_translator_model_name = 'mock-translator'

        llm = MockLLM()
        translator = MockTranslator()

        # Instantiate Transformer2
        self.model = Transformer2(translator=translator, llm=llm, hidden_dim=1024, device='cpu')

    def test_input_len_longer_than_output_len(self):
        # Input hidden states: batch_size=2, seq_len=4, input_dim=512
        hidden_states = torch.randn(2, 4, 512)

        # Input attention mask
        input_attention_mask = torch.tensor([[1, 1, 1, 1],
                                             [1, 1, 0, 0]])

        # Output attention mask
        output_attention_mask = torch.tensor([[1, 1, 1],
                                              [1, 0, 0]])

        # Forward pass
        output = self.model(hidden_states, input_attention_mask=input_attention_mask,
                            output_attention_mask=output_attention_mask)

        # Check output shape: (batch_size=2, output_seq_len=3, output_dim=768)
        self.assertEqual(output.shape, (2, 3, 768))

    def test_output_len_longer_than_input_len(self):
        # Input hidden states: batch_size=2, seq_len=3, input_dim=512
        hidden_states = torch.randn(2, 3, 512)

        # Input attention mask
        input_attention_mask = torch.tensor([[1, 1, 1],
                                             [1, 0, 0]])

        # Output attention mask
        output_attention_mask = torch.tensor([[1, 1, 1, 1],
                                              [1, 1, 0, 0]])

        # Forward pass
        output = self.model(hidden_states, input_attention_mask=input_attention_mask,
                            output_attention_mask=output_attention_mask)

        # Check output shape: (batch_size=2, output_seq_len=4, output_dim=768)
        self.assertEqual(output.shape, (2, 4, 768))


if __name__ == '__main__':
    unittest.main()
