import unittest
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.combined_model import MyCustomModel


class TestCombinedModel(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Im working with: {self.device} in TestCombinedModel")

        self.translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
        self.translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
        self.llm_model_name = "facebook/opt-125m"        

        # Path for saving/loading state
        self.path = 'transformer1_and_2_state.pth'

        # Create two instances of MyCustomModel
        self.model1 = MyCustomModel(
            src_to_target_translator_model_name=self.translator1_model_name,
            target_to_src_translator_model_name=self.translator2_model_name,
            llm_model_name=self.llm_model_name,
            device=self.device)

        self.model2 = MyCustomModel(
            src_to_target_translator_model_name=self.translator1_model_name,
            target_to_src_translator_model_name=self.translator2_model_name,
            llm_model_name=self.llm_model_name,
            device=self.device)

    def test_saving_loading_both_transformers(self):
        # Save model1 state dict & load both transformer1 and transformer2 state to model2
        self.model1.save_transformers_state_dict(self.path)
        self.model2.load_transformers_state_dict(self.path, to_transformer1=True, to_transformer2=True)

        # Check if the state dicts are identical
        self.assertTrue(self.model1.compere_state_dicts(self.model2))

    def test_loading_only_transformer1(self):
        # Save model1 state dict & load only transformer1 state to model2
        self.model1.save_transformers_state_dict(self.path)
        self.model2.load_transformers_state_dict(self.path, to_transformer1=True, to_transformer2=False)

        # Compare transformer1 only, assume model2 still differs in transformer2
        self.assertTrue(self.model1.compere_state_dicts(self.model2, only_transformer1=True))
        self.assertFalse(self.model1.compere_state_dicts(self.model2, only_transformer2=True))

    def test_loading_only_transformer2(self):
        # Save model1 state dict & load only transformer2 state to model2
        self.model1.save_transformers_state_dict(self.path)
        self.model2.load_transformers_state_dict(self.path, to_transformer1=False, to_transformer2=True)

        # Compare transformer2 only, assume model2 still differs in transformer1
        self.assertTrue(self.model1.compere_state_dicts(self.model2, only_transformer2=True))
        self.assertFalse(self.model1.compere_state_dicts(self.model2, only_transformer1=True))

    def test_no_loading(self):
        # Do not load anything into model2
        self.assertFalse(self.model1.compere_state_dicts(self.model2))


if __name__ == '__main__':
    unittest.main()
