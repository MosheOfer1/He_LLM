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
        self.llm_model_name = "facebook/opt-350m"

        # Path for saving/loading state
        self.path = '../models/transformer1_and_2_state.pth'
        
        self.moshe_transformer1 = 'models/transformer_1_Helsinki-NLP_opus-mt-tc-big-he-en_to_facebook_opt-350m.pth'

        # Create two instances of MyCustomModel
        self.model1: MyCustomModel = MyCustomModel(
            src_to_target_translator_model_name=self.translator1_model_name,
            target_to_src_translator_model_name=self.translator2_model_name,
            llm_model_name=self.llm_model_name,
            device=self.device)

        self.model2: MyCustomModel = MyCustomModel(
            src_to_target_translator_model_name=self.translator1_model_name,
            target_to_src_translator_model_name=self.translator2_model_name,
            llm_model_name=self.llm_model_name,
            device=self.device)

        self.model1.eval()
        self.model2.eval()
        
        self.input_ids = self.model1.translator.src_to_target_tokenizer.encode(["פילוסופיה"],
                                add_special_tokens=True, # Adds EOS token
                                return_tensors='pt'
                                ).unsqueeze(1)

    # def test_return_reshaped(self):
    #     pass
    
    def test_saving_loading_both_transformers_from_pretrained_combined_model(self):
                
        with torch.no_grad():
            # Save outputs before state dict change
            logits_before_model1 = self.model1(input_ids=self.input_ids, return_reshaped=True)
            logits_before_model2 = self.model2(input_ids=self.input_ids, return_reshaped=True)

            self.assertFalse(torch.equal(logits_before_model1, logits_before_model2))

            # Save model1 state dict & load both transformer1 and transformer2 state to model2
            self.model1.save_transformers_state_dict(self.path)
            self.model2.load_transformers_state_dict(self.path, to_transformer1=True, to_transformer2=True)

            # Check if the state dicts are identical
            self.assertTrue(self.model1.compere_state_dicts(self.model2))

            # Check model outputs after change
            logits_after_model1 = self.model1(input_ids=self.input_ids, return_reshaped=True)
            self.assertTrue(torch.equal(logits_before_model1, logits_after_model1))

            logits_after_model2 = self.model2(input_ids=self.input_ids, return_reshaped=True)
            self.assertTrue(torch.equal(logits_before_model1, logits_after_model2))

    def test_loading_only_transformer1_from_pretrained_combined_model(self):
        # Save model1 state dict & load only transformer1 state to model2
        self.model1.save_transformers_state_dict(self.path)
        self.model2.load_transformers_state_dict(self.path, to_transformer1=True, to_transformer2=False)

        # Compare transformer1 only, assume model2 still differs in transformer2
        self.assertTrue(self.model1.compere_state_dicts(self.model2, only_transformer1=True))
        self.assertFalse(self.model1.compere_state_dicts(self.model2, only_transformer2=True))

    def test_loading_only_transformer2_from_pretrained_combined_model(self):
        # Save model1 state dict & load only transformer2 state to model2
        self.model1.save_transformers_state_dict(self.path)
        self.model2.load_transformers_state_dict(self.path, to_transformer1=False, to_transformer2=True)

        # Compare transformer2 only, assume model2 still differs in transformer1
        self.assertTrue(self.model1.compere_state_dicts(self.model2, only_transformer2=True))
        self.assertFalse(self.model1.compere_state_dicts(self.model2, only_transformer1=True))

    def test_no_loading(self):
        # Do not load anything into model2
        self.assertFalse(self.model1.compere_state_dicts(self.model2))

    def test_loading_pretrained_transformer1(self):
        
        # Check logits before loading
        logits_before = self.model1(input_ids=self.input_ids, return_reshaped=True)

        # Load transformer1
        self.model1.load_transformers_state_dict(self.moshe_transformer1, to_transformer1=True, to_transformer2=False)

        logits_after = self.model1(input_ids=self.input_ids, return_reshaped=True)
                
        self.assertFalse(torch.equal(logits_before, logits_after))

if __name__ == '__main__':
    unittest.main()
