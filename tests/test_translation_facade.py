import unittest
import sys
import os
import torch

from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_model import MyCustomModel


class TestCustomModel(unittest.TestCase):

    def setUp(self):
        """Set up the CustomModel instance for testing."""

        self.translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
        self.translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
        self.llm_model_name = "facebook/opt-350m"

        self.customLLM = MyCustomModel(self.translator1_model_name,
                                       self.translator2_model_name,
                                       self.llm_model_name)
    # TODO - complete
    def test_setup(self):
        
        he_text = "אבא בא לגן"
        en_text = "David came home"
        
        self.assertIsInstance(self.customLLM, MyCustomModel)
        
        # Translator
        en_translation = self.customLLM.translator.translate(True, he_text)
        print(f"en_translation = {en_translation}")
        self.assertIsInstance(en_translation, str)
        
        # self.assertEqual(self.customLLM.translator.src_to_target_tokenizer, None)
        # self.assertEqual(self.customLLM.translator.src_to_target_model, None)
        
        hs = self.customLLM.translator.text_to_hidden_states(he_text,-1, self.customLLM.translator.src_to_target_tokenizer, self.customLLM.translator.src_to_target_model, False)
        # hs = self.customLLM.translator.text_to_hidden_states(he_text,-1, self.tokenizer1, self.translator1, False)

        # Transformer
        llm_hs = self.customLLM.transformer.transformer1.forward(hs)
        self.assertIsInstance(llm_hs, torch.Tensor)
        
        # LLM
        self.customLLM.llm.inject_hidden_states(llm_hs)
        print(f"hs.shape[1] = {hs.shape[1]}")
        outputs = self.customLLM.llm.get_output_by_using_dummy(hs.shape[1])
        # self.assertIsInstance(outputs, torch.Tensor)
        

        
        
if __name__ == '__main__':
    unittest.main()
