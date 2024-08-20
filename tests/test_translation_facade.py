import unittest
import sys
import os
import torch

from transformers import AutoTokenizer, AutoModel, MarianTokenizer, MarianMTModel, AutoTokenizer, OPTForCausalLM, modeling_outputs


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_model import MyCustomModel
from translation.translator import Translator
from custom_transformers.transformer import Transformer

class TestCustomModel(unittest.TestCase):

    def setUp(self):
        """Set up the CustomModel instance for testing."""

        self.translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
        self.translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
        self.llm_model_name = "facebook/opt-350m"

        self.customLLM = MyCustomModel(self.translator1_model_name,
                                       self.translator2_model_name,
                                       self.llm_model_name)
        
        self.he_text = "אבא בא לגן"
        self.en_text = "David came home"
        
    # # TODO - complete
    # def test_setup(self):
        

    #     self.assertIsInstance(self.customLLM, MyCustomModel)
        
    #     # Translator
    #     self.assertIsInstance(self.customLLM.translator, Translator)
        
    #     en_translation = self.customLLM.translator.translate(True, self.he_text)
    #     print(f"en_translation = {en_translation}")
    #     self.assertIsInstance(en_translation, str)
        
        
    #     hs = self.customLLM.translator.text_to_hidden_states(self.he_text,-1, self.customLLM.translator.src_to_target_tokenizer, self.customLLM.translator.src_to_target_model, False)


    #     # Transformer
    #     self.assertIsInstance(self.customLLM.transformer, Transformer)
        
    #     llm_hs = self.customLLM.transformer.transformer1.forward(hs)
    #     self.assertIsInstance(llm_hs, torch.Tensor)
        
        
    #     # TODO - test process_text_input_to_logits it didn't work well...
        
    #     # LLM
    #     self.customLLM.llm.inject_hidden_states(llm_hs)
    #     print(f"hs.shape[1] = {hs.shape[1]}")
    #     outputs = self.customLLM.llm.get_output_by_using_dummy(hs.shape[1])
        
    #     print(f"Type(outputs) = {type(outputs)}")
    #     print(type(outputs.keys()))
    #     print(type(outputs.logits))
    #     self.assertIsInstance(outputs, modeling_outputs.CausalLMOutputWithPast)
        
    
    def test_forward(self):
        
        logits = self.customLLM.forward(self.he_text)
        print(logits)
        
        
if __name__ == '__main__':
    unittest.main()
