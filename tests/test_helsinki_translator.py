import unittest

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from translation.helsinki_translator import HelsinkiTranslator


class TestHelsinkiTranslator(unittest.TestCase):

    def setUp(self):
        """Set up the HelsinkiTranslator instance for testing."""
        
        translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
        translator2_model_name = "Helsinki-NLP/opus-mt-en-he"

        self.translator = HelsinkiTranslator(translator1_model_name, translator2_model_name)
        self.sample_text_he = "הילד גדול."
        self.sample_text_en = "The boy is big."

        # Encode the sample text to simulate hidden states
        self.tokens = self.translator.target_to_src_tokenizer(self.sample_text_he, return_tensors="pt",
                                                                 truncation=True, padding=True)
        self.hidden_states = self.translator.target_to_src_model.get_input_embeddings()(self.tokens.input_ids)

    def test_injection(self):
        en_text = "to be or not to"
        
        llm_first_hs = self.llm_integration.text_to_first_hs(en_text, self.llm_integration.model_name)
        
        self.llm_integration.inject_hidden_states(0, llm_first_hs)
        
        llm_output = self.llm_integration.get_output_using_dummy(llm_first_hs.shape[1])
        
        self.assertIsInstance(llm_output, str)
        self.assertGreater(len(llm_output), 0)  # Ensure the decoded text is not empty
        print(llm_output.split(" ")[-1])


    
if __name__ == '__main__':
    unittest.main()
