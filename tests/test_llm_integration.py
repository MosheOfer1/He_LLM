import unittest
import torch

import sys
import os


# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from llm.llm_integration import LLMIntegration


class TestLLMIntegration(unittest.TestCase):

    def setUp(self):
        """Set up the LLMIntegration instance for testing."""
        
        model_name = "facebook/opt-350m"
        self.llm_integration = LLMIntegration(model_name)
        self.sample_text = "The capital city of France is"
        # Tokenize the sample text
        inputs = self.llm_integration.tokenizer(self.sample_text, return_tensors="pt")

        # # Pass the inputs through the model to get the hidden states
        # with torch.no_grad():  # Disable gradient calculation since this is a test
        #     outputs = self.llm_integration.model(**inputs)
        #     self.input_embeddings = self.llm_integration.model.model.decoder.embed_tokens(inputs['input_ids'])

        # self.logits = outputs.logits
        
    def test_injection(self):
        
        en_text = "Hi my name is"
        
        llm_first_hs = self.llm_integration.text_to_first_hs(en_text, self.llm_integration.model_name)
        
        self.llm_integration.inject_hs(1, llm_first_hs)
        
        llm_output = self.llm_integration.get_output()
        
        self.assertIsInstance(llm_output, str)
        self.assertGreater(len(llm_output), 0)  # Ensure the decoded text is not empty
        print(llm_output)


if __name__ == '__main__':
    unittest.main()


    
    
    