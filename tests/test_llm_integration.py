import unittest
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

    def test_injection(self):
        en_text = "to be or not to"
        
        llm_first_hs = self.llm_integration.text_to_first_hs(en_text, self.llm_integration.model_name)
        
        self.llm_integration.inject_hs(0, llm_first_hs)
        
        llm_output = self.llm_integration.get_output(llm_first_hs.shape[1])
        
        self.assertIsInstance(llm_output, str)
        self.assertGreater(len(llm_output), 0)  # Ensure the decoded text is not empty
        print(llm_output.split(" ")[-1])


if __name__ == '__main__':
    unittest.main()
    