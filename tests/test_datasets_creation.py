import unittest

from llm.llm_integration import LLMIntegration
from my_datasets.create_datasets import create_transformer1_dataset
from translation.helsinki_translator import HelsinkiTranslator


class TestTranslator(unittest.TestCase):

    def setUp(self):
        """Set up the LLMIntegration instance for testing."""

        self.model_name = "facebook/opt-125m"
        self.llm_integration = LLMIntegration(self.model_name)
        self.src_to_target_translator_model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
        self.target_to_src_translator_model_name = 'Helsinki-NLP/opus-mt-en-he'
        self.translator = HelsinkiTranslator(self.src_to_target_translator_model_name,
                                             self.target_to_src_translator_model_name)

    def test_creation(self):
        create_transformer1_dataset(self.translator, self.llm_integration)


if __name__ == '__main__':
    unittest.main()
