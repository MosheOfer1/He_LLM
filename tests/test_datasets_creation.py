import unittest

from custom_transformers.transformer_1 import Transformer1
from custom_transformers.transformer_2 import Transformer2
from llm.llm_integration import LLMIntegration
from my_datasets.create_datasets import create_transformer1_dataset, create_transformer2_dataset
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

    def test_transformer_1(self):
        tr = Transformer1(
            self.translator,
            self.llm_integration
        )
        tr.load_or_train_model()

    def test_transformer_2(self):
        tr = Transformer2(
            self.translator,
            self.llm_integration
        )
        tr.load_or_train_model()

    def test_creation1(self):
        file_path = '../my_datasets/transformer1_dataset.pt'
        create_transformer1_dataset(self.translator, self.llm_integration, file_path)

    def test_creation2(self):
        file_path = '../my_datasets/transformer2_dataset_test.pt'
        create_transformer2_dataset(self.translator, self.llm_integration, file_path)


if __name__ == '__main__':
    unittest.main()
