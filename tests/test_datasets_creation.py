import unittest

import sys
import os

from transformers import AutoTokenizer, OPTForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from custom_transformers.transformer_1 import Transformer1
from custom_transformers.transformer_2 import Transformer2
from llm.llm_integration import LLMWrapper
from my_datasets.create_datasets import create_transformer1_dataset, create_transformer2_dataset
from translation.helsinki_translator import HelsinkiTranslator


class TestDatasetCreation(unittest.TestCase):

    def setUp(self):
        """Set up the LLMIntegration instance for testing."""

        self.model_name = "facebook/opt-125m"
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.llm_model = OPTForCausalLM.from_pretrained(self.model_name)
        
        self.llm_integration = LLMWrapper(self.model_name, self.llm_tokenizer, self.llm_model)
        
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
        file_path = 'my_datasets/'
        create_transformer1_dataset(self.translator, self.llm_integration, file_path,
                                    dataset_name="SVLM_Hebrew_Wikipedia_Corpus.txt",
                                    sentence_num=5
                                    )

    def test_creation2(self):
        file_path = 'my_datasets/'
        create_transformer2_dataset(self.translator, self.llm_integration, file_path)


if __name__ == '__main__':
    unittest.main()
