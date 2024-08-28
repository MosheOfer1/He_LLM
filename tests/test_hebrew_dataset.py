import unittest
import pandas as pd
import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_datasets.hebrew_dataset_wiki import HebrewDataset
from models.custom_model import MyCustomModel

class TestHebrewDataset(unittest.TestCase):

    def setUp(self):
        """Set up for testing."""
        translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
        translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
        llm_model_name = "facebook/opt-125m"

        customLLM = MyCustomModel(translator1_model_name,
                                    translator2_model_name,
                                    llm_model_name)

        # Select the first 100 rows
        self.data = pd.read_csv("my_datasets/wikipedia_data.csv").head(100)

        # Create datasets
        self.dataset = HebrewDataset(data=self.data, 
                                    input_tokenizer=customLLM.translator.src_to_target_tokenizer, 
                                    output_tokenizer=customLLM.translator.target_to_src_tokenizer, 
                                    max_length=20)

    def test_creation(self):
        self.assertIsInstance(self.dataset, HebrewDataset)
    
    def test_len(self):
        self.assertEquals(len(self.dataset), len(self.data))
        
    def test_getitem(self):
        
        item0 = self.dataset.__getitem__(0)
        
        print(item0)
        
        self.assertIsInstance(item0.get("input_ids"), torch.Tensor)
        
        self.assertEquals(item0.get("text"), self.data.iloc[0]['Hebrew sentence'])
        
        self.assertIsInstance(item0.get("attention_mask"), torch.Tensor)
        
        self.assertIsInstance(item0.get("labels"), torch.Tensor)
        
        self.assertIsInstance(item0.get("class_weights"), torch.Tensor)


if __name__ == '__main__':
    unittest.main()
