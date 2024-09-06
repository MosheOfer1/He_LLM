import unittest
import sys
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import modeling_outputs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.combined_model import MyCustomModel
from translation.translator import Translator
from llm.llm_integration import LLMWrapper

# Transformers
from custom_transformers.transformer import Transformer

# Dataset
from my_datasets.combo_model_dataset import ComboModelDataset


class TestCustomModel(unittest.TestCase):

    def setUp(self):
        """Set up the CustomModel instance for testing."""

        self.translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
        self.translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
        self.llm_model_name = "facebook/opt-125m"

        self.customLLM = MyCustomModel(self.translator1_model_name,
                                  self.translator2_model_name,
                                  self.llm_model_name)

        self.data = pd.read_csv("my_datasets/wikipedia_data.csv")

        # Select the first 100 rows
        self.df_first_100 = self.data.head(100)

        # Split the data into training and evaluation sets
        train_data, eval_data = train_test_split(self.df_first_100, test_size=0.2)

        # Create datasets
        self.train_dataset = ComboModelDataset(text=train_data,
                                               input_tokenizer=self.customLLM.translator.src_to_target_tokenizer,
                                               output_tokenizer=self.customLLM.translator.target_to_src_tokenizer,
                                               )

        self.eval_dataset = ComboModelDataset(text=eval_data,
                                              input_tokenizer=self.customLLM.translator.src_to_target_tokenizer,
                                              output_tokenizer=self.customLLM.translator.target_to_src_tokenizer,
                                              )

        self.he_text = "אבא בא לגן"

    def test_setup(self):
        self.assertIsInstance(self.customLLM, MyCustomModel)

        # Translator
        self.assertIsInstance(self.customLLM.translator, Translator)

        en_translation = self.customLLM.translator.translate(True, self.he_text)
        print(f"en_translation = {en_translation}")
        self.assertIsInstance(en_translation, str)

        hs = self.customLLM.translator.text_to_hidden_states(self.he_text, -1,
                                                             self.customLLM.translator.src_to_target_tokenizer,
                                                             self.customLLM.translator.src_to_target_model, False)

        # Transformer
        self.assertIsInstance(self.customLLM.transformer, Transformer)

        llm_hs = self.customLLM.transformer.transformer1.forward(hs)
        self.assertIsInstance(llm_hs, torch.Tensor)

        # LLM
        self.assertIsInstance(self.customLLM.llm, LLMWrapper)

        self.customLLM.llm.inject_hidden_states(llm_hs)
        print(f"hs.shape[1] = {hs.shape[1]}")
        
        outputs = self.customLLM.llm.get_output_by_using_dummy(llm_hs.shape[1])

        self.assertIsInstance(outputs, modeling_outputs.CausalLMOutputWithPast)

    def test_forward(self):
        pass
    
    def test_params_requires_grad(self):
        # Check if only the transformers parameters are learned
        for name, param in self.customLLM.named_parameters():
            if param.requires_grad:
                name_list = name.split('.')
                self.assertEqual(name_list[0],"transformer")
                self.assertTrue(name_list[1] == "transformer1" or name_list[1] == "transformer2")


    # def test_training(self):
        
    #     # Train the model
    #     self.customLLM.train_model(train_dataset=self.train_dataset, 
    #                         eval_dataset=self.eval_dataset, 
    #                         output_dir="test_results", 
    #                         logging_dir="test_loggings",
    #                         epochs=1,
    #                         logging_steps=10,
    #                         save_steps=80,
    #                         warmup_steps=10
    #                         )

if __name__ == '__main__':
    unittest.main()
