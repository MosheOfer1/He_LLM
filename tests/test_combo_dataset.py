import unittest

import torch
from transformers import MarianTokenizer, MarianMTModel

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_datasets.combo_model_dataset import ComboModelDataset
from facade import create_datasets_from_txt_file
from translation.translator import Translator


def similarity_check(tokens1: list, tokens2: list):

    if len(tokens1) != len(tokens2):
        raise "similarity_check - tokens1 & tokens2 has to have the same len"
    
    score = 0
    
    for idx in range(len(tokens1)):
        input_token = tokens1[idx]
        target_token = tokens2[idx]
        
        min_len = min(len(input_token), len(target_token))
        max_len = max(len(input_token), len(target_token))
        
        counter = 0
        
        for i in range(min_len):
            if input_token[i] == target_token[i]:
                counter += 1
        
        score += (counter / max_len) if max_len > 0 else 0

    return (score / len(tokens1)) if len(tokens1) > 0 else 0


class TestComboModelDataset(unittest.TestCase):

    def setUp(self):
        # For sentence similarity check 
        
        
        # Initialize the tokenizers
        src_to_target_translator_model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
        target_to_src_translator_model_name = 'Helsinki-NLP/opus-mt-en-he'
        self.input_tokenizer = MarianTokenizer.from_pretrained(src_to_target_translator_model_name)
        self.output_tokenizer = MarianTokenizer.from_pretrained(target_to_src_translator_model_name)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Skip the setup for a specific test case
        if self._testMethodName == 'test_different_tokenization_case':
            return
        
        text_file_path = "my_datasets/100_sentences_form_SVLM_for_tests.txt"
        
        # Initialize the models correctly
        src_to_target_translator = MarianMTModel.from_pretrained(src_to_target_translator_model_name,
                                                         output_hidden_states=True)
        target_to_src_translator = MarianMTModel.from_pretrained(target_to_src_translator_model_name,
                                                          output_hidden_states=True)
        
        self.translator = Translator(src_to_target_translator_model_name=src_to_target_translator_model_name,
                                     target_to_src_translator_model_name=target_to_src_translator_model_name,
                                     src_to_target_model=src_to_target_translator,
                                     target_to_src_model=target_to_src_translator,
                                     src_to_target_tokenizer=self.input_tokenizer,
                                     target_to_src_tokenizer=self.output_tokenizer,
                                     device=self.device)
        # Create datasets
        self.train_dataset, _ = create_datasets_from_txt_file(translator=self.translator, 
                                                                    text_file_path=text_file_path,
                                                                    train_percentage=0.8,
                                                                    device=self.device)

    # def test_batches(self):
    #     pass
    
    def test_get_item(self):
        self.assertTrue(len(self.train_dataset) == 80)
        
        for idx, item in enumerate(self.train_dataset):
            
            input_ids = item.get("input_ids")
            labels = item.get("labels")
            
            # Check that we have as much ids as we have labels
            self.assertTrue(input_ids.shape[1] == labels.shape[0])
            
            # input_tokens = [self.input_tokenizer.decode(id, skip_special_tokens=True) for id in input_ids[0]]
            
            # target_tokens = [self.output_tokenizer.decode(id, skip_special_tokens=True) for id in labels]
                        
            # # Check similarity (without BOS & new target token) 
            # similarity = similarity_check(" ".join(input_tokens[1:]), " ".join(target_tokens[:-1]))
            # print(f"similarity: {similarity}")
            # # self.assertTrue(similarity > 0.7)
    
    def test_dataset_pairs_creation(self):
        """ Check that the pairs aligned with each other """
        
        for idx, sentence in enumerate(self.train_dataset.sentences_pair_tokens):
            for pair in sentence:
                input_token = pair[0].rstrip('_')
                target_token = pair[1].rstrip('_')
                self.assertTrue(input_token.startswith(target_token) or target_token.startswith(input_token),
                                msg=f"In sentence {idx} pair: {pair} are not aligned with each other")
    
    def test_different_tokenization_case(self):
        # כלים אקדמיים שיסייעו להבנת שוק הספרים בישראל יתקבלו כמובן בברכה

        train_data = ["האינציקלופדיה העברית","למד בחוגים לחינוך ופילוסופיה באוניברסיטה העברית בירושלים", "למד צרפתית וספרות בבודפשט ולבסוף הוסמך", "לאור קישורים חיצוניים אתר האינטרנט של המרכז הירושלמי לענייני ציבור ומדינה"]

        # train_data = [train_data[1]] 

        # Create datasets
        train_dataset = ComboModelDataset(
            text_list=train_data,
            input_tokenizer=self.input_tokenizer,
            output_tokenizer=self.output_tokenizer,
            device=self.device
        )
        
        # TODO - complete
        

if __name__ == '__main__':
    unittest.main()
    