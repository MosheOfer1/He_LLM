import unittest

import torch
from transformers import MarianTokenizer, MarianMTModel

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_datasets.combo_model_dataset import ComboModelDataset
from facade import create_datasets_from_txt_file
from translation.translator import Translator
from llm.opt_llm import OptLLM
from models.combined_model import MyCustomModel


def check_alignment(s1, s2) -> bool:
    return s1.startswith(s2) or s2.startswith(s1)

class TestComboModelDataset(unittest.TestCase):

    def setUp(self):
        
        # Initialize the tokenizers
        src_to_target_translator_model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
        target_to_src_translator_model_name = 'Helsinki-NLP/opus-mt-en-he'
        self.input_tokenizer = MarianTokenizer.from_pretrained(src_to_target_translator_model_name)
        self.output_tokenizer = MarianTokenizer.from_pretrained(target_to_src_translator_model_name)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Skip the setup for a specific test case
        if self._testMethodName == 'test_different_tokenization_case':
            return
        
        
        
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
        
        if self._testMethodName == 'test_get_item' or self._testMethodName == 'test_dataset_pairs_creation':
            text_file_path = "my_datasets/100_sentences_form_SVLM_for_tests.txt"
                
            # Create datasets
            self.train_dataset, _ = create_datasets_from_txt_file(translator=self.translator, 
                                                                        text_file_path=text_file_path,
                                                                        train_percentage=0.8,
                                                                        device=self.device)

    def test_batches(self):
        translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
        translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
        llm_model_name = "facebook/opt-125m"
        
        text_file_path = "my_datasets/test_batches_dataset.txt"
        train_dataset, eval_dataset = create_datasets_from_txt_file(translator=self.translator, 
                                                                    text_file_path=text_file_path,
                                                                    device=self.device)
        batch_size = 8
        epochs = 1
        
        customLLM = MyCustomModel(translator1_model_name,
                          translator2_model_name,
                          llm_model_name,
                          OptLLM,
                          device=self.device)
        
        trainer = customLLM.create_trainer(train_dataset=train_dataset,
                                 eval_dataset=eval_dataset,
                                 output_dir="",
                                 logging_dir="",
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 weight_decay=0.1,
                                 lr=0.0001,
                                 max_grad_norm=1.0,
                                 evaluation_strategy="steps",
                                 optimizer=None,
                                 scheduler=None,
                                 device=self.device,
                                 save_strategy="no",
                                 save_steps=1000,
                                 logging_steps=1000,
                                 save_total_limit=0
                                 )

        for batch_idx, batch in enumerate(trainer.get_train_dataloader()):
            input_ids = batch.get("input_ids")
            labels = batch.get("labels")
            
            # Check that there are the same tokens amount
            self.assertEqual(input_ids.shape[2], labels.shape[1])
            
            # Loop over the sentences in the batch
            for sentence_idx in range(input_ids.shape[0]):
                input_tokens = [self.input_tokenizer.convert_ids_to_tokens(token_id) for token_id in input_ids[sentence_idx]][0]
                labels_tokens = [self.output_tokenizer.convert_ids_to_tokens(token_id.item()) for token_id in labels[sentence_idx]]
                
                # Loop over token ids (without the last one - new prediction)
                for i in range(input_ids.shape[2] - 1):
                    # Check that the tokens are aligned with each other
                    self.assertTrue(check_alignment(input_tokens[i], labels_tokens[i]),
                    msg=f"Test Error - test_combo_dataset - test_batches - In batch {batch_idx}, sentence: {sentence_idx}, input_ids & labels are not aligned with each other")

    def test_get_item(self):
        self.assertTrue(len(self.train_dataset) == 80)
        
        for idx, item in enumerate(self.train_dataset):
            
            input_ids = item.get("input_ids")
            labels = item.get("labels")
            
            # Check that we have as much ids as we have labels
            self.assertTrue(input_ids.shape[1] == labels.shape[0])
    
    def test_dataset_pairs_creation(self):
        """ Check that the pairs aligned with each other """
        
        for idx, sentence in enumerate(self.train_dataset.sentences_pair_tokens):
            for pair in sentence:
                self.assertTrue(check_alignment(pair[0], pair[1]),
                                msg=f"In sentence {idx} pair: {pair} are not aligned with each other")
    
    def test_different_tokenization_case(self):
        # כלים אקדמיים שיסייעו להבנת שוק הספרים בישראל יתקבלו כמובן בברכה

        train_data = ["האינציקלופדיה העברית",
                      "למד בחוגים לחינוך ופילוסופיה באוניברסיטה העברית בירושלים", 
                      "למד צרפתית וספרות בבודפשט ולבסוף הוסמך", 
                      "לאור קישורים חיצוניים אתר האינטרנט של המרכז הירושלמי לענייני ציבור ומדינה",
                      "כלים אקדמיים שיסייעו להבנת שוק הספרים בישראל יתקבלו כמובן בברכה"]

        # Create datasets
        train_dataset = ComboModelDataset(
            text_list=train_data,
            input_tokenizer=self.input_tokenizer,
            output_tokenizer=self.output_tokenizer,
            device=self.device
        )
        
        expected_input = [['▁האינ', 'ציקלו', 'פד', 'יה', '</s>'], 
                          ['▁למד', '▁בחוג', 'ים', '▁לחינוך', '▁ו', 'פילוסופי', 'ה', '▁באוניברסיטה', '▁העברית', '</s>'],
                          ['▁למד', '▁צרפתית', '▁ו', 'ספר', 'ות', '▁בבודפשט', '▁ולבסוף', '▁הו', '</s>'],
                          ['▁לאור', '▁קישורים', '▁חיצוניים', '▁אתר', '▁האינטרנט', '▁של', '▁המרכז', '▁הירושלמי', '▁לענייני', '▁ציבור', '▁ו', '</s>'],
                          ['▁כל', 'ים', '▁אקדמיים', '▁שיסייע', 'ו', '▁להבנת', '▁שוק', '▁הספרים', '▁בישראל', '▁יתקבל', 'ו', '▁כמובן', '</s>']
                         ]
        expected_labels = [['▁האינ', 'ציקלופ', 'פ', 'י', '▁העבר'],
                           ['▁למד', '▁בחוג', 'ים', '▁לחינוך', '▁ו', 'פילוסופיה', 'ה', '▁באוניברסיטה', '▁העבר', '▁בירושלים'],
                           ['▁למד', '▁צרפתית', '▁וספר', 'ס', 'ות', '▁בבו', '▁ולבסוף', '▁הו', 'סמך'],
                           ['▁לאור', '▁קישור', '▁חיצוני', '▁את', '▁האינטרנט', '▁של', '▁המרכז', '▁הירו', '▁לענייני', '▁ציבור', '▁ו', 'מדינה'],
                           ['▁כל', 'ים', '▁אקדמי', '▁שי', 'ו', '▁ל', '▁שוק', '▁הספרים', '▁בישראל', '▁יתקבל', 'ו', '▁כמובן', '▁בברכה']
                          ]
        
        for sen_idx, sentence_data in enumerate(train_dataset):
            input = [self.input_tokenizer.convert_ids_to_tokens(token_id) for token_id in sentence_data.get("input_ids")][0]
            labels = [self.output_tokenizer.convert_ids_to_tokens(token_id.item()) for token_id in sentence_data.get("labels")]

            for idx in range(len(input)):
                self.assertTrue(input[idx] == expected_input[sen_idx][idx])
                self.assertTrue(labels[idx] == expected_labels[sen_idx][idx])
        

if __name__ == '__main__':
    unittest.main()
    