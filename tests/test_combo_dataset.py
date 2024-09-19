import unittest

import torch
from transformers import MarianTokenizer, MarianMTModel
from torch.utils.data import DataLoader, RandomSampler

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_datasets.combo_model_dataset_window import ComboModelDataset


class TestComboModelDataset(unittest.TestCase):

    def setUp(self):
        # Initialize the tokenizers
        src_to_target_translator_model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
        target_to_src_translator_model_name = 'Helsinki-NLP/opus-mt-en-he'
        self.input_tokenizer = MarianTokenizer.from_pretrained(src_to_target_translator_model_name)
        self.output_tokenizer = MarianTokenizer.from_pretrained(target_to_src_translator_model_name)

        # Initialize the models correctly
        self.first_model = MarianMTModel.from_pretrained(src_to_target_translator_model_name,
                                                         output_hidden_states=True)
        self.second_model = MarianMTModel.from_pretrained(target_to_src_translator_model_name,
                                                          output_hidden_states=True)

        # Example text in Hebrew
        self.text = ["לא משנה עם מי תלך לעולם לא"]
        self.window_size = 3
        
        # Initialize the dataset
        self.dataset = ComboModelDataset(
            text_list=self.text,
            input_tokenizer=self.input_tokenizer,
            output_tokenizer=self.output_tokenizer,
            window_size=self.window_size
        )

    def test_dataloader(self):
        # Create a RandomSampler for shuffling and getting indices
        sampler = RandomSampler(self.dataset)

        # Create the DataLoader with the sampler
        dataloader = DataLoader(self.dataset, batch_size=2, sampler=sampler)

        # Iterate through the batches
        for batch_idx, (indices, batch) in enumerate(zip(sampler, dataloader)):
            self.assertIn('input_ids', batch)
            self.assertIn('labels', batch)

            # Print the batch index and shuffled indices
            print(f"Batch index: {batch_idx}, Shuffled indices: {indices}")

            # Translate each input_ids in the batch to English using the first model
            input_ids_list = batch['input_ids']
            for input_ids in input_ids_list:
                
                print(input_ids.shape)
                
                # Step 1: Convert input_ids back to Hebrew text using input_tokenizer
                hebrew_text = self.input_tokenizer.decode(input_ids[0], skip_special_tokens=True)
                print(f"Original Hebrew sentence: {hebrew_text}")

                # Step 2: Tokenize the Hebrew text for the second translation process
                hebrew_tokenized_input = self.input_tokenizer(hebrew_text, return_tensors="pt")
                print("**hebrew_tokenized_input", hebrew_tokenized_input)

                # Step 3: Run the tokenized Hebrew text through the model to get translated output (English)
                with torch.no_grad():
                    outputs = self.first_model.generate(**hebrew_tokenized_input,
                                                        max_length=50)  # Increase max_length

                # Step 4: Decode the translated output (in English)
                translated_sentence1 = self.input_tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Translated sentence (to English): {translated_sentence1}")

                # Simulate the batch structure
                batch_artificial = {
                    'input_ids': input_ids[0].unsqueeze(0),  # Add batch dimension
                }

                with torch.no_grad():
                    outputs = self.first_model.generate(**batch_artificial,
                                                        max_length=50)  # Increase max_length

                # Step 4: Decode the translated output (in English)
                translated_sentence2 = self.input_tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Translated sentence (to English) with batch_artificial: {translated_sentence2}")
                self.assertEqual(translated_sentence1, translated_sentence2)

            # Print the entire batch for reference
            print(batch)

    def test_getitem(self):
        # Check the output of __getitem__
        first_item = self.dataset[0]

        self.assertIn('input_ids', first_item)
        self.assertIn('labels', first_item)

        # Validate that input_ids and labels are not empty
        self.assertTrue(len(first_item['input_ids']) > 0)
        self.assertTrue(isinstance(first_item['labels'], torch.Tensor))

        # Additional checks can be done to ensure correctness
        # expected_next_token = self.dataset.windows[0,:,0]
        expected_next_token = [pair[1][0] for pair in self.dataset.windows[0]]
        
        print(f"expected_next_token = {expected_next_token}")
        
        expected_label = self.output_tokenizer(
            text_target=expected_next_token,
            add_special_tokens=False,
            return_tensors='pt'
        )["input_ids"]

        with self.output_tokenizer.as_target_tokenizer():
            a = self.output_tokenizer(
                text=expected_next_token,
                add_special_tokens=False,
                return_tensors='pt'
            )["input_ids"]

        self.assertTrue(torch.equal(a, expected_label))
        self.assertTrue(torch.equal(first_item['labels'][0], expected_label[:,0]))


if __name__ == '__main__':
    unittest.main()
    