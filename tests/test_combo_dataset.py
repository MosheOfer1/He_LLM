import unittest
from transformers import MarianTokenizer
from torch.utils.data import DataLoader
from my_datasets.combo_model_dataset import ComboModelDataset


class TestComboModelDataset(unittest.TestCase):

    def setUp(self):
        # Initialize the tokenizers
        self.input_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-he-en")
        self.output_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-he")

        # Example text in Hebrew
        self.text = "המשפט הזה כתוב בעברית והוא למטרת בדיקה."

        # Initialize the dataset
        self.dataset = ComboModelDataset(
            text=self.text,
            input_tokenizer=self.input_tokenizer,
            output_tokenizer=self.output_tokenizer,
            window_size=5
        )

    def test_len(self):
        # Check if the length of the dataset is correct
        expected_length = len(self.input_tokenizer.encode(self.text, add_special_tokens=True)) - 5
        self.assertEqual(len(self.dataset), expected_length)

    def test_getitem(self):
        # Check the output of __getitem__
        first_item = self.dataset[0]

        self.assertIn('input_ids', first_item)
        self.assertIn('labels', first_item)

        # Validate that input_ids and labels are not empty
        self.assertTrue(len(first_item['input_ids']) > 0)
        self.assertTrue(isinstance(first_item['labels'], int))

        # Additional checks can be done to ensure correctness
        input_ids = first_item['input_ids']
        expected_next_token_id = self.dataset.data[5]
        expected_next_token = self.input_tokenizer.decode([expected_next_token_id], skip_special_tokens=True)
        with self.output_tokenizer.as_target_tokenizer():
            expected_label = self.output_tokenizer.encode(expected_next_token, add_special_tokens=False)[0]

        self.assertEqual(first_item['labels'], expected_label)

    def test_dataloader(self):
        # Test if the DataLoader works with the dataset
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)

        for batch in dataloader:
            self.assertIn('input_ids', batch)
            self.assertIn('labels', batch)
            self.assertEqual(len(batch['input_ids'][0]), 2)
            self.assertEqual(len(batch['labels']), 2)
            break  # Only check the first batch


if __name__ == '__main__':
    unittest.main()
