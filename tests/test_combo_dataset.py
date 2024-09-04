import unittest
from transformers import MarianTokenizer
from torch.utils.data import DataLoader, RandomSampler
from my_datasets.combo_model_dataset import ComboModelDataset


class TestComboModelDataset(unittest.TestCase):

    def setUp(self):
        # Initialize the tokenizers
        self.input_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-he-en")
        self.output_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-he")

        # Example text in Hebrew
        self.text = """במהלך מלחמת העולם השנייה עבד במחלקה למחקר צבאי באוניברסיטת קולומביה
למד צרפתית וספרות בבודפשט ולבסוף הוסמך בלימודים גרמניים באוניברסיטת בלגרד
פרסומים עיקריים מקורות המחשבה הצבאית המודרנית משרד הביטחון ההוצאה לאור
קישורים חיצוניים אתר האינטרנט של המרכז הירושלמי לענייני ציבור ומדינה
אם התכוונתם למושבה האמריקנית שקדמה למדינה ראו הפרובינציה של פנסילבניה
        """

        # Initialize the dataset
        self.dataset = ComboModelDataset(
            text=self.text,
            input_tokenizer=self.input_tokenizer,
            output_tokenizer=self.output_tokenizer,
            window_size=5
        )

    def test_getitem(self):
        # Check the output of __getitem__
        first_item = self.dataset[11]

        self.assertIn('input_ids', first_item)
        self.assertIn('labels', first_item)

        # Validate that input_ids and labels are not empty
        self.assertTrue(len(first_item['input_ids']) > 0)
        self.assertTrue(isinstance(first_item['labels'], int))

        # Additional checks can be done to ensure correctness
        expected_next_token = self.dataset.token_pairs[16][1][0]
        expected_label = self.output_tokenizer(
            text_target=expected_next_token,
            add_special_tokens=False
        )["input_ids"][0]

        with self.output_tokenizer.as_target_tokenizer():
            a = self.output_tokenizer(
                text=expected_next_token,
                add_special_tokens=False
            )["input_ids"][0]

        self.assertEqual(a, expected_label)
        self.assertEqual(first_item['labels'], expected_label)

    def test_dataloader(self):
        # Create a RandomSampler for shuffling and getting indices
        sampler = RandomSampler(self.dataset)

        # Create the DataLoader with the sampler
        dataloader = DataLoader(self.dataset, batch_size=2, sampler=sampler)

        # Iterate through the batches
        for batch_idx, (indices, batch) in enumerate(zip(sampler, dataloader)):
            self.assertIn('input_ids', batch)
            self.assertIn('labels', batch)
            print(f"Batch index: {batch_idx}, Shuffled indices: {indices}")
            print(batch)


if __name__ == '__main__':
    unittest.main()
