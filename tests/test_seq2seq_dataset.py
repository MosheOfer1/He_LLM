import unittest
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, OPTForCausalLM
import sys
import os
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from translation.helsinki_translator import HelsinkiTranslator
from llm.llm_wrapper import LLMWrapper
from custom_datasets.seq2seq_dataset import Seq2SeqDataset


class TestSeq2SeqDataset(unittest.TestCase):

    def setUp(self):
        self.translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
        self.translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
        self.translator = HelsinkiTranslator(self.translator1_model_name,
                                             self.translator2_model_name)

        self.llm_model_name = "facebook/opt-125m"
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.llm_model = OPTForCausalLM.from_pretrained(self.llm_model_name)
        self.llm = LLMWrapper(self.llm_model_name, self.llm_tokenizer, self.llm_model)

        # Example text in Hebrew
        self.sentences = [
            "שלום עולם",
            "איך אתה מרגיש היום?",
            "זהו ניסוי במודל תרגום"
        ]

        # Initialize the dataset
        self.dataset = Seq2SeqDataset(
            sentences=self.sentences,
            translator=self.translator,
            llm=self.llm,
            max_seq_len=10
        )

    def test_dataloader(self):
        # Create a RandomSampler for shuffling and getting indices
        sampler = RandomSampler(self.dataset)

        # Create the DataLoader with the sampler
        dataloader = DataLoader(self.dataset, batch_size=2, sampler=sampler)

        # Iterate through the batches
        for batch_idx, batch in enumerate(dataloader):
            self.assertIn('input_ids', batch)
            self.assertIn('labels', batch)

            # Ensure the shapes of input_ids and labels are correct
            input_ids = batch['input_ids']
            labels = batch['labels']

            self.assertIsInstance(input_ids, torch.Tensor)
            self.assertIsInstance(labels, torch.Tensor)

            # Print the batch for reference
            print(f"Batch index: {batch_idx}, Batch: {batch}")

            # Ensure that input_ids and labels are not empty
            self.assertTrue(input_ids.size(0) > 0)
            self.assertTrue(labels.size(0) > 0)

    def test_getitem(self):
        # Check the output of __getitem__
        first_item = self.dataset[0]

        self.assertIn('input_ids', first_item)
        self.assertIn('labels', first_item)

        # Validate that input_ids and labels are not empty
        self.assertTrue(len(first_item['input_ids']) > 0)
        self.assertTrue(isinstance(first_item['labels'], torch.Tensor))

        # Print the first item for reference
        print(f"First item: {first_item}")

        # Additional checks can be done to ensure correctness
        input_hidden_states = first_item['input_ids']
        target_hidden_states = first_item['labels']

        print(input_hidden_states, "\n\n")
        print(target_hidden_states)


if __name__ == '__main__':
    unittest.main()
