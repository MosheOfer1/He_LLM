import random
import unittest

import torch
from transformers import AutoTokenizer, OPTForCausalLM
from custom_transformers.base_transformer import Seq2SeqDataset
from custom_transformers.transformer_1 import Transformer1
from llm.llm_integration import LLMWrapper
from translation.helsinki_translator import HelsinkiTranslator


class TestBaseTransformer(unittest.TestCase):
    def setUp(self):
        """Set up the CustomModel instance for testing."""

        self.translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
        self.translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
        self.translator = HelsinkiTranslator(self.translator1_model_name,
                                             self.translator2_model_name)

        self.llm_model_name = "facebook/opt-125m"
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.llm_model = OPTForCausalLM.from_pretrained(self.llm_model_name)
        self.llm_integration = LLMWrapper(self.llm_model_name, self.llm_tokenizer, self.llm_model)

    def test_train_model(self):
        transformer = Transformer1(
            translator=self.translator,
            llm=self.llm_integration
        )
        seq_len = 4
        batch_size = 5

        # Generate toy data as tensors
        toy_inputs = torch.zeros((batch_size, seq_len, transformer.input_dim), dtype=torch.float)
        toy_targets = torch.zeros((batch_size, seq_len, transformer.output_dim), dtype=torch.float)

        for i in range(batch_size):
            for j in range(seq_len):
                toy_inputs[i, j] = 0.1 * (i + j + 1) + torch.arange(transformer.input_dim, dtype=torch.float)
                toy_targets[i, j] = 0.1 * (i + j + 2) + torch.arange(transformer.output_dim, dtype=torch.float)

        # Create the dataset
        toy_ds = Seq2SeqDataset(
            inputs=toy_inputs,
            targets=toy_targets
        )

        transformer.train_model(toy_ds)


if __name__ == '__main__':
    unittest.main()
