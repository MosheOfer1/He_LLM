import unittest
import torch.nn as nn

import torch
from transformers import AutoTokenizer, OPTForCausalLM

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_transformers.transformer_1 import Transformer1
from llm.llm_integration import LLMWrapper
from my_datasets.seq2seq_dataset import Seq2SeqDataset
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

    def test_train_transformer1(self):
        transformer = Transformer1(
            translator=self.translator,
            llm=self.llm_integration,
            model_name="transform1_test_model"
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
            toy_inputs,
            toy_targets
        )

        transformer.train_model(toy_ds, toy_ds)

    def test_load_and_evaluate_model(self):
        """Test loading a model by name and running a forward pass in evaluation mode."""
        model_name = "try_model"

        # Load the model
        loaded_model = Transformer1.load_model(model_name=model_name, translator=self.translator,
                                                  llm=self.llm_integration)

        sentence = input("הכנס משפט בעברית: ")
        # Step 1: Get the last hidden state from the first translator model
        with torch.no_grad():
            outputs = self.translator.get_output(from_first=True, text=sentence)
        input_hidden_states = outputs.decoder_hidden_states[-1]  # Shape: (seq_len, hidden_dim)

        # Step 2: Translate the sentence
        translated_text = self.translator.decode_logits(
            tokenizer=self.translator.src_to_target_tokenizer,
            logits=outputs.logits
        )
        print(translated_text)
        # Step 3: Pass the English translation through the LLM and get the first hidden state
        with torch.no_grad():
            self.llm_model.base_model.decoder.layers[self.llm_integration.injected_layer_num].set_injection_state(False)
            target_hidden_states = self.llm_integration.text_to_hidden_states(
                tokenizer=self.llm_integration.tokenizer,
                model=self.llm_model,
                text=translated_text,
                layer_num=0  # Assuming this returns a tensor of shape (seq_len, hidden_dim)
            )
            self.llm_model.base_model.decoder.layers[self.llm_integration.injected_layer_num].set_injection_state(True)

        # Perform a forward pass
        with torch.no_grad():
            output = loaded_model(input_hidden_states)

        # TODO:Calculate the MSE between target_hidden_states and output
        loss_fct = nn.MSELoss()  # Assuming regression task, modify if needed
        loss = loss_fct(output, target_hidden_states)
        print(loss)

        self.llm_integration.inject_hidden_states(output)
        outputs = self.llm_integration.get_output_by_using_dummy(output.shape[1])
        print(self.llm_integration.decode_logits(outputs.logits))

        self.llm_integration.inject_hidden_states(target_hidden_states)
        outputs = self.llm_integration.get_output_by_using_dummy(target_hidden_states.shape[1])
        print(self.llm_integration.decode_logits(outputs.logits))


if __name__ == '__main__':
    unittest.main()
