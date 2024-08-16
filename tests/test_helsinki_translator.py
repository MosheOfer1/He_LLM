import unittest
from translation.helsinki_translator import HelsinkiTranslator


class TestHelsinkiTranslator(unittest.TestCase):

    def setUp(self):
        """Set up the HelsinkiTranslator instance for testing."""
        self.src_to_target_translator_model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
        self.target_to_src_translator_model_name = 'Helsinki-NLP/opus-mt-en-he'
        self.translator = HelsinkiTranslator(self.src_to_target_translator_model_name,
                                             self.target_to_src_translator_model_name)
        self.sample_text_en = "The boy is big."

    def test_injection_second_translator_model(self):
        """
        Test the Second translator injection
        :return:
        """
        # Step 1: Get the hidden states from the first layer of the second translator model
        translator_first_hs = self.translator.text_to_hidden_states(self.sample_text_en, 0,
                                                                    self.target_to_src_translator_model_name)

        # Step 2: Inject these hidden states into the model
        self.translator.inject_hidden_states(translator_first_hs)

        # Step 3: Generate logits using a dummy input
        translator_output = self.translator.get_output_by_using_dummy(translator_first_hs.shape[1])

        # Step 4: Decode the logits into a full sentence
        translated_sentence = self.translator.decode_logits(from_first=False, logits=translator_output.logits)

        # Step 5: Validate and print the output
        self.assertIsInstance(translated_sentence, str)
        self.assertGreater(len(translated_sentence), 0)  # Ensure the decoded text is not empty
        print(translated_sentence)  # Print the full sentence


if __name__ == '__main__':
    unittest.main()
