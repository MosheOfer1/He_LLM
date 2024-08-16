import unittest
from translation.helsinki_translator import HelsinkiTranslator


class TestHelsinkiTranslator(unittest.TestCase):

    def setUp(self):
        """Set up the HelsinkiTranslator instance for testing."""
        self.src_to_target_translator_model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
        self.target_to_src_translator_model_name = 'Helsinki-NLP/opus-mt-en-he'
        self.translator = HelsinkiTranslator(self.src_to_target_translator_model_name,
                                             self.target_to_src_translator_model_name)
        self.sample_text_he = "הילד גדול."
        self.sample_text_en = "The boy is big."

    def test_injection_second_translator_model(self):
        """
        Test the Second translator injection
        :return:
        """
        translator_first_hs = self.translator.text_to_hidden_states(self.sample_text_en, 0, self.target_to_src_translator_model_name)

        self.translator.inject_hidden_states(translator_first_hs)

        translator_output = self.translator.get_output_by_using_dummy(translator_first_hs.shape[1])
        translator_output = self.translator.decode_logits(from_first=False, logits=translator_output.logits)

        self.assertIsInstance(translator_output, str)
        self.assertGreater(len(translator_output), 0)  # Ensure the decoded text is not empty
        print(translator_output.split(" "))


if __name__ == '__main__':
    unittest.main()
