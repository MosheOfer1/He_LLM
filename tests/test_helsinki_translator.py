import unittest
from translation.helsinki_translator import HelsinkiTranslator


class TestHelsinkiTranslator(unittest.TestCase):

    def setUp(self):
        """Set up the HelsinkiTranslator instance for testing."""
        self.translator = HelsinkiTranslator()
        self.sample_text_he = "הילד גדול."
        self.sample_text_en = "The boy is big."

        # Encode the sample text to simulate hidden states
        self.tokens = self.translator.target_to_source_tokenizer(self.sample_text_he, return_tensors="pt",
                                                                 truncation=True, padding=True)
        self.hidden_states = self.translator.target_to_source_model.get_input_embeddings()(self.tokens.input_ids)

    def test_translate_to_target(self):
        """Test the translation from Hebrew (source) to English (target)."""
        translated_text = self.translator.translate_to_target(self.sample_text_he)
        self.assertIsInstance(translated_text, str)
        self.assertNotEqual(translated_text, "")  # Ensure translation is not empty
        self.assertEqual(self.sample_text_en,translated_text)
        print(translated_text)

    def test_translate_to_source(self):
        """Test the translation from English (target) back to Hebrew (source)."""
        translated_text = self.translator.translate_to_source(self.sample_text_en)
        self.assertIsInstance(translated_text, str)
        self.assertNotEqual(translated_text, "")  # Ensure translation is not empty
        self.assertEqual(self.sample_text_he,translated_text)
        print(translated_text)

    def test_translate_hidden_to_source_with_fitting_tensor(self):
        """Test translating hidden states that correspond to a specific sentence back to Hebrew."""
        translated_text = self.translator.translate_hidden_to_source(self.hidden_states)
        self.assertIsInstance(translated_text, str)
        self.assertEqual(self.sample_text_he, translated_text)  # Ensure it matches the original sentence


if __name__ == '__main__':
    unittest.main()
