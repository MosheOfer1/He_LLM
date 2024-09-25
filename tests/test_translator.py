import unittest
import sys
import os

from transformers import MarianMTModel, MarianTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from translation.helsinki_translator import HelsinkiTranslator


class TestTranslator(unittest.TestCase):

    def setUp(self):
        """Set up the HelsinkiTranslator instance for testing."""
        self.src_to_target_translator_model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
        self.target_to_src_translator_model_name = 'Helsinki-NLP/opus-mt-en-he'
        self.translator = HelsinkiTranslator(self.src_to_target_translator_model_name,
                                             self.target_to_src_translator_model_name)
        self.sample_text_he = "כמה פעמים עוד אפשר?"
        self.sample_text_en = "The boy is big."

    def test_injection_second_translator_model(self):
        """Test the Second translator injection"""
        # Load the tokenizer and model
        tokenizer = MarianTokenizer.from_pretrained(self.target_to_src_translator_model_name)
        model = MarianMTModel.from_pretrained(self.target_to_src_translator_model_name, output_hidden_states=True)
        translator2 = HelsinkiTranslator(self.src_to_target_translator_model_name,
                                         self.target_to_src_translator_model_name)
        # Step 1: Get the hidden states from the first layer of the second translator model
        translator_first_hs = translator2.text_to_hidden_states(
            self.sample_text_en,
            0,
            tokenizer,
            model
        )

        # Step 2: Inject these hidden states into the model
        self.translator.inject_hidden_states(translator_first_hs)

        # Step 3: Generate logits using a dummy input
        translator_output = self.translator.get_output_by_using_dummy(translator_first_hs.shape[1])

        # Step 4: Decode the logits into text
        translated_text = self.translator.decode_logits(tokenizer=self.translator.target_to_src_tokenizer,
                                                        logits=translator_output.logits)

        # Step 5: Validate and print the output
        self.assertIsInstance(translated_text, str)
        self.assertGreater(len(translated_text), 0)  # Ensure the decoded text is not empty
        print(translated_text)

    def test_injection_second_translator_model_input_ids(self):
        """Test the Second translator injection using input_ids_to_hidden_states"""
        # Load the tokenizer and model
        tokenizer = MarianTokenizer.from_pretrained(self.target_to_src_translator_model_name)
        model = MarianMTModel.from_pretrained(self.target_to_src_translator_model_name, output_hidden_states=True)
        translator2 = HelsinkiTranslator(self.src_to_target_translator_model_name,
                                         self.target_to_src_translator_model_name)

        # Step 1: Tokenize the input text to get input_ids
        inputs = tokenizer(self.sample_text_en, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Step 2: Get the hidden states from the first layer of the second translator model using input_ids
        translator_first_hs, _ = translator2.input_ids_to_hidden_states(
            input_ids,
            layer_num=0,  # The layer number to extract the hidden states from
            tokenizer=tokenizer,
            model=model
        )

        # Step 3: Inject these hidden states into the model
        self.translator.inject_hidden_states(translator_first_hs)

        # Step 4: Generate logits using a dummy input
        translator_output = self.translator.get_output_by_using_dummy(translator_first_hs.shape[1])

        # Step 5: Decode the logits into text
        translated_text = self.translator.decode_logits(tokenizer=self.translator.target_to_src_tokenizer,
                                                        logits=translator_output.logits)

        # Step 6: Validate and print the output
        self.assertIsInstance(translated_text, str)
        self.assertGreater(len(translated_text), 0)  # Ensure the decoded text is not empty
        print(translated_text)

    def test_get_output_from_first(self):
        """Test the get_output method when using the first translator."""
        output = self.translator.get_output(from_first=True, text=self.sample_text_he)

        self.assertIsNotNone(output.logits)  # Ensure logits are generated

        translated_text = self.translator.decode_logits(tokenizer=self.translator.src_to_target_tokenizer,
                                                        logits=output.logits)
        self.assertIsInstance(translated_text, str)
        self.assertGreater(len(translated_text), 0)  # Ensure the decoded text is not empty
        print("From first:", translated_text)

    def test_get_output_from_second(self):
        """Test the get_output method when using the second translator."""
        output = self.translator.get_output(from_first=False, text=self.sample_text_en)

        self.assertIsNotNone(output.logits)  # Ensure logits are generated
        translated_text = self.translator.decode_logits(tokenizer=self.translator.target_to_src_tokenizer,
                                                        logits=output.logits)
        self.assertIsInstance(translated_text, str)
        self.assertGreater(len(translated_text), 0)  # Ensure the decoded text is not empty
        print("From second:", translated_text)

    def test_various_text_sizes(self):
        """Test the translator with different sizes of text."""
        # Define texts of various sizes in Hebrew for the first translator
        short_text_he = "שלום."
        medium_text_he = "הילד רץ לבית הספר מוקדם בבוקר."
        long_text_he = "בעולם מושלם היינו יכולים להשיג את כל מה שאנחנו רוצים, אבל המציאות היא אחרת וישנם אתגרים רבים שעלינו להתגבר עליהם."

        # Define texts of various sizes in English for the second translator
        short_text_en = "Hi."
        medium_text_en = "The boy runs to school early in the morning."
        long_text_en = "In a perfect world, we could achieve everything we want, but reality is different, and there are many challenges we need to overcome."

        # Test with the first translator (Hebrew input)
        for text in [short_text_he, medium_text_he, long_text_he]:
            with self.subTest(text=text):
                output = self.translator.get_output(from_first=True, text=text)
                self.assertIsNotNone(output.logits)
                translated_text = self.translator.decode_logits(tokenizer=self.translator.src_to_target_tokenizer,
                                                                logits=output.logits)
                self.assertIsInstance(translated_text, str)
                self.assertGreater(len(translated_text), 0)  # Ensure the decoded text is not empty
                print(f"From first, size={len(text)}:", translated_text)

        # Test with the second translator (English input)
        for text in [short_text_en, medium_text_en, long_text_en]:
            with self.subTest(text=text):
                output = self.translator.get_output(from_first=False, text=text)
                self.assertIsNotNone(output.logits)
                translated_text = self.translator.decode_logits(tokenizer=self.translator.target_to_src_tokenizer,
                                                                logits=output.logits)
                self.assertIsInstance(translated_text, str)
                self.assertGreater(len(translated_text), 0)  # Ensure the decoded text is not empty
                print(f"From second, size={len(text)}:", translated_text)

    def test_batch_translation_with_first_translator(self):
        """Test the batch translation from Hebrew to English using the first translator with get_output."""

        # Sample batch of Hebrew sentences
        batch_sentences_he = [
            "הילד רץ.",
            "החתול ישן.",
            "אני לומד פייתון.",
            "השמש זורחת היום."
        ]

        # Step 1: Get the output from the first translator using the get_output method for batch processing
        output = self.translator.get_output(from_first=True, text=batch_sentences_he)

        # Step 2: Ensure logits are generated
        self.assertIsNotNone(output.logits)

        # Step 3: Decode the logits into text
        translated_texts = self.translator.decode_logits(tokenizer=self.translator.src_to_target_tokenizer,
                                                         logits=output.logits)

        # Step 4: Validate and print the output for the batch
        self.assertIsInstance(translated_texts, str)
        self.assertGreater(len(translated_texts), 0)  # Ensure the decoded text is not empty
        print("Batch translation from Hebrew to English:", translated_texts)

    def test_batch_input_ids_to_hidden_states_and_inject(self):
        """Test the batch translation using input_ids_to_hidden_states with the first translator and inject the
        hidden states. """

        # Sample batch of Hebrew sentences
        batch_sentences_he = [
            "הילד רץ.",
            "החתול ישן.",
            "אני לומד פייתון.",
            "השמש זורחת היום."
        ]

        # Step 1: Tokenize the input text batch to get input_ids with padding and truncation
        tokenizer = self.translator.src_to_target_tokenizer
        inputs = tokenizer(batch_sentences_he, return_tensors="pt", padding=True, truncation=True)

        # Extract the input_ids and attention_mask from the tokenizer output
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        with self.translator.injection_state():
            # Step 2: Get the hidden states from the first layer of the first translator model using input_ids
            translator_first_hs, _ = self.translator.input_ids_to_hidden_states(
                input_ids=input_ids,
                layer_num=0,  # Extract hidden states from the first layer
                tokenizer=tokenizer,
                model=self.translator.src_to_target_model,
                from_encoder=True,
                attention_mask=attention_mask
            )

        # Ensure the hidden states are generated and not None
        self.assertIsNotNone(translator_first_hs)
        self.assertEqual(translator_first_hs.shape[0], input_ids.shape[0])  # Check batch size consistency


if __name__ == '__main__':
    unittest.main()
