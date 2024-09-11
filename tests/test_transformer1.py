import sys
import os
import unittest
import torch

# Assuming the necessary paths are correctly set up.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.opt_llm import OptLLM
from custom_datasets.seq2seq_dataset import Seq2SeqDataset
from translation.helsinki_translator import HelsinkiTranslator
from custom_transformers.transformer_1 import Transformer1


class TestTransformer1(unittest.TestCase):
    def setUp(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running test on: {device}")

        # Models and Translator setup
        self.translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
        self.translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
        self.llm_model_name = "facebook/opt-125m"

        self.llm = OptLLM(self.llm_model_name, device=device)
        self.translator = HelsinkiTranslator(self.translator1_model_name, self.translator2_model_name, device=device)
        self.trans1 = Transformer1(self.translator, self.llm, device=device, num_layers=1, nhead=1)

        # Short sentences in Hebrew for testing
        self.sentences = [
            "אני אוהב ללמוד.",
            "הכלב רץ מהר מאוד ולא נופל.",
            "הבית גדול מאוד.",
            "מזג האוויר יפה היום.",
            "אני אוהב ללמוד נושאים חדשים כמו מתמטיקה ומדעי המחשב.",
            "הכלב רץ מהר אחרי הכדור ברחבי הפארק המלא בשמש.",
            "הבית הגדול מאוד נמצא בקצה הרחוב, ליד הגן הציבורי הירוק.",
            "מזג האוויר יפה היום, השמש זורחת והשמיים בהירים ללא עננים.",
            "הספר הזה מעניין מאוד, יש בו סיפור עמוק על חברות ואומץ לב.",
            "בגרמניה שנתיים לאחר תחילת השפל הכלכלי היה המצב גרוע ביותר",
            "בדיון התקבלה עמדתו התקיפה של צ'רצ'יל שהתנגד לכל נסיון לפשרה"
        ]

    def test_training_with_small_sentences(self):
        split_index = int(len(self.sentences) * 0.8)
        train_data, eval_data = self.sentences[:split_index], self.sentences[split_index:]

        # Create datasets with smaller sentences
        train_dataset = Seq2SeqDataset(
            sentences=train_data,
            translator=self.translator,
            llm=self.llm,
            max_seq_len=10
        )

        eval_dataset = Seq2SeqDataset(
            sentences=eval_data,
            translator=self.translator,
            llm=self.llm,
            max_seq_len=10
        )

        # Perform training
        self.trans1.train_model(train_dataset, eval_dataset)

        # Add basic assertions, for example, checking that training completes successfully
        self.assertTrue(True, "Training completed without errors.")


if __name__ == '__main__':
    unittest.main()
