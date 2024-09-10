import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.opt_llm import OptLLM
from my_datasets.create_datasets import read_file_lines
import torch

from my_datasets.seq2seq_dataset import Seq2SeqDataset
from translation.helsinki_translator import HelsinkiTranslator


from custom_transformers.transformer_1 import Transformer1
# Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Im working with: {device}")

translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
llm_model_name = "facebook/opt-125m"
text_file_path = "../my_datasets/7k_hebrew_wiki_text.txt"


llm = OptLLM(llm_model_name,
             device=device)

translator = HelsinkiTranslator(
    translator1_model_name,
    translator2_model_name,
    device=device
)

trans1 = Transformer1(
    translator,
    llm,
    device=device
)

text = read_file_lines(text_file_path)

print(f"len(text) = {len(text)}")

split_index = int(len(text) * 0.8)
train_data, eval_data = text[:split_index], text[split_index:]

# Create datasets
train_dataset = Seq2SeqDataset(
    sentences=train_data,
    translator=translator,
    llm=llm,
    max_seq_len=18
)

eval_dataset = Seq2SeqDataset(
    sentences=eval_data,
    translator=translator,
    llm=llm,
    max_seq_len=18
)

trans1.train_model(train_dataset, eval_dataset)
