import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from llm.opt_llm import OptLLM
from custom_datasets.create_datasets import read_file_lines
import torch

from custom_datasets.seq2seq_dataset import Seq2SeqDataset
from translation.helsinki_translator import HelsinkiTranslator


from custom_transformers.transformer_1 import Transformer1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Im working with: {device}")

# Define local paths
translator1_local_path = "/home/management/scratch/talias/opus-mt-tc-big-he-en/"
translator2_local_path = "/home/management/scratch/talias/opus-mt-en-he/"
llm_local_path = "/home/management/scratch/talias/opt-350m/"
text_file_path = "my_datasets/ynet_256k.txt"

# Define default Hugging Face model names
default_translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
default_translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
default_llm_model_name = "facebook/opt-350m"

# Check if the local paths exist, otherwise assign the Hugging Face model name
translator1_model_name = translator1_local_path if os.path.exists(translator1_local_path) else default_translator1_model_name
translator2_model_name = translator2_local_path if os.path.exists(translator2_local_path) else default_translator2_model_name
llm_model_name = llm_local_path if os.path.exists(llm_local_path) else default_llm_model_name


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
    device=device,
    nhead=8,
    num_layers=6
)

text = read_file_lines(text_file_path)

print(f"len(text) = {len(text)}")

split_index = int(len(text) * 0.9)
train_data, eval_data = text[:split_index], text[split_index:]
print(f"Train number of sentences = {len(train_data)}")
print(f"Eval number of sentences = {len(eval_data)}")

# Create datasets
train_dataset = Seq2SeqDataset(
    sentences=train_data,
    translator=translator,
    llm=llm,
)

eval_dataset = Seq2SeqDataset(
    sentences=eval_data,
    translator=translator,
    llm=llm,
)

trans1.train_model(train_dataset, eval_dataset, 6)
