import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_datasets.create_datasets import read_file_to_string
from my_datasets.combo_model_dataset import ComboModelDataset
from models.custom_model import MyCustomModel

translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
llm_model_name = "facebook/opt-125m"

text_file_path = "my_datasets/SVLM_Hebrew_Wikipedia_Corpus.txt"

customLLM = MyCustomModel(translator1_model_name,
                          translator2_model_name,
                          llm_model_name)

text = read_file_to_string(text_file_path)
split_index = int(len(text) * 0.8)
train_data, eval_data = text[:split_index], text[split_index:]

# Create datasets
train_dataset = ComboModelDataset(
    text=train_data,
    input_tokenizer=customLLM.translator.src_to_target_tokenizer,
    output_tokenizer=customLLM.translator.target_to_src_tokenizer,
)

eval_dataset = ComboModelDataset(
    text=eval_data,
    input_tokenizer=customLLM.translator.src_to_target_tokenizer,
    output_tokenizer=customLLM.translator.target_to_src_tokenizer,
)

customLLM.find_best_hyper_params(train_dataset=train_dataset,
                                 eval_dataset=eval_dataset)
