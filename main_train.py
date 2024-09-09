import sys
import os
from my_datasets.create_datasets import read_file_to_string
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.combined_model import MyCustomModel
from facade import create_datasets_from_txt_file, save_model, predict, train

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Im working with: {device}")


translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
llm_model_name = "facebook/opt-125m"

# text_file_path = "my_datasets/book.txt"
text_file_path = "my_datasets/hebrew_text_for_tests.txt"
# text_file_path = "my_datasets/7k_hebrew_wiki_text.txt"
# text_file_path = "my_datasets/SVLM_Hebrew_Wikipedia_Corpus.txt"

customLLM = MyCustomModel(translator1_model_name,
                          translator2_model_name,
                          llm_model_name,
                          device=device)


train_dataset, eval_dataset = create_datasets_from_txt_file(translator=customLLM.translator, 
                                                            text_file_path=text_file_path,
                                                            window_size=30,
                                                            device=device)

train(model=customLLM,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      batches=32,
      device=device)

# # Specify the full file path for the model
model_name = "customLLM_batches32_window30.pth"
model_dir = "/home/ddn1/Documents/GitHub/He_LLM/output_dir_orel_moshe"

save_model(model=customLLM,
           model_name=model_name,
           model_dir=model_dir)
