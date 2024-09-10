import sys
import os
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.gpt2_llm import GPT2LLM
from models.combined_model import MyCustomModel
from facade import create_datasets_from_txt_file, save_model, predict, train

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Im working with: {device}")


translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
llm_model_name = "DAMO-NLP-MT/polylm-1.7b"

text_file_path = input("Enter text file path: ")

customLLM = MyCustomModel(translator1_model_name,
                          translator2_model_name,
                          llm_model_name,
                          GPT2LLM,
                          device=device)


train_dataset, eval_dataset = create_datasets_from_txt_file(translator=customLLM.translator, 
                                                            text_file_path=text_file_path,
                                                            window_size=30,
                                                            device=device)

train(model=customLLM,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      batches=16,
      device=device)

# Specify the full file path for the model
model_name = f"{text_file_path[:5]}_{llm_model_name.split('/')[-1].replace('.','_')}.pth"
model_dir = "my_datasets"

save_model(model=customLLM,
           model_name=model_name,
           model_dir=model_dir)
