import sys
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_model import MyCustomModel

# Dataset
from my_datasets.hebrew_dataset_wiki import HebrewDataset


translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
llm_model_name = "facebook/opt-125m"

customLLM = MyCustomModel(translator1_model_name,
                            translator2_model_name,
                            llm_model_name)

print(f"len(customLLM.parameters()) = {len(list(customLLM.parameters()))}")

# Translator1 parameters
for param in customLLM.parameters():
    print(param.requires_grad)
    
# data = pd.read_csv("my_datasets/wikipedia_data.csv").head(1000)


# # Split the data into training and evaluation sets
# train_data, eval_data = train_test_split(data, test_size=0.2)

# # Create datasets
# train_dataset = HebrewDataset(data=train_data, 
#                               input_tokenizer=customLLM.translator.src_to_target_tokenizer, 
#                               output_tokenizer=customLLM.translator.target_to_src_tokenizer, 
#                               max_length=20)

# eval_dataset = HebrewDataset(data=eval_data, 
#                              input_tokenizer=customLLM.translator.src_to_target_tokenizer, 
#                              output_tokenizer=customLLM.translator.target_to_src_tokenizer, 
#                              max_length=20)


    
# # Train the model
# customLLM.train_model(train_dataset=train_dataset, 
#                       eval_dataset=eval_dataset, 
#                       output_dir="results", 
#                       logging_dir="loggings",
#                       epochs= 5,
#                       batch_size=1,
#                       warmup_steps=100,
#                       weight_decay=0.01,
#                       logging_steps=1000,
#                       evaluation_strategy="steps",
#                       save_steps=4000,
#                       lr=0.006334926670051613)


# {'loss': 8.7391, 'grad_norm': 0.0, 'learning_rate': 0.004873020515424318, 'epoch': 1.25}
