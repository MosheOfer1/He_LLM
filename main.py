import sys
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_model import MyCustomModel
from custom_trainers.combined_model_trainer import CombinedTrainer

# Dataset
from my_datasets.hebrew_dataset_wiki import HebrewDataset


translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
llm_model_name = "facebook/opt-125m"

customLLM = MyCustomModel(translator1_model_name,
                            translator2_model_name,
                            llm_model_name)



# print(f"\n\n len(customLLM.parameters()) = {len(list(customLLM.parameters()))}\n\n")

# # Print the parameter names for the model customLLM
# for name, param in customLLM.named_parameters():
#     if param.requires_grad:
#         print(name)


data = pd.read_csv("my_datasets/wikipedia_data.csv")

# Choose a random sample of 1000 rows
sampled_data = data.sample(n=1250, random_state=42)

# Split the data into training and evaluation sets
train_data, eval_data = train_test_split(sampled_data, test_size=0.2)

# Create datasets
train_dataset = HebrewDataset(data=train_data, 
                              input_tokenizer=customLLM.translator.src_to_target_tokenizer, 
                              output_tokenizer=customLLM.translator.target_to_src_tokenizer, 
                              max_length=20)

eval_dataset = HebrewDataset(data=eval_data, 
                             input_tokenizer=customLLM.translator.src_to_target_tokenizer, 
                             output_tokenizer=customLLM.translator.target_to_src_tokenizer, 
                             max_length=20)


trainer: CombinedTrainer = customLLM.create_trainer(train_dataset=train_dataset, 
                      eval_dataset=eval_dataset, 
                      output_dir="results", 
                      logging_dir="loggings",
                      epochs= 5,
                      batch_size=1,
                      weight_decay=0.01,
                      logging_steps=1000,
                      evaluation_strategy="steps",
                      lr=0.006334926670051613)

# Train the model
customLLM.train_model(train_dataset=train_dataset, 
                      eval_dataset=eval_dataset, 
                      output_dir="results", 
                      logging_dir="loggings",
                      epochs= 5,
                      batch_size=1,
                      weight_decay=0.01,
                      logging_steps=1000,
                      evaluation_strategy="steps",
                      lr=0.00001)

# {'loss': 6.314, 'grad_norm': 0.19041453301906586, 'learning_rate': 0.004873020515424318, 'epoch': 1.25}
# {'loss': 6.2977, 'grad_norm': 0.3401281535625458, 'learning_rate': 0.0032486803436162118, 'epoch': 2.5}
# {'loss': 6.6034, 'grad_norm': 0.16888704895973206, 'learning_rate': 0.0016243401718081059, 'epoch': 3.75}
# {'loss': 6.2835, 'grad_norm': 0.39941009879112244, 'learning_rate': 0.0, 'epoch': 5.0}