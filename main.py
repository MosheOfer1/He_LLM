import sys
import os
from my_datasets.create_datasets import read_file_to_string
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.custom_model import MyCustomModel
from custom_trainers.combined_model_trainer import CombinedTrainer
# Dataset
from my_datasets.combo_model_dataset import ComboModelDataset


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

trainer: CombinedTrainer = customLLM.create_trainer(train_dataset=train_dataset,
                                                    eval_dataset=eval_dataset,
                                                    output_dir="results",
                                                    logging_dir="loggings",
                                                    epochs=5,
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
                      epochs=5,
                      batch_size=1,
                      weight_decay=0.01,
                      logging_steps=1000,
                      evaluation_strategy="steps",
                      lr=0.00001)

# {'loss': 6.314, 'grad_norm': 0.19041453301906586, 'learning_rate': 0.004873020515424318, 'epoch': 1.25}
# {'loss': 6.2977, 'grad_norm': 0.3401281535625458, 'learning_rate': 0.0032486803436162118, 'epoch': 2.5}
# {'loss': 6.6034, 'grad_norm': 0.16888704895973206, 'learning_rate': 0.0016243401718081059, 'epoch': 3.75}
# {'loss': 6.2835, 'grad_norm': 0.39941009879112244, 'learning_rate': 0.0, 'epoch': 5.0}
