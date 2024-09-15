import sys
import os
from custom_datasets.create_datasets import read_file_to_string
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.combined_model import MyCustomModel
# Dataset
from custom_datasets.combo_model_dataset import ComboModelDataset
from translation.translator import Translator


def create_datasets_from_txt_file(translator: Translator, text_file_path: str, window_size=30, device='cpu'):
    text = read_file_to_string(text_file_path)

    print(f"len(text) = {len(text)}")

    split_index = int(len(text) * 0.8)
    train_data, eval_data = text[:split_index], text[split_index:]

    # Create datasets
    train_dataset = ComboModelDataset(
        text=train_data,
        input_tokenizer=translator.src_to_target_tokenizer,
        output_tokenizer=translator.target_to_src_tokenizer,
        window_size=window_size,
        device=device
    )

    eval_dataset = ComboModelDataset(
        text=eval_data,
        input_tokenizer=translator.src_to_target_tokenizer,
        output_tokenizer=translator.target_to_src_tokenizer,
        window_size=window_size,
        device=device
    )

    return train_dataset, eval_dataset


def train(model, train_dataset, eval_dataset, batches=32, device='cpu'):
    # Train the model
    model.train_model(train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      output_dir="results",
                      logging_dir="loggings",
                      epochs=5,
                      batch_size=batches,
                      weight_decay=0.01,
                      logging_steps=1000,
                      evaluation_strategy="steps",
                      lr=0.006334926670051613,
                      device=device)
    return model


def save_model(model: MyCustomModel, model_name: str, model_dir: str):
    model_path = os.path.join(model_dir, model_name)  # Save as customLLM.pth in the directory

    # Ensure the directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model state
    torch.save(model.state_dict(), model_path)


def load_model(model_path: str, translator1_model_name, translator2_model_name, llm_model_name, device):
    customLLM = MyCustomModel(translator1_model_name,
                              translator2_model_name,
                              llm_model_name,
                              device=device)

    # Load the saved state dictionary into the model
    customLLM.load_state_dict(torch.load(model_path))
    return customLLM


def predict(model: MyCustomModel, text: str):
    device = model.module.device if hasattr(model, 'module') else model.device

    inputs = model.translator.src_to_target_tokenizer(text, return_tensors="pt").to(device)

    outputs = model(**inputs)
    print(outputs)
