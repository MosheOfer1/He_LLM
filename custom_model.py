import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import Trainer, TrainingArguments
from custom_trainers.combined_model_trainer import CombinedTrainer

# Transformers
from custom_transformers.transformer import Transformer

# Translators
from translation.helsinki_translator import HelsinkiTranslator
from llm.facebook_llm import FacebookLLM

# Dataset
from my_datasets.hebrew_dataset_wiki import HebrewDataset


class MyCustomModel(nn.Module):

    def __init__(self,
                src_to_target_translator_model_name,
                target_to_src_translator_model_name,
                llm_model_name,
                pretrained_transformer1_path: str = None,
                pretrained_transformer2_path: str = None):
        
        super(MyCustomModel, self).__init__()

        # Custom Translator
        self.translator = HelsinkiTranslator(src_to_target_translator_model_name,
                                             target_to_src_translator_model_name)
        # Custom LLM
        self.llm = FacebookLLM(llm_model_name)

        # Custom Transformer
        self.transformer = Transformer(translator=self.translator, 
                                       llm=self.llm, 
                                       pretrained_transformer1_path=pretrained_transformer1_path, 
                                       pretrained_transformer2_path=pretrained_transformer2_path)

        # Freeze Translator1 parameters
        self.translator.set_requires_grad(False)

        # Freeze LLM parameters
        self.llm.set_requires_grad(False)

    def forward(self, input_ids = None, text = None, attention_mask=None, labels = None, class_weights = None) -> torch.Tensor:
                
        
        # Remove batch if batch_size=1
        input_ids = input_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        
        # print(f"input_ids shape: {input_ids.shape}")
        # print(f"attention_mask shape: {attention_mask.shape}")
        
        if text:
            # Get hidden states from text
            translator_last_hs = self.translator.text_to_hidden_states(text, -1, self.translator.src_to_target_tokenizer,
                                                                    self.translator.src_to_target_model, False)
        else:                    
            translator_last_hs = self.translator.input_ids_to_hidden_states(input_ids, -1, self.translator.src_to_target_tokenizer,
                                                                   self.translator.src_to_target_model, False, attention_mask=attention_mask)

        # Transform to llm first hidden states
        transformed_to_llm_hs = self.transformer.transformer1.forward(translator_last_hs)

        # Inject the new hidden states to the llm first layer
        self.llm.inject_hidden_states(transformed_to_llm_hs)

        token_num = transformed_to_llm_hs.shape[1]
        
        # print(f"\n\n transformed_to_llm_hs.shape = {transformed_to_llm_hs.shape}, token_num = {token_num}\n\n")
        
        # Input dummy text but the it is ignored and uses the injected 
        llm_outputs = self.llm.get_output_by_using_dummy(token_num=token_num)

        # Extract the last hidden states
        llm_last_hidden_state = llm_outputs.hidden_states[-1]
        # llm_last_hidden_state = llm_outputs.hidden_states[-1][:,-1,:].unsqueeze(0)
        
        # print(f"llm_last_hidden_state.shape = {llm_last_hidden_state.shape}")

        # Transform to translator first hidden states
        transformed_to_translator_hs = self.transformer.transformer2.forward(llm_last_hidden_state)
        
        # print(f"transformed_to_translator_hs.shape = {transformed_to_translator_hs.shape}")

        # Inject the new hidden states to translator2 first layer
        self.translator.inject_hidden_states(transformed_to_translator_hs)
        
        token_num = transformed_to_translator_hs.shape[1]
        
        # print(f"\n\n transformed_to_translator_hs.shape = {transformed_to_translator_hs.shape}, token_num2 = {token_num}\n\n")
        
        # Input dummy text but the it is ignored and uses the injected 
        translator_outputs = self.translator.get_output_by_using_dummy(token_num=token_num)
        
        # print(f"translator_outputs = {translator_outputs}")

        return translator_outputs

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        
        return {"f1": f1}
        
    def create_trainer(
        self, train_dataset: Dataset, eval_dataset: Dataset,
        output_dir: str, logging_dir: str, epochs: int = 5, 
        batch_size: int = 1, weight_decay: float = 0.01,
        logging_steps: int = 1000, evaluation_strategy: str = "steps",
        lr=0.006334926670051613, max_grad_norm: float = 1.0, 
        optimizer = None, scheduler = None
        ) -> CombinedTrainer:
        
        epoch = len(train_dataset)
        total_steps = int(epoch // batch_size * epochs)
        warmup_steps = int(0.1 * total_steps)
        
        print(f"\n\n epoch = {epoch}, total = {total_steps}, warmup = {warmup_steps} \n\n")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=f"./{output_dir}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=f"./{logging_dir}",
            logging_steps=logging_steps,
            evaluation_strategy=evaluation_strategy,
            eval_steps=epoch,
            learning_rate=lr,
            max_grad_norm=max_grad_norm,
            fp16=True, # nable mixed precision training
        )
        

        trainer = CombinedTrainer(
        
            model=self,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            total_steps=total_steps,
        )
        
        return trainer
        
    def train_model(
        self, train_dataset: Dataset, eval_dataset: Dataset,
        output_dir: str, logging_dir: str, epochs: int = 5, 
        batch_size: int = 1, weight_decay: float = 0.01,
        logging_steps: int = 1000, evaluation_strategy: str = "steps",
        lr=0.006334926670051613, max_grad_norm: float = 1.0, 
        optimizer = None, scheduler = None
        ):

        trainer = self.create_trainer(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            logging_dir=logging_dir,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            evaluation_strategy=evaluation_strategy,
            lr=lr,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        trainer.train()
        
        # Save the finetuned model and tokenizer with a new name
        pretrained_model_dir = f"./pretrained_models/end_to_end_model/{output_dir}"
        trainer.save_model(pretrained_model_dir)