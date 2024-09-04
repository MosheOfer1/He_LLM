import torch
import torch.nn as nn
import os
import sys
from my_datasets.combo_model_dataset import ComboModelDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import TrainingArguments
from custom_trainers.combined_model_trainer import CombinedTrainer

# Transformers
from custom_transformers.transformer import Transformer

# Translators
from translation.helsinki_translator import HelsinkiTranslator
from llm.facebook_llm import FacebookLLM

# Optuna best hypers finder
from utilty.BestHyper import BestHyper


class MyCustomModel(nn.Module, BestHyper):
    def __init__(self,
                 src_to_target_translator_model_name,
                 target_to_src_translator_model_name,
                 llm_model_name,
                 pretrained_transformer1_path: str = None,
                 pretrained_transformer2_path: str = None,
                 device = 'cpu'):

        print(f"MyCustomModel.__init__ - uses: {device}")
        
        nn.Module.__init__(self)

        self.device = device
        
        # Custom Translator
        self.translator = HelsinkiTranslator(src_to_target_translator_model_name,
                                             target_to_src_translator_model_name,
                                             device=device)
        # Custom LLM
        self.llm = FacebookLLM(llm_model_name,
                               device=device)

        # Custom Transformer
        self.transformer = Transformer(translator=self.translator,
                                       llm=self.llm,
                                       pretrained_transformer1_path=pretrained_transformer1_path,
                                       pretrained_transformer2_path=pretrained_transformer2_path,
                                       device=device)

        # Freeze Translator1 parameters
        self.translator.set_requires_grad(False)

        # Freeze LLM parameters
        self.llm.set_requires_grad(False)

    def forward(self, input_ids, text=None, attention_mask=None, labels=None) -> torch.Tensor:
                
        input_ids = input_ids.to(self.device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if text:
            # Get hidden states from text
            translator_last_hs = self.translator.text_to_hidden_states(text, -1,
                                                                       self.translator.src_to_target_tokenizer,
                                                                       self.translator.src_to_target_model, False,
                                                                       attention_mask=attention_mask).to(self.device)
        else:
            translator_last_hs = self.translator.input_ids_to_hidden_states(input_ids, -1,
                                                                            self.translator.src_to_target_tokenizer,
                                                                            self.translator.src_to_target_model, False,
                                                                            attention_mask=attention_mask).to(self.device)

        
        # Transform to llm first hidden states
        transformed_to_llm_hs = self.transformer.transformer1.forward(translator_last_hs)#.to(self.device)
        
        # Inject the new hidden states to the llm first layer
        self.llm.inject_hidden_states(transformed_to_llm_hs)

        token_num = transformed_to_llm_hs.shape[1]
        
        # Input dummy text but the it is ignored and uses the injected 
        llm_outputs = self.llm.get_output_by_using_dummy(token_num=token_num)

        # Extract the last hidden states
        llm_last_hidden_state = llm_outputs.hidden_states[-1]

        
        # Transform to translator first hidden states
        transformed_to_translator_hs = self.transformer.transformer2.forward(llm_last_hidden_state).to(self.device)

        # Inject the new hidden states to translator2 first layer
        self.translator.inject_hidden_states(transformed_to_translator_hs)

        token_num = transformed_to_translator_hs.shape[1]

        # Input dummy text but the it is ignored and uses the injected 
        translator_outputs = self.translator.get_output_by_using_dummy(token_num=token_num)

        return translator_outputs

    def create_trainer(
            self, train_dataset: ComboModelDataset, eval_dataset: ComboModelDataset,
            output_dir: str, logging_dir: str, epochs: int = 5,
            batch_size: int = 1, weight_decay: float = 0.01,
            logging_steps: int = 1000, evaluation_strategy: str = "steps",
            lr=0.006334926670051613, max_grad_norm: float = 1.0,
            optimizer=None, scheduler=None, device='cpu'
    ) -> CombinedTrainer:

        # Move datasets to 'coda' device
        train_dataset = train_dataset
        eval_dataset = eval_dataset
        
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
            log_level="info",
            max_grad_norm=max_grad_norm,
            fp16=True,  # Enable mixed precision training
        )

        trainer = CombinedTrainer(
            model=self.to(self.device),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            total_steps=total_steps,
            device=device
        )

        return trainer

    def train_model(
            self, train_dataset: ComboModelDataset, eval_dataset: ComboModelDataset,
            output_dir: str, logging_dir: str, epochs: int = 5,
            batch_size: int = 1, weight_decay: float = 0.01,
            logging_steps: int = 1000, evaluation_strategy: str = "steps",
            lr=0.006334926670051613, max_grad_norm: float = 1.0,
            optimizer=None, scheduler=None, device='cpu'
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
            eval_strategy=evaluation_strategy,
            lr=lr,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )

        trainer.train()

        return trainer

    def save_model(self, trainer, output_dir):
        # Save the finetuned model and tokenizer with a new name
        pretrained_model_dir = f"./pretrained_models/end_to_end_model/{output_dir}"
        trainer.save_model(pretrained_model_dir)

    # Overrides BestHyper func
    def train_and_evaluate(self, train_dataset, eval_dataset, lr, weight_decay, batch_size, epochs, output_dir,
                           logging_dir):
        """Subclasses should implement this method to define how to train and evaluate the model using transformers.Trainer."""
        trainer = self.create_trainer(train_dataset=train_dataset,
                                      eval_dataset=eval_dataset,
                                      output_dir=output_dir,
                                      logging_dir=logging_dir,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      lr=lr,
                                      weight_decay=weight_decay,
                                      device=self.device)
        # Train the model
        trainer.train()

        # Results
        eval_results = trainer.evaluate()

        return eval_results['eval_loss']

    def printTrainableParams(self):
        # Print the parameter names for the model customLLM
        for name, param in self.parameters():
            if param.requires_grad:
                print(name)
