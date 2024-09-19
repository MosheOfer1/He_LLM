import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.opt_llm import OptLLM
from custom_datasets.combo_model_dataset_window import ComboModelDataset

from transformers import TrainingArguments
from models.combined_model_trainer import CombinedTrainer

# Transformers
from custom_transformers.transformer import Transformer

# Translators
from translation.helsinki_translator import HelsinkiTranslator

# Optuna best hypers finder
from utilty.BestHyper import BestHyper


class MyCustomModel(nn.Module, BestHyper):
    def __init__(self,
                 src_to_target_translator_model_name,
                 target_to_src_translator_model_name,
                 llm_model_name,
                 llm_model_cls=OptLLM,
                 pretrained_transformer1_path: str = None,
                 pretrained_transformer2_path: str = None,
                 device='cpu'):

        print(f"MyCustomModel.__init__ - uses: {device}")

        nn.Module.__init__(self)

        self.device = device

        # Custom Translator
        self.translator = HelsinkiTranslator(src_to_target_translator_model_name,
                                             target_to_src_translator_model_name,
                                             device=device)
        # Custom LLM
        self.llm = llm_model_cls(
            llm_model_name,
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

    def forward(self, input_ids, attention_mask=None, labels=None, keep_reshaped=False) -> torch.Tensor:
        
        batch_size = input_ids.shape[0]
        token_num = input_ids.shape[2]
        
        # Step 1: Get hidden states from the translator for input_ids
        translator_last_hs = self.get_translator_hidden_states(input_ids, attention_mask)
        
        # print(f"translator_last_hs.shape = {translator_last_hs.shape}")

        # Step 2: Transform to LLM hidden states
        transformed_to_llm_hs = self.transformer.transformer1.forward(translator_last_hs)

        # print(f"transformed_to_llm_hs.shape = {transformed_to_llm_hs.shape}")
                
        # Step 3: Get LLM output using dummy input        
        llm_last_hidden_state = self.get_llm_hidden_states(transformed_to_llm_hs) # shape: [batch * tokens, 1, dim]
        
        # print(f"llm_last_hidden_state.shape = {llm_last_hidden_state.shape}")

        # Step 4: Transform LLM hidden states to translator's hidden states and inject
        transformed_to_translator_hs = self.get_reshaped_translator2_hidden_states(llm_last_hidden_state, reshape=True) # [batch * tokens, 2, trans_dim]
        
        # print(f"final - reshaped_transformed_to_translator_hs.shape = {transformed_to_translator_hs.shape}")
        
        # Step 5: Get translator output using dummy input
        outputs = self.get_translator_outputs(transformed_to_translator_hs)
        
        if not keep_reshaped:
            self.reverse_reshaping_translator2_outputs(outputs, batch_size, token_num)
        
        return outputs

    def reverse_reshaping_translator2_outputs(outputs, batch_size, token_num):
        outputs.logits
    
    
    def get_translator_hidden_states(self, input_ids, attention_mask):
        return self.translator.input_ids_to_hidden_states(
            input_ids,
            -1,
            self.translator.src_to_target_tokenizer,
            self.translator.src_to_target_model,
            False,
            attention_mask=attention_mask
        ).to(self.device)

    def get_llm_last_hidden_state(self, transformed_to_llm_hs):
        llm_hs = self.get_llm_hidden_states(transformed_to_llm_hs)
        
        # print(f"llm_hs.shape: {llm_hs.shape}")
        
        return llm_hs[:, -1, :].unsqueeze(1)  # Shape: [batch_size, 1, dim]

    def get_llm_hidden_states(self, transformed_to_llm_hs):
        self.llm.inject_hidden_states(transformed_to_llm_hs)

        batch_size = transformed_to_llm_hs.shape[0]
        token_num = transformed_to_llm_hs.shape[1]

        llm_outputs = self.llm.get_output_by_using_dummy(token_num=token_num, batch_size=batch_size)
        
        return llm_outputs.hidden_states[-1] # shape: [batch_size, token_num, dim]
    
    def get_translator_outputs(self, transformed_to_translator_hs):
        self.translator.inject_hidden_states(transformed_to_translator_hs)
        outputs = self.translator.get_output_by_using_dummy(
            token_num=transformed_to_translator_hs.shape[1],
            batch_size=transformed_to_translator_hs.shape[0]
        )
        return outputs

    def get_reshaped_translator2_hidden_states(self, llm_last_hidden_state, reshape = False):
        """
        Transforms the LLM's last hidden state to the translator's hidden state and
        concatenates it with the EOS token embedding.

        Args:
        - llm_last_hidden_state: Tensor of the last hidden state from the LLM. Shape: [batch_size, 1, dim].

        Returns:
        - transformed_to_translator_hs: Tensor after transforming and concatenating the EOS embedding.
        """
        
        # Transform to translator's first hidden states
        transformed_to_translator_hs = self.get_transformer2_output(llm_last_hidden_state=llm_last_hidden_state,
                                                                    reshape=reshape)
        batch_size = transformed_to_translator_hs.shape[0]
         
        # Get hidden states of the EOS token
        with torch.no_grad():
            # Use the context manager without specifying the layer number or state
            with self.translator.injection_state():
                eos_embedding = self.translator.text_to_hidden_states(
                    "a",
                    0,
                    self.translator.target_to_src_tokenizer,
                    self.translator.target_to_src_model,
                    True
                )  # Shape: [1, 2, dim]

            # Reshape eos_embedding and repeat for the batch size
            eos_embedding = eos_embedding[:, -1, :].unsqueeze(0)  # Shape: [1, 2, dim] -> [1, dim]
            eos_embedding = eos_embedding.repeat(batch_size, 1, 1)  # Shape: [batch_size, 1, dim]

        # Concatenate llm_last_hidden_state with eos_embedding along the token dimension
        transformed_to_translator_hs = torch.cat((transformed_to_translator_hs, eos_embedding),
                                                 dim=1)  # Shape: [batch_size, 2, dim]

        return transformed_to_translator_hs
    
    def get_transformer2_output(self, llm_last_hidden_state, reshape = False):
        
        # [batch, tokens, dim]
        transformed_to_translator_hs = self.transformer.transformer2.forward(llm_last_hidden_state).to(self.device)
        
        # Reshaped: [batch * tokens, 1, dim]
        if (reshape):
            transformed_to_translator_hs = transformed_to_translator_hs.reshape(-1, 1, transformed_to_translator_hs.shape[-1])
        
        return transformed_to_translator_hs
    
    def create_trainer(
            self, train_dataset: ComboModelDataset, eval_dataset: ComboModelDataset,
            output_dir: str, logging_dir: str, epochs: int,
            batch_size: int, weight_decay: float,
            logging_steps: int, evaluation_strategy: str,
            lr, max_grad_norm: float,
            optimizer, scheduler, device, save_strategy,
            save_steps, save_total_limit
    ) -> CombinedTrainer:

        epoch = len(train_dataset)
        total_steps = int(epoch // batch_size * epochs)
        eval_steps = int(total_steps // epochs)
        warmup_steps = int(0.1 * total_steps)
        warmup_steps = warmup_steps if warmup_steps > 1 else 0

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
            eval_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            learning_rate=lr,
            log_level="info",
            max_grad_norm=max_grad_norm,
            fp16=True,  # Enable mixed precision training
            # Saving-related parameters:
            save_strategy=save_strategy,  # Save every X steps
            save_steps=save_steps,  # Save a checkpoint every 500 steps
            save_total_limit=save_total_limit  # Keep only the last 3 checkpoints
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
            optimizer=None, scheduler=None, device='cpu', save_strategy="no",
            save_steps=0, save_total_limit=0
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
            scheduler=scheduler,
            device=device,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit
        )

        self.printTrainableParams()
        trainer.train()

        return trainer

    def save_transformers_state_dict(self, path: str):

        # Get the full state dict
        full_state_dict = self.state_dict()

        # Filter out only transformer1 and transformer2 keys
        transformer1_and_2_state_dict = {k: v for k, v in full_state_dict.items() if
                                         k.startswith('transformer.transformer1') or k.startswith(
                                             'transformer.transformer2')}

        # Save the filtered state dictionary
        torch.save(transformer1_and_2_state_dict, path)

        print(f"Saved state dictionary with {len(transformer1_and_2_state_dict)} keys.")

    def load_transformers_state_dict(self, path: str, to_transformer1=True, to_transformer2=True):
        # Load the state dictionary from the file
        loaded_state_dict = torch.load(path, weights_only=True)

        # Get the current state dictionary of the model
        current_state_dict = self.state_dict()

        # Create a filtered dictionary depending on the flags
        filtered_state_dict = {}

        if to_transformer1:
            # Add keys that belong to transformer1
            filtered_state_dict.update(
                {k: v for k, v in loaded_state_dict.items() if k.startswith('transformer.transformer1')})

        if to_transformer2:
            # Add keys that belong to transformer2
            filtered_state_dict.update(
                {k: v for k, v in loaded_state_dict.items() if k.startswith('transformer.transformer2')})

        # Update the model's current state dictionary with the filtered weights
        current_state_dict.update(filtered_state_dict)

        # Load the updated state dict back into the model
        self.load_state_dict(current_state_dict)

        print(
            f"Loaded {'transformer1 ' if to_transformer1 else ''}{'and ' if to_transformer1 and to_transformer2 else ''}{'transformer2' if to_transformer2 else ''} state into the model.")

    def compere_state_dicts(self, model2, only_transformer1=False, only_transformer2=False):

        excluded = []

        if only_transformer1:
            excluded.append('transformer.transformer2')

        elif only_transformer2:
            excluded.append('transformer.transformer1')

        # Get the state dictionaries of both models
        state_dict1 = self.state_dict()
        state_dict2 = model2.state_dict()

        for key in state_dict1:
            if len(excluded) == 0 or not key.startswith(excluded[0]):
                if key in state_dict2:
                    # Compare the tensors
                    if not torch.equal(state_dict1[key], state_dict2[key]):
                        return False
                else:
                    return False
        return True

    # Overrides BestHyper func
    def train_and_evaluate(self, train_dataset, eval_dataset, lr, weight_decay, batch_size, epochs, output_dir,
                           logging_dir):
        """Subclasses should implement this method to define how to train and evaluate the model using
        transformers.Trainer. """
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
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)
