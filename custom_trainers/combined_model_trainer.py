from torch import nn
import torch
from transformers import Trainer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class CombinedTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        print(f"inputs.keys() = {inputs.keys()}")
        
        # Extract labels
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        class_weights = inputs.get("class_weights")
        
        print(f"\n\n labels.shape = {labels.shape}, labels = {labels}\n\n")
        
        # Feed inputs to model and extract logits
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        
        logits = outputs.get("logits")
        
        print(f"\n\n logits.shape = {logits.shape}")
        print(f"\n\n class_weights = {class_weights}")
        
        learning_index = min(logits.shape[1], labels.shape[1])
        
        # (f"inputs.input_ids.shape = {inputs.input_ids.shape}")
        
        # Compute loss
        loss_func = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"logits.shape = {logits.shape}")
        loss = loss_func(logits[0, :learning_index, :], labels.squeeze(0)[:learning_index])
        
        return (loss, outputs) if return_outputs else loss
