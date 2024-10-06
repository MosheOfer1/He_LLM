import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict
import torch
from torch import nn
from transformers import Trainer, get_linear_schedule_with_warmup
from torch.optim import Adam
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use 'Agg' for non-GUI environments
import os
import sys
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def compute_metrics_fun(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics focusing on the last token prediction.

    Args:
        eval_pred: EvalPrediction object containing:
            - predictions: tuple of arrays, where the first element contains logits
            - label_ids: numpy array of shape (batch_size, seq_len)

    Returns:
        dict containing metrics:
            - accuracy: accuracy score for last token
            - precision: precision score for last token
            - recall: recall score for last token
            - f1: f1 score for last token
            - perplexity: perplexity score for last token
    """
    predictions, labels = eval_pred

    # Unpack predictions tuple and get the logits
    if isinstance(predictions, tuple):
        logits = predictions[0]  # Get the first element which should be the logits
    else:
        logits = predictions

    # Remove the middle dimension (seq_len=1)
    if logits.ndim == 3 and logits.shape[1] == 1:
        logits = logits.squeeze(1)  # Shape: [batch_size, vocab_size]

    # Get the last valid token for each sequence
    batch_size = labels.shape[0]
    last_token_indices = []

    for i in range(batch_size):
        # Find the last non-padding token index
        valid_indices = np.where(labels[i] != 0)[0]
        if len(valid_indices) > 0:
            last_token_indices.append(valid_indices[-1])
        else:
            last_token_indices.append(0)  # Fallback to first position if no valid tokens

    # Get the labels for the last tokens
    last_token_labels = labels[np.arange(batch_size), last_token_indices]

    # Filter out any remaining padding tokens
    valid_mask = last_token_labels != 0
    filtered_logits = logits[valid_mask]
    filtered_labels = last_token_labels[valid_mask]

    if len(filtered_labels) == 0:
        # Return zero metrics if no valid tokens
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "perplexity": float('inf')
        }

    # Convert logits to predictions
    pred_classes = np.argmax(filtered_logits, axis=-1)

    # Calculate metrics
    accuracy = accuracy_score(filtered_labels, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_labels,
        pred_classes,
        average='weighted',
        zero_division=0
    )

    # Calculate perplexity
    # First apply softmax to get probabilities
    exp_preds = np.exp(filtered_logits - np.max(filtered_logits, axis=-1, keepdims=True))
    probs = exp_preds / exp_preds.sum(axis=-1, keepdims=True)

    # Get the probability of the correct class for each sample
    correct_probs = probs[np.arange(len(filtered_labels)), filtered_labels]

    # Calculate cross entropy loss
    eps = 1e-10  # Small constant to prevent log(0)
    log_probs = np.log(correct_probs + eps)
    cross_entropy = -np.mean(log_probs)

    # Calculate perplexity
    perplexity = np.exp(cross_entropy)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "perplexity": float(perplexity)
    }


class CombinedTrainer(Trainer):
    def __init__(self,
                 model,
                 args,
                 train_dataset,
                 eval_dataset,
                 optimizer,
                 scheduler,
                 total_steps,
                 data_collator,
                 compute_metrics_fun,
                 output_tokenizer,
                 device='cpu'):
        
        print(f"CombinedTrainer.__init__ - uses: {device}")

        self.device = device

        self.output_tokenizer = output_tokenizer  # This should be the tokenizer for your output vocabulary

        model = model.to(device)
        
        if not optimizer or not scheduler:
            # Initialize the optimizer and learning rate scheduler
            optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            total_steps = total_steps
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=total_steps)

        Trainer.__init__(self,
                         model=model,
                         args=args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         optimizers=(optimizer, scheduler),
                         data_collator=data_collator,
                         compute_metrics=compute_metrics_fun
                         )

    def compute_loss(self, model, inputs, return_outputs=False, only_last_token=True):
        """
        Overrides the Trainer lib default loss computation
        Returns loss, and optionally the model outputs
        Also computes and logs accuracy and perplexity
        Logs predictions and labels to a file
        """
        model = model.to(self.device)
        input_ids = inputs.get("input_ids").to(self.device)
        attention_mask = inputs.get("input_mask").to(self.device)
        outputs = model(input_ids=input_ids, input_attention_mask=attention_mask)
        logits = outputs.get("logits").to(self.device)  # Shape: [batch_size, 1, vocab_size]
        labels = inputs.get("labels").to(self.device)

        # Initialize lists to store predictions and labels
        predictions_list: List[str] = []
        labels_list: List[str] = []

        if only_last_token:
            batch_size = input_ids.size(0)
            single_token_logits = logits.squeeze(1)  # Shape: [batch_size, vocab_size]

            last_token_indices = (attention_mask.sum(dim=1) - 1).to(torch.long)
            last_token_labels = labels[torch.arange(batch_size), last_token_indices]  # Shape: [batch_size]

            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(single_token_logits, last_token_labels)

            predictions = torch.argmax(single_token_logits, dim=-1)
            correct = (predictions == last_token_labels).float()
            accuracy = correct.sum() / len(correct)

            perplexity = torch.exp(loss)

            # Convert predictions and labels to tokens
            predictions_list = self.output_tokenizer.convert_ids_to_tokens(predictions.tolist())
            labels_list = self.output_tokenizer.convert_ids_to_tokens(last_token_labels.tolist())

        else:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)  # Shape: [batch_size * seq_len, vocab_size]
            labels = labels.view(-1)  # Shape: [batch_size * seq_len]

            valid_mask = (labels != 0).float()  # Assuming 0 is the padding token

            loss_func = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
            loss = loss_func(logits, labels)

            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).float() * valid_mask
            total_valid = valid_mask.sum()
            accuracy = correct.sum() / total_valid if total_valid > 0 else torch.tensor(0.0).to(self.device)

            per_token_loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=0, reduction='none')
            perplexity = torch.exp(per_token_loss[valid_mask.bool()].mean())

            # Convert predictions and labels to tokens (only for non-padding tokens)
            valid_predictions = predictions[valid_mask.bool()]
            valid_labels = labels[valid_mask.bool()]
            predictions_list = self.output_tokenizer.convert_ids_to_tokens(valid_predictions.tolist())
            labels_list = self.output_tokenizer.convert_ids_to_tokens(valid_labels.tolist())

        # Ensure everything is on the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss = loss.to(device)

        # Log the metrics
        self.log({
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "perplexity": perplexity.item()
        })

        # Log predictions and labels to file
        self._log_predictions_and_labels(predictions_list, labels_list)

        return (loss, outputs) if return_outputs else loss

    def _log_predictions_and_labels(self, predictions: List[str], labels: List[str]):
        """
        Log predictions and labels to a file in a human-readable format
        """
        log_dir = "prediction_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "predictions_and_labels.txt")

        with open(log_file, "a") as f:
            f.write(f"Batch: {self.state.global_step}\n")
            f.write("Predictions | Labels\n")
            f.write("-" * 30 + "\n")
            for pred, label in zip(predictions, labels):
                f.write(f"{pred:15} | {label}\n")
            f.write("\n")

    def lr_finder(self, start_lr=1e-7, end_lr=10, num_iter: int = None):
        """
           This method runs a short training loop where the learning rate is gradually increased from `start_lr` to `end_lr` over a specified number of iterations (`num_iter`). The method records the learning rate and the corresponding loss at each step, allowing the user to analyze how the loss changes with different learning rates.

            - **Parameters:**
                - `start_lr`: The initial learning rate at the beginning of the test (default is `1e-7`).
                - `end_lr`: The final learning rate at the end of the test (default is `10`).
                - `num_iter`: The number of iterations to run the learning rate finder (default is `100`).

            - **Procedure:**
                - The method initializes two empty lists, `lrs` and `losses`, to store the learning rates and corresponding losses.
                - It loops over batches from the training data loader, incrementally increasing the learning rate on an exponential scale.
                - At each step, the learning rate is updated in the optimizer, and the model performs a training step to compute the loss.
                - The current learning rate and loss are appended to the respective lists.
                - The loop breaks after the specified number of iterations (`num_iter`), ensuring the process is not too long.
                - Finally, the method returns the lists of learning rates and losses.
        """

        lrs = []
        losses = []
        self.model.train()  # Ensure the model is in training mode

        data = self.get_train_dataloader()
        num_iter = num_iter if num_iter else len(data)

        print_every = int(num_iter // 10)

        for i, batch in enumerate(data):
            if i >= num_iter:
                break

            if i % print_every == 0:
                print(f"Finished {i}/{num_iter}.")

            # Increase learning rate exponentially
            lr = start_lr * (end_lr / start_lr) ** (i / num_iter)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Call the training_step method with the model and inputs
            loss = self.training_step(model=self.model, inputs=batch)

            lrs.append(lr)
            losses.append(loss)

        return lrs, losses

    def lr_finder_with_plot(self, model_name: str = ""):
        """
           This method calls `lr_finder()` to execute the learning rate finder test and then plots the results.

            - **Procedure:**
                - The method calls `lr_finder()` to obtain the learning rates and losses recorded during the test.
                - It then plots the losses against the learning rates using `matplotlib`, with the x-axis on a logarithmic scale (common practice when visualizing learning rates).
                - The plot is labeled appropriately with 'Learning Rate' on the x-axis and 'Loss' on the y-axis, and it is displayed using `plt.show()`.

            - **Purpose:**
                - The resulting plot allows the user to visually identify the optimal learning rate.
                  The ideal learning rate is usually chosen from the steepest downward slope on the plot, 
                  just before the loss starts to increase rapidly.
        """

        lrs, losses = self.lr_finder()
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')

        name = model_name + "_lr_finder_plot.png"
        # plt.show()
        plt.savefig(name)
        print(f"Plot saved as {name}")
