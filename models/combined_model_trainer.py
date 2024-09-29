import torch
from torch import nn
from transformers import Trainer, get_linear_schedule_with_warmup
from torch.optim import Adam

import matplotlib

matplotlib.use('Agg')  # Use 'Agg' for non-GUI environments

import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def compute_metrics_fun(eval_pred):
    logits, labels = eval_pred

    # Reshape logits and labels for CrossEntropyLoss
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)  # Flatten logits to [batch_size * seq_len, vocab_size]
    labels = labels.view(-1)  # Flatten labels to [batch_size * seq_len]

    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(logits, labels)

    # Compute accuracy and perplexity
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).float()
    accuracy = correct.sum() / len(correct)
    perplexity = torch.exp(loss)

    # Return both accuracy and perplexity in a dictionary format
    return {
        "accuracy": accuracy.item(),
        "perplexity": perplexity.item()
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
                 device='cpu'):
        
        print(f"CombinedTrainer.__init__ - uses: {device}")

        self.device = device

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

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overrides the Trainer lib default loss computation
        """
        model = model.to(self.device)
        input_ids = inputs.get("input_ids").to(self.device)
        attention_mask = inputs.get("input_mask").to(self.device)
        outputs = model(input_ids=input_ids, input_attention_mask=attention_mask)

        logits = outputs.get("logits").to(self.device)
        labels = inputs.get("labels").to(self.device)

        # Reshape logits and labels for CrossEntropyLoss
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)  # Flatten logits to [batch_size * seq_len, vocab_size]
        labels = labels.view(-1)  # Flatten labels to [batch_size * seq_len]

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)

        # Compute accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float()
        accuracy = correct.sum() / len(correct)
        loss = loss.to(self.device)
        # Log the accuracy
        self.log({"accuracy": accuracy.item()})

        return (loss, outputs) if return_outputs else loss

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
