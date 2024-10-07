import argparse
import math
from datetime import datetime
import logging
import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, OPTForCausalLM
from torch.utils.data import Dataset, DataLoader


def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")

    # Create a logger
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(format)
    f_handler.setFormatter(format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


class FactorizedEmbedding(nn.Module):
    def __init__(self, hidden_size, vocab_size, bottleneck_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.out_proj = nn.Linear(bottleneck_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.dense(x)
        return self.out_proj(x)


class CustomLLM(nn.Module):
    def __init__(self, he_en_model, en_he_model, llm_model, vocab_size, bottleneck_size):
        super().__init__()

        # Hebrew-English encoder components
        self.he_en_embeddings = he_en_model.shared
        self.he_en_encoder = he_en_model.encoder

        # First custom layer
        self.custom_layer1 = nn.Sequential(
            nn.Linear(he_en_model.config.hidden_size, llm_model.config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(llm_model.config.hidden_size)
        )

        # LLM layers (main body of the model)
        self.main_layers = llm_model.model.decoder.layers

        # Second custom layer
        self.custom_layer2 = nn.Sequential(
            nn.Linear(llm_model.config.hidden_size, en_he_model.config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(en_he_model.config.hidden_size)
        )

        # English-Hebrew decoder layers
        self.en_he_decoder_layers = en_he_model.decoder.layers

        # Factorized output projection
        self.output_projection = FactorizedEmbedding(
            en_he_model.config.hidden_size,
            vocab_size,
            bottleneck_size
        )

        # Freeze most pre-trained layers
        self._freeze_layers()

    def _freeze_layers(self):
        # Freeze all parameters in he_en_encoder
        for param in self.he_en_encoder.parameters():
            param.requires_grad = False

        # Freeze all main LLM layers except the first and last
        for layer in self.main_layers[1:-1]:
            for param in layer.parameters():
                param.requires_grad = False

        # Freeze all EN-HE decoder layers except the first and last
        for layer in self.en_he_decoder_layers[1:-1]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        # Ensure input_ids is of type Long
        input_ids = input_ids.long()

        # Initial encoding
        embeddings = self.he_en_embeddings(input_ids)
        encoder_output = self.he_en_encoder(inputs_embeds=embeddings, attention_mask=attention_mask).last_hidden_state
        # First custom layer
        x = self.custom_layer1(encoder_output)

        # Main LLM processing
        for layer in self.main_layers:
            x = layer(x, attention_mask=attention_mask)[0]

        # Second custom layer
        x = self.custom_layer2(x)

        # EN-HE decoder processing
        for layer in self.en_he_decoder_layers:
            x = layer(x, attention_mask=attention_mask)[0]

        # Final projection
        logits = self.output_projection(x)

        return logits


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, eval_split=0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()

        self.tokenized_data = tokenizer.encode(self.data)

        # Split data into train and eval
        split_point = int(len(self.tokenized_data) * (1 - eval_split))
        self.train_data = self.tokenized_data[:split_point]
        self.eval_data = self.tokenized_data[split_point:]

    def __len__(self):
        return len(self.train_data) - self.max_length

    def __getitem__(self, idx):
        chunk = self.train_data[idx:idx + self.max_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def get_eval_data(self):
        eval_dataset = []
        for i in range(0, len(self.eval_data) - self.max_length, self.max_length):
            chunk = self.eval_data[i:i + self.max_length + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            eval_dataset.append((x, y))
        return eval_dataset


def print_progress_bar(iteration, total, epoch, num_epochs, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create a terminal progress bar with step and epoch information.
    :param iteration: current iteration (int)
    :param total: total iterations (int)
    :param epoch: current epoch (int)
    :param num_epochs: total number of epochs (int)
    :param prefix: prefix string (str)
    :param suffix: suffix string (str)
    :param decimals: positive number of decimals in percent complete (int)
    :param length: character length of bar (int)
    :param fill: bar fill character (str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    step_info = f"Step {iteration}/{total}"
    epoch_info = f"Epoch {epoch}/{num_epochs}"
    print(f'\r{prefix} |{bar}| {percent}% {step_info} {epoch_info} {suffix}', end='')
    # Print New Line on Complete
    if iteration == total:
        print()


def print_model_info(model):
    print("\nModel Architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%")

    print("\nLayer-wise parameter count:")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        trainable_param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}: Total params = {param_count:,}, Trainable params = {trainable_param_count:,}")


def evaluate_batch(logits, targets):
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    # Accuracy
    pred = logits.argmax(dim=-1)
    correct = (pred == targets).float().sum()
    total = targets.numel()
    accuracy = correct / total

    # Perplexity
    perplexity = math.exp(loss.item())

    return loss.item(), accuracy.item(), perplexity


def evaluate_full(model, dataloader, device, dataset_name):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1), reduction='sum')
            total_loss += loss.item()

            # Accuracy
            pred = logits.argmax(dim=-1)
            total_correct += (pred == batch_y).sum().item()
            total_tokens += batch_y.numel()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        "dataset": dataset_name,
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": perplexity
    }


def train_llm(model, dataset, num_epochs=5, batch_size=8, learning_rate=5e-5, device='cuda',
              log_dir='logs', save_dir='model_checkpoints'):
    logger = setup_logger(log_dir)
    logger.info(f"Using device: {device}")
    model.to(device)

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger.info("Model Architecture:")
    logger.info(model)
    print(model)
    print_model_info(model)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = dataset.get_eval_data()
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Training dataset size: {len(dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    logger.info(f"Number of training batches per epoch: {len(train_dataloader)}")

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'custom_layer' in n or n.startswith('output_projection')],
         'lr': learning_rate},
        {'params': [p for n, p in model.named_parameters() if
                    'custom_layer' not in n and not n.startswith('output_projection')], 'lr': learning_rate / 10}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    logger.info("Starting training...")
    model.train()
    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (batch_x, batch_y) in enumerate(train_dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            # Evaluate on current batch every 10 steps
            if global_step % 10 == 0:
                batch_loss, batch_accuracy, batch_perplexity = evaluate_batch(logits, batch_y)
                logger.info(f"Step {global_step}, Batch metrics:")
                logger.info(f"  Loss: {batch_loss:.4f}")
                logger.info(f"  Accuracy: {batch_accuracy:.4f}")
                logger.info(f"  Perplexity: {batch_perplexity:.4f}")

            # Update progress bar
            print_progress_bar(i + 1, len(train_dataloader), epoch + 1, num_epochs,
                               prefix='Training:', suffix=f'Loss: {loss.item():.4f}', length=30)

            # Evaluate on full datasets every half epoch
            if (i + 1) % (len(train_dataloader) // 2) == 0:
                model.eval()
                train_metrics = evaluate_full(model, train_dataloader, device, "Training")
                eval_metrics = evaluate_full(model, eval_dataloader, device, "Evaluation")

                logger.info(f"Full dataset metrics at epoch {epoch + 1}, step {i + 1}:")
                for metrics in [train_metrics, eval_metrics]:
                    logger.info(f"  {metrics['dataset']} dataset:")
                    logger.info(f"    Loss: {metrics['loss']:.4f}")
                    logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
                    logger.info(f"    Perplexity: {metrics['perplexity']:.4f}")

                model.train()  # Set the model back to training mode

        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.2e}")

        # Save model after each epoch
        checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

        # Save best model based on evaluation loss
        if eval_metrics['loss'] < best_eval_loss:
            best_eval_loss = eval_metrics['loss']
            best_model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_eval_loss,
            }, best_model_path)
            logger.info(f"Best model saved to {best_model_path}")

        scheduler.step()

    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train a custom LLM model")
    parser.add_argument("--he-en-model", type=str, default="Helsinki-NLP/opus-mt-tc-big-he-en",
                        help="Name or path of the Hebrew-English model")
    parser.add_argument("--en-he-model", type=str, default="Helsinki-NLP/opus-mt-en-he",
                        help="Name or path of the English-Hebrew model")
    parser.add_argument("--llm-model", type=str, default="facebook/opt-350m",
                        help="Name or path of the LLM model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Path to the data file")
    parser.add_argument("--bottleneck-size", type=int, default=256,
                        help="Bottleneck size for the factorized embedding")
    parser.add_argument("--num-epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory to save log files")
    parser.add_argument("--save-dir", type=str, default="model_checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--eval-split", type=float, default=0.1,
                        help="Proportion of data to use for evaluation")

    args = parser.parse_args()

    # Load models
    he_en_model = AutoModel.from_pretrained(args.he_en_model)
    en_he_model = AutoModel.from_pretrained(args.en_he_model)
    llm_model = OPTForCausalLM.from_pretrained(args.llm_model)

    # Use the tokenizer from the Hebrew-English model
    tokenizer = AutoTokenizer.from_pretrained(args.he_en_model)

    # Create dataset
    dataset = TextDataset(args.data_file, tokenizer, eval_split=args.eval_split)

    # Create custom LLM
    custom_llm = CustomLLM(he_en_model, en_he_model, llm_model, len(tokenizer), args.bottleneck_size)

    # Train the model
    train_llm(custom_llm, dataset,
              num_epochs=args.num_epochs,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate,
              device=args.device,
              log_dir=args.log_dir,
              save_dir=args.save_dir)


if __name__ == "__main__":
    main()
