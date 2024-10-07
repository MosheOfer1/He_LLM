import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, OPTForCausalLM
from torch.utils.data import Dataset, DataLoader


class FactorizedEmbedding(nn.Module):
    def __init__(self, hidden_size, vocab_size, bottleneck_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.out_proj = nn.Linear(bottleneck_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.dense(x)
        return self.out_proj(x)


class CustomLLM(nn.Module):
    def __init__(self, he_en_model, en_he_model, llm_model, vocab_size, bottleneck_size=1024):
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
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()

        self.tokenized_data = tokenizer.encode(self.data)

    def __len__(self):
        return len(self.tokenized_data) - self.max_length

    def __getitem__(self, idx):
        chunk = self.tokenized_data[idx:idx + self.max_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create a terminal progress bar.
    :param iteration: current iteration (int)
    :param total: total iterations (int)
    :param prefix: prefix string (str)
    :param suffix: suffix string (str)
    :param decimals: positive number of decimals in percent complete (int)
    :param length: character length of bar (int)
    :param fill: bar fill character (str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()  # New line on completion


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


def train_llm(model, file_path, tokenizer, num_epochs=5, batch_size=8, learning_rate=5e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model.to(device)

    print_model_info(model)

    dataset = TextDataset(file_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of batches per epoch: {len(dataloader)}")

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'custom_layer' in n or n.startswith('output_projection')],
         'lr': learning_rate},
        {'params': [p for n, p in model.named_parameters() if
                    'custom_layer' not in n and not n.startswith('output_projection')], 'lr': learning_rate / 10}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print("\nStarting training...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (batch_x, batch_y) in enumerate(dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            print_progress_bar(i + 1, len(dataloader), prefix=f'Epoch {epoch + 1}/{num_epochs}',
                               suffix=f'Loss: {loss.item():.4f}', length=30)

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")

        scheduler.step()

    print("Training completed!")


# Setup and usage example:
he_en_model = AutoModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-he-en")
en_he_model = AutoModel.from_pretrained("Helsinki-NLP/opus-mt-en-he")
llm_model = OPTForCausalLM.from_pretrained("facebook/opt-350m")

# Use the tokenizer from the Hebrew-English model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-he-en")

custom_llm = CustomLLM(he_en_model, en_he_model, llm_model, len(tokenizer))

# Assuming 'train_data.txt' is your file with training text
train_llm(custom_llm, '../my_datasets/unique_150k_lines.txt', tokenizer)
