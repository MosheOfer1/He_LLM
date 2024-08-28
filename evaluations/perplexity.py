import os
import sys
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_datasets.create_datasets import read_file_lines

def calculate_perplexity(model, tokenizer, texts, batch_size=8, max_length=512):
    model.eval()
    total_loss = 0.0
    total_length = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', max_length=max_length, truncation=True, padding=True)

            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)

            print(f"Batch {i // batch_size + 1}:")
            print(f"Input Texts: {batch_texts}")
            print(f"Input IDs: {input_ids}")
            print(f"Attention Mask: {attention_mask}")

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            print(f"Loss for this batch: {loss.item()}")

            total_loss += loss.item() * input_ids.size(1)
            total_length += input_ids.size(1)

    perplexity = math.exp(total_loss / total_length)
    print(f"Total Loss: {total_loss}, Total Length: {total_length}")
    print(f"Calculated Perplexity: {perplexity}")

    return perplexity


# Ensure GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead.")

# Initialize tokenizer and model
model_path = "DAMO-NLP-MT/polylm-1.7b"

tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, use_fast=False)

# Set pad_token to eos_token or add a new pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, offload_folder="offload_folder").to(device)
model.eval()

texts = read_file_lines('my_datasets/He-En_1000_each.txt')
he_text = texts[:1000]
en_text = texts[1000:]
print("he text len: ", len(he_text))
print("en text len: ", len(en_text))

# Calculate perplexity for polylm-1.7b in English
print("Calculating perplexity for English...")
perplexity_en = calculate_perplexity(model, tokenizer, en_text)
print(f"Perplexity for polylm-1.7b in English: {perplexity_en}")

# Calculate perplexity for polylm-1.7b in Hebrew
print("Calculating perplexity for Hebrew...")
perplexity_he = calculate_perplexity(model, tokenizer, he_text)
print(f"Perplexity for polylm-1.7b Hebrew: {perplexity_he}")
