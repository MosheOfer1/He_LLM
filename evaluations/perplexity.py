from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math

from my_datasets.create_datasets import read_file_lines

"""
results:
English wikitext-2-raw-v1 len(dataset[i]['text']) > 50 1962 sentences
Perplexity for OPT-125M: 51.94367612955789
Perplexity for OPT-350M: 40.732918062992304

Hebrew SVLM_Hebrew_Wikipedia_Corpus.txt 1000 sentences
Perplexity for OPT-125M: 9.923426167073405
Perplexity for OPT-350M: 8.272500443599894

"""


def calculate_perplexity(model, tokenizer, texts, batch_size=8, max_length=512):
    model.eval()
    total_loss = 0.0
    total_length = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', max_length=max_length, truncation=True, padding=True)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_length += input_ids.size(1)
            print(f"The loss in {i} is {loss}")

    perplexity = math.exp(total_loss / total_length)
    return perplexity


# Load the WikiText dataset
# texts = read_file_lines('../my_datasets/SVLM_Hebrew_Wikipedia_Corpus.txt')[:1000]
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = [dataset[i]['text'] for i in range(len(dataset)) if len(dataset[i]['text']) > 50][:1000]
print(len(texts))
# Initialize tokenizer and model
model_path = "DAMO-NLP-MT/polylm-1.7b"

tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, offload_folder="../offload_folder",)
model.eval()
# Calculate perplexity for OPT-125M
perplexity_125m = calculate_perplexity(model, tokenizer, texts)
print(f"Perplexity for polylm-1.7b: {perplexity_125m}")

texts = read_file_lines('../my_datasets/SVLM_Hebrew_Wikipedia_Corpus.txt')[:1000]
print(len(texts))

# Calculate perplexity for OPT-125M
perplexity_125m = calculate_perplexity(model, tokenizer, texts)
print(f"Perplexity for polylm-1.7b Hebrew: {perplexity_125m}")

