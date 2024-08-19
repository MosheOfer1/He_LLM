from transformers import AutoTokenizer, OPTForCausalLM
import torch
import math

from my_datasets.create_datasets import load_sentences_from_csv

"""
results:
English wikitext-2-raw-v1 len(dataset[i]['text']) > 50 1962 sentences
Perplexity for OPT-125M: 51.94367612955789
Perplexity for OPT-350M: 40.732918062992304



"""


# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, texts, max_length=512):
    model.eval()
    total_loss = 0.0
    total_length = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_length += input_ids.size(1)

    perplexity = math.exp(total_loss / total_length)
    return perplexity


# Load the WikiText dataset
texts = load_sentences_from_csv("../my_datasets/hebrew_sentences.csv", "sentence")  # load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# texts = [dataset[i]['text'] for i in range(len(dataset)) if len(dataset[i]['text']) > 50]

# Initialize tokenizer and model for OPT-125M
tokenizer_125m = AutoTokenizer.from_pretrained("facebook/opt-125m")
model_125m = OPTForCausalLM.from_pretrained("facebook/opt-125m")

# Calculate perplexity for OPT-125M
perplexity_125m = calculate_perplexity(model_125m, tokenizer_125m, texts)
print(f"Perplexity for OPT-125M: {perplexity_125m}")

# Initialize tokenizer and model for OPT-350M
tokenizer_350m = AutoTokenizer.from_pretrained("facebook/opt-350m")
model_350m = OPTForCausalLM.from_pretrained("facebook/opt-350m")

# Calculate perplexity for OPT-350M
perplexity_350m = calculate_perplexity(model_350m, tokenizer_350m, texts)
print(f"Perplexity for OPT-350M: {perplexity_350m}")
