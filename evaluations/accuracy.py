import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_datasets.create_datasets import read_file_lines


def clean_token(token):
    """ Remove special characters like Ġ and convert to more readable form. """
    return token.replace('Ġ', '_').replace('▁', '_')


def display_token_comparison(true_tokens, predicted_tokens, attention_mask, correct_mask):
    print(f"{'Index':<5} {'True Token':<20} {'Predicted Token':<20} {'Correct?':<8}")
    print("=" * 60)
    for idx, (true_token, pred_token, attn, corr) in enumerate(zip(true_tokens, predicted_tokens, attention_mask, correct_mask)):
        if attn:  # Only display non-padding tokens
            correct_label = '✔' if corr else '✘'
            print(f"{idx:<5} {clean_token(true_token):<20} {clean_token(pred_token):<20} {correct_label:<8}")
    print("\n")


def calculate_accuracy(model, tokenizer, texts, batch_size=8, max_length=512):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', max_length=max_length, truncation=True, padding=True)

            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift the input_ids and attention mask to align with the next token for prediction comparison
            shift_labels = input_ids[:, 1:].contiguous()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_attention_mask = attention_mask[:, 1:].contiguous()  # Shift attention mask to ignore padding for next token

            # Get the predicted tokens by taking the argmax of the logits
            predicted_tokens = torch.argmax(shift_logits, dim=-1)

            # Compare the predicted tokens with the true next tokens, ignoring padding (where attention_mask is 0)
            correct_mask = (predicted_tokens == shift_labels) * shift_attention_mask

            # Convert input_ids and predicted tokens to words
            true_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in shift_labels]
            predicted_tokens_text = [tokenizer.convert_ids_to_tokens(ids) for ids in predicted_tokens]

            # Display tokens with correct/incorrect labels
            for j, (true_seq, pred_seq, att_mask, corr_mask) in enumerate(zip(true_tokens, predicted_tokens_text, shift_attention_mask, correct_mask)):
                print(f"\nSample {j+1} in batch {i // batch_size + 1}:")
                display_token_comparison(true_seq, pred_seq, att_mask.cpu().tolist(), corr_mask.cpu().tolist())

            correct_predictions += correct_mask.sum().item()
            total_predictions += shift_attention_mask.sum().item()

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Total Correct Predictions: {correct_predictions}, Total Predictions (non-padding): {total_predictions}")
    print(f"Calculated Accuracy: {accuracy * 100:.2f}%")

    return accuracy


# Ensure GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead.")

# Initialize tokenizer and model
# model_path = "DAMO-NLP-MT/polylm-1.7b"
model_path = input("Enter model name to evaluate:")
# model_path = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, use_fast=False)

# Set pad_token to eos_token or add a new pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token


model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True,
                                             offload_folder="offload_folder").to(device)

model.eval()

texts = read_file_lines('../my_datasets/He-En_1000_each.txt')
he_text = texts[:1000]
en_text = texts[1000:]
print("he text len: ", len(he_text))
print("en text len: ", len(en_text))

# Calculate accuracy for polylm-1.7b in English
print("Calculating accuracy for English...")
accuracy_en = calculate_accuracy(model, tokenizer, en_text)
print(f"Accuracy for {model_path} in English: {accuracy_en * 100:.2f}%")

# Calculate accuracy for polylm-1.7b in Hebrew
print("Calculating accuracy for Hebrew...")
accuracy_he = calculate_accuracy(model, tokenizer, he_text)
print(f"Accuracy for {model_path} in Hebrew: {accuracy_he * 100:.2f}%")
