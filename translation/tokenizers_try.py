from transformers import MarianTokenizer

# Load the tokenizers for the two models
he_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-he-en")
en_he_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-he")

# Step 1: Define the Hebrew sentence
hebrew_sentence = "המשפט הזה כתוב בעברית"

# Step 2: Tokenize the Hebrew sentence using the he-en tokenizer
he_en_token_ids = he_en_tokenizer.encode(hebrew_sentence, add_special_tokens=True)

# Step 3: Extract the last token ID from the he-en tokenizer
last_token_id = he_en_token_ids[-2]  # Last token before the special token, if exists

# Step 4: Decode the last token back to text using the he-en tokenizer
last_token = he_en_tokenizer.decode([last_token_id], skip_special_tokens=True)

# Step 5: Tokenize this token using the en-he tokenizer
with en_he_tokenizer.as_target_tokenizer():
    en_he_token_id = en_he_tokenizer.encode(last_token, add_special_tokens=False)[0]

# Output the results
print(f"Last token ID in he-en tokenizer: {last_token_id}")
print(f"Corresponding token ID in en-he tokenizer: {en_he_token_id}")

print(en_he_tokenizer.decode([en_he_token_id], skip_special_tokens=True))
