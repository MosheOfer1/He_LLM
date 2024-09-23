import re
from matplotlib import pyplot as plt
from transformers import MarianTokenizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define a helper function to identify Hebrew tokens
from custom_datasets.create_datasets import read_file_to_string


def is_hebrew_token(token):
    return bool(re.search('[\u0590-\u05FF]', token))


# Define the comparison function
def compare_dictionaries(dict1, dict2):
    keys_in_first_not_in_second = set(dict1.keys()) - set(dict2.keys())
    keys_in_second_not_in_first = set(dict2.keys()) - set(dict1.keys())

    count_in_first_not_in_second = len(keys_in_first_not_in_second)
    count_in_second_not_in_first = len(keys_in_second_not_in_first)

    return {
        "in_first_not_in_second": count_in_first_not_in_second,
        "in_second_not_in_first": count_in_second_not_in_first
    }


# Function to plot the comparison results as pie charts
def plot_pie_charts(full_result, hebrew_result, total_full_1, total_full_2, total_hebrew_1, total_hebrew_2):
    # Full vocabulary pie charts
    full_labels_1 = ['Unique to First', 'Shared']
    full_sizes_1 = [full_result['in_first_not_in_second'], total_full_1 - full_result['in_first_not_in_second']]
    full_labels_2 = ['Unique to Second', 'Shared']
    full_sizes_2 = [full_result['in_second_not_in_first'], total_full_2 - full_result['in_second_not_in_first']]

    # Hebrew vocabulary pie charts
    hebrew_labels_1 = ['Unique to First', 'Shared']
    hebrew_sizes_1 = [hebrew_result['in_first_not_in_second'], total_hebrew_1 - hebrew_result['in_first_not_in_second']]
    hebrew_labels_2 = ['Unique to Second', 'Shared']
    hebrew_sizes_2 = [hebrew_result['in_second_not_in_first'], total_hebrew_2 - hebrew_result['in_second_not_in_first']]

    # Plot the pie charts
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.pie(full_sizes_1, labels=full_labels_1, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
    plt.title('Full Vocabulary - First Tokenizer')

    plt.subplot(2, 2, 2)
    plt.pie(full_sizes_2, labels=full_labels_2, autopct='%1.1f%%', colors=['#99ff99', '#ffcc99'])
    plt.title('Full Vocabulary - Second Tokenizer')

    plt.subplot(2, 2, 3)
    plt.pie(hebrew_sizes_1, labels=hebrew_labels_1, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
    plt.title('Hebrew Vocabulary - First Tokenizer')

    plt.subplot(2, 2, 4)
    plt.pie(hebrew_sizes_2, labels=hebrew_labels_2, autopct='%1.1f%%', colors=['#99ff99', '#ffcc99'])
    plt.title('Hebrew Vocabulary - Second Tokenizer')

    plt.tight_layout()
    plt.show()


def compare_tokenization(text):
    # Tokenize the text using both tokenizers
    tokens1 = tokenizer1.tokenize(text)
    with tokenizer2.as_target_tokenizer():
        tokens2 = tokenizer2.tokenize(text)

    # Print the differences in tokenization
    print("Differences in tokenization:")
    print("=" * 40)
    print(f"Original Text: {text}\n")
    print(f"Tokenizer 1 Tokens: {tokens1}")
    print(f"Tokenizer 2 Tokens: {tokens2}\n")

    print("Comparison of tokenized segments:")
    print("=" * 40)

    aligned_pairs, differences_count, more_tokens_first, more_tokens_second = align_tokens(tokens1, tokens2)

    for pair in aligned_pairs:
        # Mark differences with asterisks
        if pair[0] != pair[1]:
            print(f"**{pair}**")
        else:
            print(pair)

    # Print summary of differences
    print("\nSummary of differences:")
    print(f"Total number of differences: {differences_count}")
    total_pairs = len(aligned_pairs)
    if total_pairs > 0:
        difference_percentage = (differences_count / total_pairs) * 100
        more_first_percentage = (more_tokens_first / total_pairs) * 100
        more_second_percentage = (more_tokens_second / total_pairs) * 100
    else:
        difference_percentage = 0
        more_first_percentage = 0
        more_second_percentage = 0

    print(f"Percentage of differences: {difference_percentage:.2f}%")
    print(f"Percentage where Tokenizer 1 has more tokens: {more_first_percentage:.2f}%")
    print(f"Percentage where Tokenizer 2 has more tokens: {more_second_percentage:.2f}%")
    print(f"Percentage where Tokenizer 1 has same as Tokenizer 2: {(difference_percentage - more_second_percentage - more_first_percentage):.2f}%")


def align_tokens(tokens1, tokens2):
    aligned_pairs = []
    differences_count = 0
    more_tokens_first = 0
    more_tokens_second = 0
    i, j = 0, 0

    while i < len(tokens1) and j < len(tokens2):
        token_group_1 = [tokens1[i]]
        token_group_2 = [tokens2[j]]

        # Collect tokens from the first list until they match the beginning of the next token in the second list
        while not ''.join(token_group_1) == ''.join(token_group_2):
            if len(''.join(token_group_1)) < len(''.join(token_group_2)):
                i += 1
                token_group_1.append(tokens1[i])
            else:
                j += 1
                token_group_2.append(tokens2[j])

        # Add the aligned pair of groups to the result
        aligned_pairs.append((tuple(token_group_1), tuple(token_group_2)))

        # Increment the difference counter if the tokens don't match
        if token_group_1 != token_group_2:
            differences_count += 1

        # Count which tokenizer has more tokens
        if len(token_group_1) > len(token_group_2):
            more_tokens_first += 1
        elif len(token_group_2) > len(token_group_1):
            more_tokens_second += 1

        i += 1
        j += 1

    return aligned_pairs, differences_count, more_tokens_first, more_tokens_second


# Define the model names
translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator2_model_name = "Helsinki-NLP/opus-mt-en-he"

# Load the tokenizers
tokenizer1 = MarianTokenizer.from_pretrained(translator1_model_name)
tokenizer2 = MarianTokenizer.from_pretrained(translator2_model_name)

# Get the source vocabulary of the first tokenizer
source_vocab_1 = tokenizer1.get_vocab()

# Get the target vocabulary of the second tokenizer using as_target_tokenizer
with tokenizer2.as_target_tokenizer():
    target_vocab_2 = tokenizer2.get_vocab()

# Filter the vocabularies to include only Hebrew tokens
hebrew_vocab_1 = {k: v for k, v in source_vocab_1.items() if is_hebrew_token(k)}
hebrew_vocab_2 = {k: v for k, v in target_vocab_2.items() if is_hebrew_token(k)}

# Compare the full vocabularies
full_comparison_result = compare_dictionaries(source_vocab_1, target_vocab_2)

# Compare the Hebrew vocabularies
hebrew_comparison_result = compare_dictionaries(hebrew_vocab_1, hebrew_vocab_2)


# Print the results
print("Comparison Result:")
print(f"Full dictionary comparison:")
print(f"  First tokenizer dictionary size: {len(source_vocab_1)}")
print(f"  Second tokenizer dictionary size: {len(target_vocab_2)}")
print(f"  Keys in the first tokenizer but not in the second: {full_comparison_result['in_first_not_in_second']}")
print(f"  Keys in the second tokenizer but not in the first: {full_comparison_result['in_second_not_in_first']}\n")

print(f"Hebrew dictionary comparison:")
print(f"  Hebrew keys in the first tokenizer: {len(hebrew_vocab_1)}")
print(f"  Hebrew keys in the second tokenizer: {len(hebrew_vocab_2)}")
print(
    f"  Hebrew keys in the first tokenizer but not in the second: {hebrew_comparison_result['in_first_not_in_second']}")
print(
    f"  Hebrew keys in the second tokenizer but not in the first: {hebrew_comparison_result['in_second_not_in_first']}")

text_file_path = "../my_datasets/SVLM_Hebrew_Wikipedia_Corpus.txt"

hebrew_text = read_file_to_string(text_file_path)[10000:30000]

compare_tokenization(hebrew_text)


# Plot the comparison results as pie charts
plot_pie_charts(
    full_comparison_result,
    hebrew_comparison_result,
    len(source_vocab_1),
    len(target_vocab_2),
    len(hebrew_vocab_1),
    len(hebrew_vocab_2)
)
