from torch.utils.data import Dataset


class ComboModelDataset(Dataset):
    def __init__(self, text: str, input_tokenizer, output_tokenizer, window_size=5):
        self.token_pairs = align_tokens(input_tokenizer, output_tokenizer, text)
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.window_size = window_size

    def __len__(self):
        return len(self.token_pairs) - self.window_size - 1

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary with 'input_ids' containing the tokenized Hebrew sentence and 
                  'labels' containing the associated label.
        """
        input_ids = self._get_input_ids(idx)
        next_token = self.token_pairs[idx + self.window_size][1][0]

        label = self.output_tokenizer(
            text_target=next_token,
            add_special_tokens=False
        )["input_ids"][0]

        return {
            'input_ids': input_ids,
            'labels': label,
        }

    def _get_input_ids(self, idx):
        input_ids = []

        # Limit the range of token pairs based on idx and window_size
        for i in range(idx, min(idx + self.window_size, len(self.token_pairs))):
            pair = self.token_pairs[i]
            first_tuple_tokens = pair[0]

            tokens = []
            for token in first_tuple_tokens:
                tokens.append(token)

            input_ids.extend(self.input_tokenizer.encode(tokens, add_special_tokens=False))

        return input_ids[(len(input_ids) - self.window_size):len(input_ids)]


def align_tokens(tokenizer1, tokenizer2, text):
    # Tokenize the text using both tokenizers
    tokens1 = tokenizer1.tokenize(text)
    with tokenizer2.as_target_tokenizer():
        tokens2 = tokenizer2.tokenize(text)

    aligned_pairs = []
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

        i += 1
        j += 1

    return aligned_pairs

