import torch
from torch.utils.data import Dataset


class ComboModelDataset(Dataset):
    def __init__(self, text_list: list[str], input_tokenizer, output_tokenizer, window_size=10, device='cpu'):

        self.device = device

        self.token_pairs = create_token_pairs(input_tokenizer, output_tokenizer, text_list, window_size)
        

        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.window_size = window_size
        self.counter = 0

    def __len__(self):
        length = len(self.token_pairs) - self.window_size - 1
        if length <= 0:
            raise ValueError(
                f"Your ComboModelDataset.__len__ <= 0. Check if your window_size is greater then you data size (clean "
                f"data)")
        return length

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary with 'input_ids' containing the tokenized Hebrew sentence and 
                  'labels' containing the associated label.
        """
        input_ids = self._get_input_ids(idx)
        next_token = self.token_pairs[idx + self.window_size][1][0]

        outputs = self.output_tokenizer(
            text_target=next_token,
            add_special_tokens=False
        )["input_ids"]
        labels = outputs[0]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': labels,
        }

    # TODO - Change it to fit the new dataset - list[sentences]
    def _get_input_ids(self, idx):
        input_ids = []

        # Limit the range of token pairs based on idx and window_size
        for i in range(idx, min(idx + self.window_size, len(self.token_pairs))):
            pair = self.token_pairs[i]
            first_tuple_tokens = pair[0]

            tokens = []
            for token in first_tuple_tokens:
                tokens.append(token)

            encoded = self.input_tokenizer.encode(tokens, add_special_tokens=False)

            # Maybe Bug
            input_ids.extend(encoded)
        input_ids = input_ids[(len(input_ids) - self.window_size):len(input_ids)]
        input_ids.append(self.input_tokenizer.eos_token_id)
        return input_ids


def align_tokens(tokenizer1, tokenizer2, text):
    count = 0
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

        outputs = tokenizer2(
            text_target=token_group_2[0],
            add_special_tokens=False
        )["input_ids"]

        if len(outputs) <= 0:
            aligned_pairs = aligned_pairs[:len(aligned_pairs) - 1]
            count += 1

    # print(f"count: {count}, len {len(aligned_pairs)}")
    return aligned_pairs

def create_token_pairs(input_tokenizer, output_tokenizer, list_text, window_size):
    
    ans = []
    for text in list_text:
        sentence_pairs = align_tokens(input_tokenizer, output_tokenizer, text)
        ans.append(sentence_pairs)
    return delete_unmatched_pairs(ans, window_size)

def delete_unmatched_pairs(token_pairs: list[tuple], window: int):
    
    print(len(token_pairs))
    
    ans =[]
    
    temp = []
    
    for sentence_idx, sentence_pairs in enumerate(token_pairs):
        
        # Append only matched pairs window
        if check_pair(sentence_pairs, sentence_idx):
            temp.append(sentence_pairs)
            
            if len(temp) == window:
                ans.extend(temp)
                temp.clear()
        else:
            temp.clear()

    print(len(ans))
    return ans

def check_pair(sentence_pairs, sentence_idx):
    if not sentence_pairs:
        return False
    
    for pair_idx, pair in enumerate(sentence_pairs):
        left, right = pair
        
        if len(left) != len(right):
            return False

        for i in range(len(left)):
            if left[i] != right[i]:
                return False
    return True
