import torch
from torch.utils.data import Dataset


class ComboModelDataset(Dataset):
    def __init__(self, text: str, input_tokenizer, output_tokenizer, window_size=10, device='cpu'):
        
        self.device = device
        
        text = _filter_data(text=text,
                            tokenizer=input_tokenizer)
        
        with output_tokenizer.as_target_tokenizer():
            text = _filter_data(text=text,
                                tokenizer=output_tokenizer)
        
        self.token_pairs = align_tokens(input_tokenizer, output_tokenizer, text)
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.window_size = window_size
        self.counter = 0
        
    def __len__(self):
        length = len(self.token_pairs) - self.window_size - 1
        if length <= 0:
            raise ValueError(f"Your ComboModelDataset.__len__ <= 0. Check if your window_size is greater then you data size (clean data)")
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
        
        if(len(outputs) > 0):
            label = outputs[0]
        else:
            self.counter += 1
            print(f"Got unexpected empty list from __getitem__. Sending default: [0] tensor. Counter: {self.counter}")
            label = self.output_tokenizer(
                text_target="a",
                add_special_tokens=False
            )["input_ids"][0]
            
        input_ids = torch.tensor(input_ids, dtype=torch.long)#.to(self.device)
        labels = torch.tensor(label, dtype=torch.long)#.to(self.device)
        return {
            'input_ids': input_ids,
            'labels': labels,
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

            encoded = self.input_tokenizer.encode(tokens, add_special_tokens=False)
            
            # Maybe Bug
            input_ids.extend(encoded)
        input_ids = input_ids[(len(input_ids) - self.window_size):len(input_ids)]
        input_ids.append(self.input_tokenizer.eos_token_id)
        return input_ids


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

def _filter_data(text, tokenizer):
    
    data = tokenizer(text, add_special_tokens=False)
    
    clean_data = tokenizer.decode(data['input_ids'], skip_special_tokens=True)
        
    print(f"len(data) = {len(data)}, len(clean_data) = {len(clean_data)}")

    return clean_data