import torch
from torch.utils.data import Dataset


class ComboModelDataset(Dataset):
    def __init__(self, text_list: list[str], input_tokenizer, output_tokenizer, window_size=5, device='cpu'):

        self.device = device

        self.windows = self.create_windows(input_tokenizer, output_tokenizer, text_list, window_size)
        

        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.window_size = window_size

    def __len__(self):
        length = len(self.windows)
        if length <= 0:
            raise ValueError(
                f"Your ComboModelDataset.__len__ <= 0. Check if your window_size is greater then your data size")
        return length

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary with 'input_ids' containing the tokenized Hebrew sentence and 
                  'labels' containing the associated label.
        """
        if idx > self.__len__():
            raise ValueError(f"Your ComboModelDataset len is: {self.__len__()} and you are trying to get idx: {idx}")
        
        window = self.windows[idx]
        
        # print(f"window: {window}")
        
        # Get input tokens
        tokens = [pair[0][0] for pair in window[:-1]]
        
        input_ids = self.input_tokenizer.encode(tokens,
                                                add_special_tokens=True, # Adds bos token
                                                return_tensors='pt'
                                                ).to(self.device)

        # Get Labels tokens
        next_token = [pair[1][0] for pair in window]
        
        labels = self.output_tokenizer.convert_tokens_to_ids(next_token)
                
        labels = torch.tensor([labels], dtype=torch.long)
        
        # print(f"Dataset item ({idx}) Labels: {labels}")
    
        return {
            'input_ids': input_ids,
            'labels': labels,
        }

    def create_windows(self, input_tokenizer, output_tokenizer, list_text, window_size):
        print("in create_windows")
        
        pairs = []
        for text in list_text:
            sentence_pairs = align_tokens(input_tokenizer, output_tokenizer, text)
            pairs.append(sentence_pairs)
        windows = delete_unmatched_pairs(pairs, window_size)

        return windows

        
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


def delete_unmatched_pairs(sentences_pairs: list[tuple], window: int):
    
    print("in delete_unmatched_pairs")
    
    """
        Use check_pair func in order to ignore unwanted windows.
    """
    windows =[]
    temp = []
    
    # print(f"Before deleting: {sentences_pairs[:10]}")
    
    for sentence_pairs in sentences_pairs:

        temp.clear()
        
        for pair in sentence_pairs:
            
            if len(temp) == window - 1:
                windows.append(temp + [pair])
                temp.pop(0)
                
            if check_pair(pair):
                temp.append(pair)
            else:
                temp.clear()

    # print(f"I have {len(windows)} windows")
    # print(f"Windows example: {windows[:10]}")
    return windows


def check_pair(pair):
    """
        Check that the sentence is tokenized to 1 token per word
    """
    if not pair:
        return False
    
    left, _ = pair
    
    if len(left) != 1:
        return False

    return True
