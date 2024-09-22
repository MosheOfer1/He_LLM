import torch
from torch.utils.data import Dataset


class ComboModelDataset(Dataset):
    def __init__(self, text_list: list[str], input_tokenizer, output_tokenizer, device='cpu'):

        self.device = device

        self.sentences_pair_tokens = self.create_pair_tokens_data(input_tokenizer, output_tokenizer, text_list)
        

        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

    def __len__(self):
        length = len(self.sentences_pair_tokens)
        if length <= 0:
            raise ValueError(
                f"Your ComboModelDataset.__len__ <= 0")
        return length

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary with 'input_ids' containing the tokenized Hebrew sentence and 
                  'labels' containing the associated label.
        """
        if idx > self.__len__():
            raise ValueError(f"Your ComboModelDataset len is: {self.__len__()} and you are trying to get idx: {idx}")
        
        sentence_pairs = self.sentences_pair_tokens[idx]
        
        # print(f"sentence_pairs: \n{sentence_pairs}")
        
        # Get input tokens (last token is the expected generated token)
        tokens = [pair[0] for pair in sentence_pairs[:-1]]
        
        # print(f"tokens = {tokens}")
        
        input_ids = self.input_tokenizer.encode(tokens,
                                                add_special_tokens=True, # Adds bos token
                                                return_tensors='pt'
                                                )

        # Get Labels tokens
        next_token = [pair[1] for pair in sentence_pairs]
        
        # print(next_token)
        
        labels = self.output_tokenizer.convert_tokens_to_ids(next_token)
                
        labels = torch.tensor(labels, dtype=torch.long)
        
        print(f"Dataset item ({idx}) \nLabels: {labels}")
    
        # for idx in range(len(tokens)):
        #     print(f"input: {tokens[idx]}, target: {next_token[idx]}")
        
        # print(f"Expected generated token: {next_token[len(tokens)]}")
        
        # raise("stop here pls")
    
        return {
            'input_ids': input_ids,
            'labels': labels,
        }

    def create_pair_tokens_data(self, input_tokenizer, output_tokenizer, list_text):
        print("in create_windows")
        
        pairs = []
        for text in list_text:
            sentence_pairs = align_tokens(input_tokenizer, output_tokenizer, text)
            pairs.append(sentence_pairs)
        
        # print(f"After align - {pairs}")
        
        return make_matching_pairs(sentences_pairs=pairs,
                                   output_tokenizer=output_tokenizer)

def make_matching_pairs(sentences_pairs: list[tuple], output_tokenizer):

    # print("in make_matching_pairs")
    
    """
        Use check_pair func in order to ignore unwanted windows.
    """
    ans =[]
    temp = []
            
    for sentence_idx, sentence_pairs in enumerate(sentences_pairs):

        
        if len(sentence_pairs) > 0:
            temp = []

            for pair in sentence_pairs:
                
                input, target = pair

                # Only 1 token for input
                if check_pair(pair, sentence_idx):
                    new_pair = (input[0], target[0])
                    temp.append(new_pair)
                    
                # More then 1 input token
                else:
                    flag = True
                    for idx in range(len(input)):
                        if flag:
                            new_pair = (input[idx], target[idx])
                            temp.append(new_pair)

                            if input[idx] != target[idx]:
                                flag = False
                        else:
                            print(f"input[0] = {input[0]}, target[0] = {target[0]}")
                            new_pair = (input[idx], get_new_target_token(idx, input, output_tokenizer))
                            temp.append(new_pair)
            ans.append(temp)
        
    return ans


def check_pair(pair, sentence_idx):
    """
        Check that the sentence is tokenized to 1 token per word
    """
    if not pair:
        raise ValueError(f"You have a problem in sentence {sentence_idx}, one of the pairs holds a None value")
    
    input, _ = pair
    
    len_input = len(input)
    
    if len_input == 1:
        return True

    return False

def get_new_target_token(idx, input, output_tokenizer):
    target_sentence = "".join([item.rstrip("_") for item in input[idx:-1]])


    
    print(f"target_sentence = {target_sentence}, len = {len(target_sentence)}")
    
    # new_token = output_tokenizer.tokenize(target_sentence, add_special_tokens=False)[0]
    
    tokens = output_tokenizer.encode(target_sentence, add_special_tokens=True)
    tokenized_sentence = output_tokenizer.convert_ids_to_tokens(tokens)
    
    new_token = tokenized_sentence[1] if len(tokenized_sentence) > 1 else tokenized_sentence[0]
    
    print(f"new token = {new_token}")
    
    # return the first token
    return new_token

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

