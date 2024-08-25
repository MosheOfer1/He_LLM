from torch.utils.data import Dataset
import pandas as pd
import torch


class HebrewDataset(Dataset):
    def __init__(self, data: pd.DataFrame, input_tokenizer, output_tokenizer, max_length: int = 128):
        """
        Args:
            data (DataFrame): The input data containing Hebrew sentences and corresponding labels (both as strings).
            tokenizer: A tokenizer to convert text into input_ids.
            max_length (int): The maximum sequence length for padding/truncation.
        """
        self.data = data
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.max_length = max_length
        
        class_weights = (1 - (self.data["label"].value_counts().sort_index() / len(self.data))).values
        
        self.class_weights = torch.from_numpy(class_weights).float()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary with 'input_ids' containing the tokenized Hebrew sentence and 
                  'labels' containing the associated label.
        """
        # Get the raw Hebrew sentence and the label (both are strings)
        sentence = self.data.iloc[idx]['Hebrew sentence']
        label = self.data.iloc[idx]['label']

        # # Tokenize the sentence
        # tokenized_input = self.input_tokenizer(
        #     sentence,
        #     return_tensors='pt',
        # )
        
        # input_ids_len = tokenized_input.input_ids
        
        # Tokenize the sentence
        tokenized_input = self.input_tokenizer(
            sentence,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # Tokenize the label
        tokenized_label = self.output_tokenizer(
            sentence + label,
            return_tensors='pt',
            # max_length=1,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        input_ids = tokenized_input.input_ids
        attention_mask = tokenized_input.attention_mask
        label = tokenized_label.input_ids.squeeze(0)
        # label = tokenized_label.input_ids.squeeze(0)[0]
        
        # print(input_ids.shape)
        # print(f"input_ids shape: {input_ids.shape}")
        # print(f"attention_mask shape: {attention_mask.shape}")

        return {
            'input_ids': input_ids,
            'text': sentence,
            'attention_mask': attention_mask, 
            'labels': label,
            'class_weights': self.class_weights
        }
