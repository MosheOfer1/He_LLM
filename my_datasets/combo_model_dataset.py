from torch.utils.data import Dataset


class ComboModelDataset(Dataset):
    def __init__(self, text: str, input_tokenizer, output_tokenizer, window_size=5):
        self.data = input_tokenizer.encode(text, add_special_tokens=False)
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary with 'input_ids' containing the tokenized Hebrew sentence and 
                  'labels' containing the associated label.
        """
        input_ids = self.data[idx:idx+self.window_size]
        next_token_id = self.data[idx+self.window_size]
        next_token = self.input_tokenizer.decode([next_token_id], skip_special_tokens=True)

        with self.output_tokenizer.as_target_tokenizer():
            label = self.output_tokenizer.encode(next_token, add_special_tokens=False)[0]

        return {
            'input_ids': input_ids,
            'labels': label,
        }
