from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    def __init__(self, inputs, targets):
        """
        inputs: Tensor of shape (num_samples, seq_len, hidden_state_dim)
        targets: Tensor of shape (num_samples, seq_len, hidden_state_dim)
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'labels': self.targets[idx]
        }
