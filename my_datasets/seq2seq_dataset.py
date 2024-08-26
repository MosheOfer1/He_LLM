from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    def __init__(self, input_tensors, target_tensors):
        self.input_tensors = input_tensors
        self.target_tensors = target_tensors

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_tensors[idx],  # Match expected key
            "labels": self.target_tensors[idx]  # Match expected key
        }
