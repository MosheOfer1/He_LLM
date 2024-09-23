import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import Seq2SeqTrainer
from llm.llm_wrapper import LLMWrapper


def create_positional_encoding(max_seq_len, hidden_dim):
    # Initialize the positional encoding matrix
    position = torch.arange(0, max_seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-torch.log(torch.tensor(10000.0)) / hidden_dim))

    positional_encoding = torch.zeros(max_seq_len, hidden_dim)
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)

    return positional_encoding.unsqueeze(0)  # Shape (1, max_seq_len, hidden_dim)


class BlackBox:
    def __init__(self, llm: LLMWrapper):
        self.llm = llm


def logits_from_first_layer(llm, hidden_states):
    # Inject the noised hidden states back into the model
    llm.inject_hidden_states(hidden_states)
    # Get the logits after injecting the hidden states
    batch_size = hidden_states.shape[0]
    token_num = hidden_states.shape[1]
    llm_outputs = llm.get_output_by_using_dummy(token_num=token_num, batch_size=batch_size)
    predicted_logits = llm_outputs.logits
    return predicted_logits


def calculate_KL_div(llm, outputs, label, mask=None):
    predicted_logits = logits_from_first_layer(llm, outputs)
    true_logits = logits_from_first_layer(llm, label)

    # Convert logits to probabilities using softmax
    predicted_probs = F.softmax(predicted_logits, dim=-1)
    original_probs = F.softmax(true_logits, dim=-1)

    # Calculate KL divergence between the original and predicted logits (per token)
    kl_div = F.kl_div(predicted_probs.log(), original_probs, reduction='none')

    if mask is not None:
        # Reshape mask to be applied on all tokens, ensuring it matches the sequence shape
        mask = mask.unsqueeze(-1).expand_as(kl_div)  # Shape: (batch_size, seq_len, vocab_size)
        kl_div = kl_div * mask  # Apply the mask to ignore padding positions

        # Sum over the vocabulary dimension, apply mask
        kl_div = kl_div.sum(dim=-1)  # Sum over the vocab_size

        # Calculate the total number of non-masked elements
        non_padded_elements = mask.sum() / mask.size(-1)  # Normalize by the vocab_size
    else:
        # Sum over the vocabulary dimension
        kl_div = kl_div.sum(dim=-1)
        non_padded_elements = kl_div.numel()

    # Calculate the average KL divergence over the batch and sequence length
    kl_div_mean = kl_div.sum() / non_padded_elements

    return kl_div_mean


class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        device = model.module.device if hasattr(model, 'module') else model.device

        input_ids = inputs.get("input_ids").to(device)
        labels = inputs.get("labels").to(device)
        input_mask = inputs.get("input_mask").to(device)
        label_mask = inputs.get("label_mask").to(device)

        outputs = model(input_ids, labels, input_mask=input_mask, label_mask=label_mask)

        # loss_fct = nn.MSELoss()
        # loss = loss_fct(outputs, labels)

        # cosine_sim = F.cosine_similarity(outputs, labels, dim=-1)
        # loss = 1 - cosine_sim.mean()

        llm = model.module.black_box.llm if hasattr(model, 'module') else model.black_box.llm
        loss = calculate_KL_div(llm, outputs, labels, label_mask)

        return (loss, outputs) if return_outputs else loss


def pad_hidden_states(hidden_states, max_len, device='cpu'):
    """Pad hidden states to a fixed length with ones and create a mask."""
    seq_len, hidden_dim = hidden_states.shape
    pad_size = max_len - seq_len
    padding = torch.ones((pad_size, hidden_dim)).to(device)  # Pad with ones

    # Create attention mask: 1s for non-padded tokens, 0s for padded tokens
    mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_size)]).to(device)

    padded_hidden_states = torch.cat([hidden_states, padding], dim=0).to(device)

    return padded_hidden_states, mask


def collate_fn(batch, max_seq_len=None, device='cpu'):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Determine the maximum sequence length in the batch
    max_input_len = max([x.size(0) for x in input_ids]) + 1
    max_label_len = max([x.size(0) for x in labels]) + 1

    # Set the max sequence length to pad if provided, otherwise use the batch max
    max_input_len = min(max_seq_len, max_input_len) if max_seq_len is not None else max_input_len
    max_label_len = min(max_seq_len, max_label_len) if max_seq_len is not None else max_label_len

    # Pad input_ids and labels using your custom pad_hidden_states function, and get masks
    padded_input_ids, input_mask = zip(*[pad_hidden_states(x, max_input_len, device=device) for x in input_ids])
    padded_labels, label_mask = zip(*[pad_hidden_states(x, max_label_len, device=device) for x in labels])

    # Stack tensors
    padded_input_ids = torch.stack(padded_input_ids).to('cpu')
    padded_labels = torch.stack(padded_labels).to('cpu')
    input_mask = torch.stack(input_mask).to('cpu')
    label_mask = torch.stack(label_mask).to('cpu')

    return {
        'input_ids': padded_input_ids,
        'labels': padded_labels,
        'input_mask': input_mask,
        'label_mask': label_mask
    }

