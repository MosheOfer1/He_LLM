import torch
from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    def __init__(self, sentences, translator, llm, device='cpu'):
        """
        Args:
            sentences (list): List of Hebrew sentences.
            translator (Translator): The translator wrapper.
            llm (LLMWrapper): The language model wrapped with the injection state context manager.
        """
        self.device = device
        self.sentences = sentences
        self.translator = translator
        self.llm = llm

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Step 1: Get the last hidden state from the first translator model
        with torch.no_grad():
            outputs = self.translator.get_output(from_first=True, text=sentence)
        input_hidden_states = outputs.decoder_hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)

        # Step 2: Translate the sentence
        translated_text = self.translator.decode_logits(
            tokenizer=self.translator.src_to_target_tokenizer,
            logits=outputs.logits
        )

        # Step 3: Pass the English translation through the LLM and get the first hidden state
        with self.llm.injection_state():
            with torch.no_grad():
                target_hidden_states = self.llm.text_to_hidden_states(
                    tokenizer=self.llm.tokenizer,
                    model=self.llm.model,
                    text=translated_text,
                    layer_num=0
                )

        input_hidden_states = input_hidden_states.squeeze(0)
        target_hidden_states = target_hidden_states.squeeze(0)

        # Return the input hidden states and target hidden states
        return {
            "input_ids": input_hidden_states,  # Last layer of translator
            "labels": target_hidden_states  # First layer of LLM
        }
