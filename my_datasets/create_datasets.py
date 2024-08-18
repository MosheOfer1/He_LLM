import pandas as pd
import torch
from transformers import AutoTokenizer, OPTForCausalLM
import os
from llm.llm_integration import LLMIntegration
from translation.translator import Translator


def load_sentences_from_csv(file_path, sentence_column):
    """
    Load sentences from a CSV file.

    :param file_path: Path to the CSV file.
    :param sentence_column: Name of the column containing sentences.
    :return: A list of sentences.
    """
    df = pd.read_csv(file_path)
    sentences = df[sentence_column].dropna().tolist()
    return sentences


def filter_sentences(sentences, min_length=5, max_length=50):
    return [sentence for sentence in sentences if min_length <= len(sentence.split()) <= max_length]


def pad_hidden_states(hidden_states, max_length):
    """
    Pads hidden states to the max length with zeros.

    :param hidden_states: The hidden states tensor of shape (batch_size, sequence_length, hidden_dim).
    :param max_length: The maximum length to pad the hidden states to.
    :return: A padded hidden states tensor of shape (batch_size, max_length, hidden_dim).
    """
    padding_size = max_length - hidden_states.size(1)
    if padding_size > 0:
        padding = torch.zeros((hidden_states.size(0), padding_size, hidden_states.size(2)))
        padded_hidden_states = torch.cat([hidden_states, padding], dim=1)
    else:
        padded_hidden_states = hidden_states
    return padded_hidden_states


def save_to_file(input_hidden_states_tensor, target_hidden_states_tensor, file_path):
    """
    Save tensors to a file, appending if the file exists.

    :param input_hidden_states_tensor: The input hidden states tensor.
    :param target_hidden_states_tensor: The target hidden states tensor.
    :param file_path: Path to the file where tensors should be saved.
    """
    if os.path.exists(file_path):
        existing_data = torch.load(file_path)
        input_hidden_states_tensor = torch.cat([existing_data['input'], input_hidden_states_tensor], dim=0)
        target_hidden_states_tensor = torch.cat([existing_data['target'], target_hidden_states_tensor], dim=0)

    torch.save({'input': input_hidden_states_tensor, 'target': target_hidden_states_tensor}, file_path)


def create_transformer1_dataset(translator: Translator, llm: LLMIntegration, file_path: str, save_interval: int = 100):
    """
    Create a dataset for training Transformer1 and save it to a file in chunks.

    :param translator: The translator instance used to generate hidden states.
    :param llm: The LLM instance used to generate hidden states.
    :param file_path: Path to the file where the dataset should be saved.
    :param save_interval: Number of sentences to process before saving to a file.
    """
    hebrew_sentences = load_sentences_from_csv("../my_datasets/hebrew_sentences.csv", "sentence")
    filtered_hebrew_sentences = filter_sentences(hebrew_sentences)

    llm_tokenizer = AutoTokenizer.from_pretrained(llm.model_name)
    llm_model = OPTForCausalLM.from_pretrained(llm.model_name)

    input_hidden_states_list = []
    target_hidden_states_list = []
    max_input_length = 0
    max_target_length = 0
    processed_count = 0

    for sentence in filtered_hebrew_sentences:
        # Step 1: Get the last hidden state from the first translator model
        with torch.no_grad():
            outputs = translator.get_output(
                from_first=True,
                text=sentence
            )
        input_hidden_states = outputs.decoder_hidden_states[-1]
        max_input_length = max(max_input_length, input_hidden_states.size(1))

        # Step 2: Translate the sentence
        translated_text = translator.decode_logits(
            tokenizer=translator.source_to_target_tokenizer,
            logits=outputs.logits
        )

        # Step 3: Pass the English translation through the LLM and get the first hidden state
        with torch.no_grad():
            target_hidden_states = llm.text_to_hidden_states(
                tokenizer=llm_tokenizer,
                model=llm_model,
                text=translated_text,
                layer_num=0
            )
        max_target_length = max(max_target_length, target_hidden_states.size(1))

        input_hidden_states_list.append(input_hidden_states)
        target_hidden_states_list.append(target_hidden_states)
        processed_count += 1

        if processed_count % save_interval == 0:
            # Pad all hidden states to the maximum length
            input_hidden_states_list = [pad_hidden_states(h, max_input_length) for h in input_hidden_states_list]
            target_hidden_states_list = [pad_hidden_states(h, max_target_length) for h in target_hidden_states_list]

            # Convert lists to tensors
            input_hidden_states_tensor = torch.stack(input_hidden_states_list, dim=0)
            target_hidden_states_tensor = torch.stack(target_hidden_states_list, dim=0)

            # Save tensors to a file
            save_to_file(input_hidden_states_tensor, target_hidden_states_tensor, file_path)

            # Reset lists for next batch
            input_hidden_states_list = []
            target_hidden_states_list = []
            max_input_length = 0
            max_target_length = 0

    # Save any remaining data after the loop
    if input_hidden_states_list:
        input_hidden_states_list = [pad_hidden_states(h, max_input_length) for h in input_hidden_states_list]
        target_hidden_states_list = [pad_hidden_states(h, max_target_length) for h in target_hidden_states_list]

        input_hidden_states_tensor = torch.stack(input_hidden_states_list, dim=0)
        target_hidden_states_tensor = torch.stack(target_hidden_states_list, dim=0)

        save_to_file(input_hidden_states_tensor, target_hidden_states_tensor, file_path)


def create_transformer2_dataset(translator: Translator, llm: LLMIntegration, file_path: str, save_interval: int = 100):
    """
    Create a dataset for training Transformer2 and save it to a file in chunks.

    :param llm: The LLM instance used to generate hidden states.
    :param translator: The translator instance used to generate hidden states.
    :param file_path: Path to the file where the dataset should be saved.
    :param save_interval: Number of sentences to process before saving to a file.
    """
    english_sentences = load_sentences_from_csv("../my_datasets/english_sentences.csv", "sentence")
    filtered_english_sentences = filter_sentences(english_sentences)

    llm_tokenizer = AutoTokenizer.from_pretrained(llm.model_name)
    llm_model = OPTForCausalLM.from_pretrained(llm.model_name)

    input_hidden_states_list = []
    target_hidden_states_list = []
    max_input_length = 0
    max_target_length = 0
    processed_count = 0

    for sentence in filtered_english_sentences:
        # Step 1: Pass the English sentence through the LLM and get the last hidden state
        with torch.no_grad():
            input_hidden_states = llm.text_to_hidden_states(
                tokenizer=llm_tokenizer,
                model=llm_model,
                text=sentence,
                layer_num=-1  # Last layer of the LLM
            )
        max_input_length = max(max_input_length, input_hidden_states.size(1))

        # Step 2: Translate the sentence to Hebrew using the second translator
        with torch.no_grad():
            outputs = translator.get_output(
                from_first=False,
                text=sentence
            )
        target_hidden_states = outputs.encoder_hidden_states[0]  # First layer of the second transformer
        max_target_length = max(max_target_length, target_hidden_states.size(1))

        input_hidden_states_list.append(input_hidden_states)
        target_hidden_states_list.append(target_hidden_states)
        processed_count += 1

        if processed_count % save_interval == 0:
            # Pad all hidden states to the maximum length
            input_hidden_states_list = [pad_hidden_states(h, max_input_length) for h in input_hidden_states_list]
            target_hidden_states_list = [pad_hidden_states(h, max_target_length) for h in target_hidden_states_list]

            # Convert lists to tensors
            input_hidden_states_tensor = torch.stack(input_hidden_states_list, dim=0)
            target_hidden_states_tensor = torch.stack(target_hidden_states_list, dim=0)

            # Save tensors to a file
            save_to_file(input_hidden_states_tensor, target_hidden_states_tensor, file_path)

            # Reset lists for next batch
            input_hidden_states_list = []
            target_hidden_states_list = []
            max_input_length = 0
            max_target_length = 0

    # Save any remaining data after the loop
    if input_hidden_states_list:
        input_hidden_states_list = [pad_hidden_states(h, max_input_length) for h in input_hidden_states_list]
        target_hidden_states_list = [pad_hidden_states(h, max_target_length) for h in target_hidden_states_list]

        input_hidden_states_tensor = torch.stack(input_hidden_states_list, dim=0)
        target_hidden_states_tensor = torch.stack(target_hidden_states_list, dim=0)

        save_to_file(input_hidden_states_tensor, target_hidden_states_tensor, file_path)
