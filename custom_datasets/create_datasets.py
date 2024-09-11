import torch
import random

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from llm.llm_wrapper import LLMWrapper
from custom_datasets.seq2seq_dataset import Seq2SeqDataset
from translation.translator import Translator


def load_and_create_dataset(file_path: str) -> Seq2SeqDataset:
    """
    Load tensors from a file and create a Seq2SeqDataset instance.

    :param file_path: Path to the file where the tensors are saved.
    :return: A Seq2SeqDataset instance created from the loaded tensors.
    """
    # Load the saved tensors
    loaded_data = torch.load(file_path)
    input_tensor = loaded_data['input']
    target_tensor = loaded_data['target']

    # Create Seq2SeqDataset instance
    dataset = Seq2SeqDataset(input_tensor, target_tensor)

    return dataset


def create_transformer1_dataset(
        translator: Translator,
        llm: LLMWrapper,
        file_path: str,
        max_length=15,
        dataset_name='hebrew_sentences.csv',
        sentence_num=900,
        test_portion=0.3,
        chunk_size=100,
        starting_point=0
) -> (Seq2SeqDataset, Seq2SeqDataset):
    """
    Create a dataset for training Transformer1 and save it to a file in chunks.

    :param starting_point: Where to start taking sentences from the dataset_name.
    :param chunk_size: The number of samples to save in each chunk.
    :param test_portion: Proportion of the dataset to use for testing.
    :param sentence_num: The number of sentences to use from the dataset.
    :param dataset_name: Name of the dataset file containing Hebrew sentences.
    :param max_length: Maximum sequence length for padding.
    :param translator: The translator instance used to generate hidden states.
    :param llm: The LLM instance used to generate hidden states.
    :param file_path: Path to the file where the dataset should be saved.
    """
    if dataset_name.endswith('.csv'):
        hebrew_sentences = load_sentences_from_csv(f"my_datasets/{dataset_name}", "sentence")
    elif dataset_name.endswith('.txt'):
        hebrew_sentences = read_file_lines(f"my_datasets/{dataset_name}")
    else:
        raise ValueError("Dataset file must be either .csv or .txt")

    # Filter and limit the number of sentences
    filtered_hebrew_sentences = filter_sentences(hebrew_sentences)[starting_point:starting_point+sentence_num]

    # Split into training and test sets
    random.shuffle(filtered_hebrew_sentences)
    test_size = int(len(filtered_hebrew_sentences) * test_portion)
    test_sentences = filtered_hebrew_sentences[:test_size]
    train_sentences = filtered_hebrew_sentences[test_size:]

    # Define the EOS vector (e.g., a vector of zeros or a specific learned vector)
    eos_vector_input = torch.zeros(translator.src_to_target_model.config.hidden_size)
    eos_vector_output = torch.zeros(llm.model.config.hidden_size)

    # Process training and test sentences in chunks
    for i in range(0, len(train_sentences), chunk_size):
        chunk_train_sentences = train_sentences[i:i + chunk_size]
        chunk_train_input_tensor, chunk_train_target_tensor = process_sentences(chunk_train_sentences, translator, llm,
                                                                                eos_vector_input, eos_vector_output,
                                                                                max_length)
        save_to_file(chunk_train_input_tensor, chunk_train_target_tensor, file_path + f"transformer1_dataset_train_{sentence_num}.pt")
        print(f"added the {(i+1)} chunk to the file for training")

    for i in range(0, len(test_sentences), chunk_size):
        chunk_test_sentences = test_sentences[i:i + chunk_size]
        chunk_test_input_tensor, chunk_test_target_tensor = process_sentences(chunk_test_sentences, translator, llm,
                                                                              eos_vector_input, eos_vector_output,
                                                                              max_length)
        save_to_file(chunk_test_input_tensor, chunk_test_target_tensor, file_path + f"transformer1_dataset_test_{sentence_num}.pt")
        print(f"added the {(i+1)} chunk to the file for testing")

    # Create Seq2SeqDataset instances
    train_dataset = Seq2SeqDataset(torch.load(file_path + f"transformer1_dataset_train_{sentence_num}.pt")['input'],
                                   torch.load(file_path + f"transformer1_dataset_train_{sentence_num}.pt")['target'])
    test_dataset = Seq2SeqDataset(torch.load(file_path + f"transformer1_dataset_test_{sentence_num}.pt")['input'],
                                  torch.load(file_path + f"transformer1_dataset_test_{sentence_num}.pt")['target'])

    return train_dataset, test_dataset


def process_sentences(sentences, translator, llm, eos_vector_in, eos_vector_out, max_length):
    input_hidden_states_list = []
    target_hidden_states_list = []

    for sentence in sentences:
        # Step 1: Get the last hidden state from the first translator model
        with torch.no_grad():
            outputs = translator.get_output(from_first=True, text=sentence)
        input_hidden_states = outputs.decoder_hidden_states[-1]  # Shape: (seq_len, hidden_dim)

        # Step 2: Translate the sentence
        translated_text = translator.decode_logits(
            tokenizer=translator.src_to_target_tokenizer,
            logits=outputs.logits
        )

        # Step 3: Pass the English translation through the LLM and get the first hidden state
        with torch.no_grad():
            llm.model.base_model.decoder.layers[llm.injected_layer_num].set_injection_state(False)
            target_hidden_states = llm.text_to_hidden_states(
                tokenizer=llm.tokenizer,
                model=llm.model,
                text=translated_text,
                layer_num=0  # Assuming this returns a tensor of shape (seq_len, hidden_dim)
            )
            llm.model.base_model.decoder.layers[llm.injected_layer_num].set_injection_state(True)

        input_hidden_states_list.append(input_hidden_states)
        target_hidden_states_list.append(target_hidden_states)

    # Pad all hidden states to the maximum length
    input_hidden_states_list = [pad_target_with_eos(h.squeeze(0), max_length, eos_vector_in) for h in input_hidden_states_list]
    target_hidden_states_list = [pad_target_with_eos(h.squeeze(0), max_length, eos_vector_out) for h in target_hidden_states_list]

    # Convert lists to tensors
    input_hidden_states_tensor = torch.stack(input_hidden_states_list,
                                             dim=0)  # Shape: (batch_size, max_length, hidden_dim)
    target_hidden_states_tensor = torch.stack(target_hidden_states_list,
                                              dim=0)  # Shape: (batch_size, max_length, hidden_dim)

    return input_hidden_states_tensor, target_hidden_states_tensor


def pad_target_with_eos(target_hidden_states, max_length, eos_vector):
    """
    Pad target hidden states with the EOS vector at the end.

    :param target_hidden_states: Tensor of shape (seq_len, hidden_dim)
    :param max_length: Maximum sequence length for padding
    :param eos_vector: Tensor of shape (hidden_dim,) representing the EOS vector
    :return: Padded tensor of shape (max_length, hidden_dim)
    """
    seq_len, hidden_dim = target_hidden_states.shape
    if seq_len < max_length:
        # Calculate how many EOS vectors to add
        num_padding = max_length - seq_len
        # Create a padding tensor with EOS vectors
        padding = eos_vector.unsqueeze(0).expand(num_padding, hidden_dim)
        padded_hidden_states = torch.cat([target_hidden_states, padding], dim=0)
    else:
        # Truncate to max_length and replace the last vector with the EOS vector
        padded_hidden_states = target_hidden_states[:max_length]
        padded_hidden_states[-1] = eos_vector
    return padded_hidden_states


def save_to_file(input_tensor, target_tensor, file_path):
    """
    Save tensors to a file in chunks by appending to the existing file.

    :param input_tensor: The input tensor to be saved.
    :param target_tensor: The target tensor to be saved.
    :param file_path: The file path to save the tensors.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Check if the file exists
    if os.path.exists(file_path):
        # Load the existing tensors from the file
        existing_data = torch.load(file_path)
        existing_input = existing_data['input']
        existing_target = existing_data['target']
    else:
        existing_input = torch.tensor([])
        existing_target = torch.tensor([])

    # Concatenate the new tensors with the existing ones
    new_input = torch.cat([existing_input, input_tensor], dim=0)
    new_target = torch.cat([existing_target, target_tensor], dim=0)

    # Save the combined tensors to the file
    torch.save({'input': new_input, 'target': new_target}, file_path, _use_new_zipfile_serialization=False)


def create_transformer2_dataset(
        translator: Translator,
        llm: LLMWrapper,
        file_path: str,
        save_interval: int = 100,
        max_length=15,
        dataset_name='english_sentences.csv'
):
    """
    Create a dataset for training Transformer2 and save it to a file in chunks.

    :param dataset_name:
    :param max_length:
    :param llm: The LLM instance used to generate hidden states.
    :param translator: The translator instance used to generate hidden states.
    :param file_path: Path to the file where the dataset should be saved.
    :param save_interval: Number of sentences to process before saving to a file.
    """
    pass


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


def read_file_lines(file_name):
    """
    Reads a text file and returns a list of its lines.

    Args:
        file_name (str): The name of the text file to read.

    Returns:
        list: A list of strings, each representing a line in the file.
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            return [line.strip() for line in lines]  # Remove newline characters
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return []
    except IOError:
        print(f"Error: An IOError occurred while reading the file '{file_name}'.")
        return []


def read_file_to_string(file_path):
    """
    Reads the content of a file and returns it as a string.

    :param file_path: Path to the file to be read
    :return: String containing the content of the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"


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
