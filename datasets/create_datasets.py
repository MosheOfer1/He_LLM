import torch
from torch.utils.data import TensorDataset
import csv


def load_hebrew_sentences(csv_file_path):
    """
    Load Hebrew sentences from a CSV file.

    :param  csv_file_path: Path to the CSV file containing Hebrew sentences.
    :return: A list of Hebrew sentences.
    """
    sentences = []
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:  # Ensure the row is not empty
                sentences.append(row[0])
    return sentences


def create_transformer1_dataset(translator, llm, file_name):
    """
    Create a dataset for training Transformer1.

    :param translator: The translator instance used to generate hidden states.
    :param llm: The LLM instance used to generate hidden states.
    :param file_name: The name of the CSV file containing Hebrew sentences.
    :return: A TensorDataset containing input-output pairs for Transformer1.
    """
    hebrew_sentences = load_hebrew_sentences(file_name)

    input_hidden_states_list = []
    target_hidden_states_list = []

    for sentence in hebrew_sentences:
        # Step 1: Get the last hidden state from the translation model
        input_hidden_states = translator.translate_to_en_returns_hidden_states(sentence)

        # Step 2: Pass the English translation through the LLM and get the first hidden state
        with torch.no_grad():
            target_hidden_states = llm.model(translator.translate_to_target(sentence))

        input_hidden_states_list.append(input_hidden_states)
        target_hidden_states_list.append(target_hidden_states)

    # Convert lists to tensors
    input_hidden_states_tensor = torch.cat(input_hidden_states_list, dim=0)
    target_hidden_states_tensor = torch.cat(target_hidden_states_list, dim=0)

    return TensorDataset(input_hidden_states_tensor, target_hidden_states_tensor)


def create_transformer2_dataset(translator, llm, file_name):
    """
    Create a dataset for training Transformer2.

    :param translator: The translator instance used to generate hidden states.
    :param llm: The LLM instance used to generate hidden states.
    :param file_name: The name of the CSV file containing Hebrew sentences.
    :return: A TensorDataset containing input-output pairs for Transformer2.
    """
    hebrew_sentences = load_hebrew_sentences(file_name)

    input_hidden_states_list = []
    target_hidden_states_list = []

    for sentence in hebrew_sentences:
        # Step 1: Pass the English translation through the LLM and get the last hidden state
        english_translation = translator.translate_to_target(sentence)
        with torch.no_grad():
            input_hidden_states = llm.process_text_input(english_translation)

        # Step 2: Translate English back to Hebrew and get the first hidden state
        target_hidden_states = translator.translate_hidden_to_source(input_hidden_states)

        input_hidden_states_list.append(input_hidden_states)
        target_hidden_states_list.append(target_hidden_states)

    # Convert lists to tensors
    input_hidden_states_tensor = torch.cat(input_hidden_states_list, dim=0)
    target_hidden_states_tensor = torch.cat(target_hidden_states_list, dim=0)

    return TensorDataset(input_hidden_states_tensor, target_hidden_states_tensor)
