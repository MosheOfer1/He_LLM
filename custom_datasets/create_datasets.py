import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd


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
