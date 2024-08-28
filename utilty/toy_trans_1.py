from transformers import AutoTokenizer, OPTForCausalLM
from custom_transformers.short_rnn_transformer1 import Transformer1
from llm.llm_integration import LLMWrapper
from my_datasets.create_datasets import create_transformer1_dataset, load_and_create_dataset
from translation.helsinki_translator import HelsinkiTranslator


def load_dataset_and_train_model(dataset_name="SVLM_Hebrew_Wikipedia_Corpus.txt", sentence_num=10000, test_portion=0, save_interval=100):
    """
    Load or create the dataset and train the Transformer1 model.

    :param dataset_name: The name of the dataset file containing Hebrew sentences.
    :param sentence_num: The number of sentences to use from the dataset.
    :param test_portion: Proportion of the dataset to use for testing.
    :param save_interval: Number of sentences to process before saving to a file.
    """
    # Initialize the translator and LLM
    translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
    translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
    translator = HelsinkiTranslator(translator1_model_name, translator2_model_name)

    llm_model_name = "facebook/opt-125m"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = OPTForCausalLM.from_pretrained(llm_model_name)
    llm = LLMWrapper(llm_model_name, llm_tokenizer, llm_model)

    model_name = input("model_name: ")
    # Initialize Transformer1
    transformer1 = Transformer1.load_model(model_name=model_name, translator=translator, llm=llm)

    file_path = '../my_datasets/'

    # Create or load the dataset
    train_dataset_path = file_path + input("Enter train dataset name")
    test_dataset_path = file_path + input("Enter test dataset name")

    try:
        # Try to load the datasets
        train_ds = load_and_create_dataset(train_dataset_path)
        test_ds = load_and_create_dataset(test_dataset_path)
        print(f"Datasets loaded from {train_dataset_path} and {test_dataset_path}")
    except FileNotFoundError:
        print("Failed to load the dataset")
        # If the datasets do not exist, create them
        train_ds, test_ds = create_transformer1_dataset(
            translator,
            llm,
            file_path,
            dataset_name=dataset_name,
            sentence_num=sentence_num,
            test_portion=test_portion,
            chunk_size=save_interval,
            starting_point=5000
        )
        print(f"Datasets created and saved to {file_path}")

    # Train the model
    transformer1.train_model(train_ds, test_ds)


# Call the function to load the dataset and train the model
load_dataset_and_train_model()
