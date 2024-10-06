import sys
import os
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.gpt2_llm import GPT2LLM
from llm.opt_llm import OptLLM
from models.combined_model import MyCustomModel
from facade import create_datasets_from_txt_file, save_model, train


def main(args):
    device = args.device
    print(f"I'm working with: {device}")

    # Create the custom LLM model
    customLLM = MyCustomModel(
        args.translator1_model_name,
        args.translator2_model_name,
        args.llm_model_name,
        GPT2LLM if args.llm_class == "GPT2LLM" else OptLLM,
        device=device
    )

    # Create the datasets
    train_dataset, eval_dataset = create_datasets_from_txt_file(
        translator=customLLM.translator,
        text_file_path=args.text_file_path,
        device=device
    )

    # Train the model
    train(
        model=customLLM,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batches=args.batch_size,
        device=device
    )

    # Generate a unique model name
    model_name = f"{args.text_file_path.split('/')[-1][:5]}_{args.llm_model_name.split('/')[-1]}_{args.translator1_model_name.split('/')[-1]}_{args.translator2_model_name.split('/')[-1]}.pth"
    model_name = model_name.replace('.', '_')  # To avoid issues with file name

    # Directory to save the model
    model_dir = args.model_dir

    # Save the model
    save_model(
        model=customLLM,
        model_name=model_name,
        model_dir=model_dir
    )

    # Print the save location
    save_path = os.path.join(model_dir, model_name)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Train a custom LLM model with specific translators and LLM.")

    parser.add_argument('--translator1_model_name', type=str, required=True,
                        help="Translator 1 model name (e.g., 'Helsinki-NLP/opus-mt-tc-big-he-en')")

    parser.add_argument('--translator2_model_name', type=str, required=True,
                        help="Translator 2 model name (e.g., 'Helsinki-NLP/opus-mt-en-he')")

    parser.add_argument('--llm_model_name', type=str, required=True,
                        help="LLM model name (e.g., 'facebook/opt-350m')")

    parser.add_argument('--llm_class', type=str, required=True, choices=['GPT2LLM', 'OptLLM'],
                        help="LLM class (either 'GPT2LLM' or 'OptLLM')")

    parser.add_argument('--text_file_path', type=str, required=True,
                        help="Path to the text file (e.g., 'my_datasets/ynet_256k.txt')")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training")

    parser.add_argument('--model_dir', type=str, default="my_models",
                        help="Directory where the model will be saved")
    parser.add_argument('--device', type=str, default='cpu',
                        help="The device cuda/cpu")
    args = parser.parse_args()
    main(args)
