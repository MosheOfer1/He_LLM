import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.opt_llm import OptLLM


def calculate_kl_with_random_injection(llm, sentence, random_std=0.1):
    """
    Injects a random vector into the model, and calculates the KL divergence between
    the original and the random-injected output logits.

    Parameters:
    llm (OptLLM): The language model instance.
    original_hidden_states (torch.Tensor): The original hidden states.
    sentence (str): The input sentence.
    random_std (float): The standard deviation for the random noise vector.

    Returns:
    kl_div (float): The KL divergence between the original and random-injected logits.
    """

    # Step 1: Get the original logits for comparison
    with llm.injection_state():
        with torch.no_grad():
            outputs = llm.process_text_input_to_outputs(sentence)
            original_logits = outputs.logits
            original_probs = F.softmax(original_logits, dim=-1)
            original_hidden_states = outputs.hidden_states[0]

    # Step 2: Generate a random vector with the same shape as the hidden states
    random_vector = torch.normal(mean=0.0, std=random_std, size=original_hidden_states.shape).to(
        original_hidden_states.device)

    # Step 3: Inject the random vector into the model
    llm.inject_hidden_states(random_vector)

    # Step 4: Get the logits after injecting the random vector
    batch_size = random_vector.shape[0]
    token_num = random_vector.shape[1]

    with torch.no_grad():
        random_injected_outputs = llm.get_output_by_using_dummy(token_num=token_num, batch_size=batch_size)
        random_injected_logits = random_injected_outputs.logits

    # Step 5: Convert logits to probabilities using softmax
    random_injected_probs = F.softmax(random_injected_logits, dim=-1)

    # Step 6: Calculate KL divergence between the original and random-injected probabilities
    kl_div = F.kl_div(original_probs.log(), random_injected_probs, reduction='batchmean').item()

    return kl_div


# Function to add Gaussian noise
def add_gaussian_noise(tensor, mean=0.0, std=0.1):
    noise = torch.normal(mean=mean, std=std, size=tensor.shape).to(tensor.device)
    return tensor + noise


def plot_predictions(sentence_idx, noised_probs, mse, num):
    # Get top 5 tokens for noised probs
    top_noised_probs, top_noised_indices = torch.topk(noised_probs, num, dim=-1)
    top_noised_tokens = [llm.tokenizer.decode([idx.item()]) for idx in top_noised_indices[0, -1]]

    print(f"Noised Probs - Top {num} Tokens (Noise {noise_level:.2f}) for Sentence {sentence_idx + 1}:")
    for i, (prob, idx) in enumerate(zip(top_noised_probs[0, -1], top_noised_indices[0, -1])):
        token = llm.tokenizer.decode([idx.item()])  # Ensure the index is a scalar
        print(f"{i + 1}: Token: {token}, Prob: {prob.item()}")

    print("\n" + "=" * 40 + "\n")

    # Plot the probability distribution for original vs noised
    plot_prob_distribution(
        top_original_probs[0, -1],
        top_noised_probs[0, -1],
        list(zip(top_tokens, top_noised_tokens)),
        noise_level,
        mse,
        num,
        sentence_idx
    )


# Function to plot stacked bars for original and noised probabilities
def plot_prob_distribution(original_probs, noised_probs, tokens, noise_level, mse, num, sentence_idx):
    n_tokens = len(tokens)

    x = np.arange(n_tokens)  # Token indices
    width = 0.35  # Bar width

    # Convert PyTorch tensors to numpy arrays
    original_probs_np = original_probs.detach().cpu().numpy()  # Detach from graph
    noised_probs_np = noised_probs.detach().cpu().numpy()  # Detach from graph

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot original probabilities
    ax.bar(x - width / 2, original_probs_np, width, label='Original Probs', color='blue')

    # Plot noised probabilities
    ax.bar(x + width / 2, noised_probs_np, width, label=f'Noised Probs (Noise={noise_level:.2f}) (MSE={mse})', color='red')

    # Add token labels
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=10, ha='center', fontsize=10)

    # Add labels and title
    ax.set_ylabel('Probability')
    ax.set_title(f'Sentence {sentence_idx + 1} - Top {num} Token Probability Distribution (Noise Level: {noise_level:.2f})')
    ax.legend()
    image_name = f'Noise={noise_level:.2f} MSE={mse:.2f} Sentence {sentence_idx + 1}'.replace('.', '_') + '.png'
    plt.savefig(f'../images/' + image_name)
    plt.close()  # Close the figure after saving to avoid warning


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Im working with: {device}")

    model_name = 'facebook/opt-125m'

    llm = OptLLM(
        model_name=model_name,
        device=device
    )

    sentences = [
        "Artificial intelligence will shape the future of",
        "Quantum computing holds the key to solving complex",
        "The mysteries of the deep ocean are yet to be",
        "Exploring Mars could unlock the secrets of life beyond"
    ]
    random_kl = calculate_kl_with_random_injection(llm=llm, sentence=sentences[0])
    print("KL-divergence with a random vector is: ", random_kl)

    mse_all_sentences = []
    kl_all_sentences = []

    noise_levels = np.linspace(0.01, 0.18, 8)

    for sentence_idx, sentence in enumerate(sentences):
        mse_values = []
        kl_values = []

        with llm.injection_state():
            with torch.no_grad():
                outputs = llm.process_text_input_to_outputs(sentence)
                first_hidden_states = outputs.hidden_states[0]
                logits = outputs.logits

        original_probs = F.softmax(logits, dim=-1)

        # Get top 5 tokens for original probs
        num = 7
        top_original_probs, top_original_indices = torch.topk(original_probs, num, dim=-1)
        top_tokens = [llm.tokenizer.decode([idx.item()]) for idx in top_original_indices[0, -1]]

        print(f"Original Probs - Top 5 Tokens for Sentence {sentence_idx + 1}:")
        for i, (prob, idx) in enumerate(zip(top_original_probs[0, -1], top_original_indices[0, -1])):
            token = llm.tokenizer.decode([idx.item()])  # Ensure the index is a scalar
            print(f"{i + 1}: Token: {token}, Prob: {prob.item()}")
        print("\n" + "=" * 40 + "\n")

        # Loop over different noise levels
        for index, noise_level in enumerate(noise_levels):
            # Add Gaussian noise to the first_hidden_states
            noised_first_hidden_states = add_gaussian_noise(first_hidden_states, std=noise_level)

            # Inject the noised hidden states back into the model
            llm.inject_hidden_states(noised_first_hidden_states)

            # Get the logits after injecting the noised hidden states
            batch_size = noised_first_hidden_states.shape[0]
            token_num = noised_first_hidden_states.shape[1]
            llm_outputs = llm.get_output_by_using_dummy(token_num=token_num, batch_size=batch_size)
            noised_logits = llm_outputs.logits

            # Convert logits to probabilities using softmax
            noised_probs = F.softmax(noised_logits, dim=-1)

            # Calculate MSE between original and noised hidden states
            mse = F.mse_loss(first_hidden_states, noised_first_hidden_states).item()
            mse_values.append(mse)

            if index % 2 == 0:
                plot_predictions(sentence_idx, noised_probs, mse, num)

            # Calculate KL divergence between the original and noised logits
            kl_div = F.kl_div(original_probs.log(), noised_probs, reduction='batchmean').item()
            kl_values.append(kl_div)

        mse_all_sentences.append(mse_values)
        kl_all_sentences.append(kl_values)

        # Plot the results
    plt.figure(figsize=(8, 6))

    for i, sentence in enumerate(sentences):
        plt.plot(mse_all_sentences[i], kl_all_sentences[i], marker='o', label=sentences[i])

    plt.xlabel('MSE (Original vs Noised Hidden States)')
    plt.ylabel('KL Divergence (Logits Probability Distributions)')
    plt.title(f'MSE vs KL-divergence for Different Noise Levels. A random vector KL is {random_kl}')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.close()
