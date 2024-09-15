import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.opt_llm import OptLLM

sentences = [
    "Artificial intelligence will shape the future of",
    "Quantum computing holds the key to solving complex",
    "The mysteries of the deep ocean are yet to be",
    "Exploring Mars could unlock the secrets of life beyond",
    "Advances in biotechnology will revolutionize the field of",
    "Self-driving cars are becoming more common on the",
    "The development of renewable energy sources is critical for",
    "Climate change is one of the biggest challenges facing",
    "Blockchain technology has the potential to transform the world of",
    "The internet of things connects everyday devices to the",
    "Machine learning is being applied to improve medical",
    "Space exploration will help humanity better understand the universe and",
    "Virtual reality is changing the way we experience",
    "5G technology promises to significantly enhance mobile",
    "The rise of e-commerce has transformed the way people",
    "Social media platforms have altered the way we communicate and share",
    "Artificial neural networks are inspired by the structure of the human",
    "Cybersecurity is crucial for protecting sensitive",
    "Robots are increasingly being used in industries such as manufacturing and",
    "Electric vehicles are gaining popularity as an alternative to",
    "Drones are being used for everything from package delivery to",
    "The rise of smart cities is reshaping urban",
    "Quantum mechanics challenges our understanding of the nature of",
    "Artificial intelligence is making significant strides in natural language",
    "Genetic engineering allows scientists to modify the DNA of",
    "Augmented reality overlays digital information onto the physical",
    "The development of fusion energy could provide a nearly limitless source of",
    "Wearable technology is becoming more integrated into everyday",
    "The use of AI in healthcare is expected to improve patient",
    "Big data analytics helps companies make better",
    "The sharing economy has changed the way people use",
    "The future of transportation may include flying",
    "Renewable energy sources like wind and solar power are becoming more",
    "Cryptocurrency is a digital form of",
    "The rise of automation is changing the nature of",
    "The study of black holes gives us insights into the fabric of",
    "Nanotechnology allows us to manipulate matter at the scale of",
    "The human genome project mapped the entire sequence of human",
    "The rise of remote work has changed the way companies",
    "The field of robotics is advancing to create more sophisticated",
    "The search for extraterrestrial life continues to fascinate",
    "The development of autonomous ships will revolutionize global",
    "Personalized medicine aims to tailor treatments based on an individual's",
    "The exploration of the deep ocean remains one of the last great frontiers of",
    "Biodegradable materials are being developed to reduce plastic",
    "The rise of digital currencies could disrupt traditional financial",
    "Artificial intelligence is being used to create more realistic video",
    "The integration of AI into education could personalize learning for",
    "Smart home devices are becoming more common in households around the",
    "The future of artificial intelligence holds both promise and"
]


def calculate_kl_with_random_injection(llm, sentence, random_std=0.1):
    with llm.injection_state():
        with torch.no_grad():
            outputs = llm.process_text_input_to_outputs(sentence)
            original_logits = outputs.logits[:, -1, :]
            original_probs = F.softmax(original_logits, dim=-1)
            original_hidden_states = outputs.hidden_states[0]

    random_vector = torch.normal(mean=0.0, std=random_std, size=original_hidden_states.shape).to(
        original_hidden_states.device)

    llm.inject_hidden_states(random_vector)

    batch_size = random_vector.shape[0]
    token_num = random_vector.shape[1]

    with torch.no_grad():
        random_injected_outputs = llm.get_output_by_using_dummy(token_num=token_num, batch_size=batch_size)
        random_injected_logits = random_injected_outputs.logits[:, -1, :]

    random_injected_probs = F.softmax(random_injected_logits, dim=-1)

    kl_div = F.kl_div(random_injected_probs.log(), original_probs, reduction='batchmean').item()

    return kl_div


# Function to add Gaussian noise
def add_gaussian_noise(tensor, mean=0.0, std=0.1):
    noise = torch.normal(mean=mean, std=std, size=tensor.shape).to(tensor.device)
    return tensor + noise


# New function for calculating cuisine similarity
def calculate_cuisine_similarity(original_hidden_states, noised_hidden_states):
    similarity = F.cosine_similarity(original_hidden_states, noised_hidden_states).mean().item()
    return 1 - similarity


def plot_predictions(sentence_idx, noised_probs, metric_value, num, metric_name):
    top_noised_probs, top_noised_indices = torch.topk(noised_probs, num, dim=-1)
    top_noised_tokens = [llm.tokenizer.decode([idx.item()]) for idx in top_noised_indices[0, -1]]

    print(f"Noised Probs - Top {num} Tokens (Noise {noise_level:.2f}) for Sentence {sentence_idx + 1}:")
    for i, (prob, idx) in enumerate(zip(top_noised_probs[0, -1], top_noised_indices[0, -1])):
        token = llm.tokenizer.decode([idx.item()])
        print(f"{i + 1}: Token: {token}, Prob: {prob.item()}")

    print("\n" + "=" * 40 + "\n")

    plot_prob_distribution(
        top_original_probs[0, -1],
        top_noised_probs[0, -1],
        list(zip(top_tokens, top_noised_tokens)),
        noise_level,
        metric_value,
        num,
        sentence_idx,
        metric_name
    )


def plot_prob_distribution(original_probs, noised_probs, tokens, noise_level, metric_value, num, sentence_idx,
                           metric_name):
    n_tokens = len(tokens)

    x = np.arange(n_tokens)
    width = 0.35

    original_probs_np = original_probs.detach().cpu().numpy()
    noised_probs_np = noised_probs.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width / 2, original_probs_np, width, label='Original Probs', color='blue')
    ax.bar(x + width / 2, noised_probs_np, width,
           label=f'Noised Probs (Noise={noise_level:.2f}) ({metric_name}={metric_value:.2f})', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=10, ha='center', fontsize=10)

    ax.set_ylabel('Probability')
    ax.set_title(
        f'Sentence {sentence_idx + 1} - Top {num} Token Probability Distribution (Noise Level: {noise_level:.2f})')
    ax.legend()
    image_name = f'Noise={noise_level:.2f} {metric_name}={metric_value:.2f} Sentence {sentence_idx + 1}'.replace('.',
                                                                                                                 '_') + '.png'
    plt.savefig(f'../images/' + image_name)
    plt.close()


if __name__ == '__main__':
    # Prompt the user to select either 'MSE' or 'Cuisine Similarity'
    metric_choice = input("Choose the metric: 'MSE' or 'Cuisine Similarity': ").strip().lower()

    if metric_choice not in ['mse', 'cuisine similarity']:
        print("Invalid choice! Defaulting to 'MSE'.")
        metric_choice = 'mse'

    metric_name = 'MSE' if metric_choice == 'mse' else 'Cuisine Similarity'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Im working with: {device}")

    model_name = 'facebook/opt-350m'

    llm = OptLLM(
        model_name=model_name,
        device=device
    )

    random_kl = calculate_kl_with_random_injection(llm=llm, sentence=sentences[0])
    print("KL-divergence with a random vector is: ", random_kl)

    metric_all_sentences = []
    kl_all_sentences = []
    noise_levels = np.linspace(0.01, 0.1, 12)

    # To store the sum of metric values for each noise level
    metric_sum = np.zeros(len(noise_levels))
    # To store the sum of KL-divergence values for each noise level
    kl_sum = np.zeros(len(noise_levels))

    for sentence_idx, sentence in enumerate(sentences):
        metric_values = []
        kl_values = []

        with llm.injection_state():
            with torch.no_grad():
                outputs = llm.process_text_input_to_outputs(sentence)
                first_hidden_states = outputs.hidden_states[0]
                logits = outputs.logits[:, -1, :]

        original_probs = F.softmax(logits, dim=-1)

        num = 7
        top_original_probs, top_original_indices = torch.topk(original_probs, num, dim=-1)
        top_tokens = [llm.tokenizer.decode([idx.item()]) for idx in top_original_indices[0]]

        print(f"Original Probs - Top {num} Tokens for Sentence {sentence_idx + 1}:")
        for i, (prob, idx) in enumerate(zip(top_original_probs[0], top_original_indices[0])):
            token = llm.tokenizer.decode([idx.item()])
            print(f"{i + 1}: Token: {token}, Prob: {prob.item()}")
        print("\n" + "=" * 40 + "\n")

        for index, noise_level in enumerate(noise_levels):
            noised_first_hidden_states = add_gaussian_noise(first_hidden_states, std=noise_level)

            llm.inject_hidden_states(noised_first_hidden_states)

            batch_size = noised_first_hidden_states.shape[0]
            token_num = noised_first_hidden_states.shape[1]
            llm_outputs = llm.get_output_by_using_dummy(token_num=token_num, batch_size=batch_size)
            noised_logits = llm_outputs.logits[:, -1, :]

            noised_probs = F.softmax(noised_logits, dim=-1)

            # Calculate either MSE or Cuisine Similarity based on the user's choice
            if metric_choice == 'mse':
                metric_value = F.mse_loss(first_hidden_states, noised_first_hidden_states).item()
            else:
                metric_value = calculate_cuisine_similarity(first_hidden_states, noised_first_hidden_states)

            metric_values.append(metric_value)

            # To create images of the bars uncommitted
            # plot_predictions(sentence_idx, noised_probs, metric_value, num, metric_name)

            metric_sum[index] += metric_value  # Add the value to the sum for averaging later

            kl_div = F.kl_div(noised_probs.log(), original_probs, reduction='batchmean').item()
            kl_values.append(kl_div)

            kl_sum[index] += kl_div  # Add the value to the sum for averaging later

        metric_all_sentences.append(metric_values)
        kl_all_sentences.append(kl_values)

    # Calculate the average metric across all sentences for each noise level
    metric_avg = metric_sum / len(sentences)
    kl_avg = kl_sum / len(sentences)

    plt.figure(figsize=(8, 6))

    for i, sentence in enumerate(sentences):
        plt.plot(metric_all_sentences[i], kl_all_sentences[i], marker='o', label=sentences[i])
        if i == 3:
            break

    # Plot the average metric line
    plt.plot(metric_avg, kl_avg, marker='x', linestyle='--', color='black', label=f'Average {metric_name}')

    plt.xlabel(f'{metric_name} (Original vs Noised Hidden States)')
    plt.ylabel('KL Divergence (Logits Probability Distributions)')
    plt.title(f'{metric_name} vs KL-divergence for Different Noise Levels.')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.close()
