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


def calculate_kl_with_random_injection(llm, sentence, random_std=0.1, num_random_vectors=50):
    kl_values = []

    for _ in range(num_random_vectors):
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
        kl_values.append(kl_div)

    # Return the average KL-divergence of 50 random vectors
    return np.mean(kl_values)


# Function to add Gaussian noise
def add_gaussian_noise(tensor, mean=0.0, std=0.1):
    noise = torch.normal(mean=mean, std=std, size=tensor.shape).to(tensor.device)
    return tensor + noise


# New function for calculating cosine similarity
def calculate_cuisine_similarity(original_hidden_states, noised_hidden_states):
    similarity = F.cosine_similarity(original_hidden_states, noised_hidden_states).mean().item()
    return 1 - similarity


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Im working with: {device}")

    model_name = 'facebook/opt-350m'

    llm = OptLLM(
        model_name=model_name,
        device=device
    )

    noise_levels = np.linspace(0.01, 0.15, 10)

    # To store the sum of metric values (MSE and Cosine Similarity) for each noise level
    mse_sum = np.zeros(len(noise_levels))
    cosine_sim_sum = np.zeros(len(noise_levels))

    kl_sum = np.zeros(len(noise_levels))  # To store the sum of KL-divergence values for each noise level

    for sentence_idx, sentence in enumerate(sentences):
        print(sentence_idx)
        with llm.injection_state():
            with torch.no_grad():
                outputs = llm.process_text_input_to_outputs(sentence)
                first_hidden_states = outputs.hidden_states[0]
                logits = outputs.logits[:, -1, :]

        for index, noise_level in enumerate(noise_levels):
            noised_first_hidden_states = add_gaussian_noise(first_hidden_states, std=noise_level)

            llm.inject_hidden_states(noised_first_hidden_states)

            batch_size = noised_first_hidden_states.shape[0]
            token_num = noised_first_hidden_states.shape[1]
            llm_outputs = llm.get_output_by_using_dummy(token_num=token_num, batch_size=batch_size)
            noised_logits = llm_outputs.logits[:, -1, :]

            # Calculate MSE
            mse_value = F.mse_loss(first_hidden_states, noised_first_hidden_states).item()
            mse_sum[index] += mse_value  # Add the value to the sum for averaging later

            # Calculate Cosine Similarity (Cuisine Similarity)
            cosine_sim_value = calculate_cuisine_similarity(first_hidden_states, noised_first_hidden_states)
            cosine_sim_sum[index] += cosine_sim_value  # Add the value to the sum for averaging later

            # Calculate KL-divergence
            noised_probs = F.softmax(noised_logits, dim=-1)
            original_probs = F.softmax(logits, dim=-1)
            kl_div = F.kl_div(noised_probs.log(), original_probs, reduction='batchmean').item()
            kl_sum[index] += kl_div  # Add the value to the sum for averaging later

    # Calculate the average MSE, Cosine Similarity, and KL-divergence across all sentences for each noise level
    mse_avg = mse_sum / len(sentences)
    cosine_sim_avg = cosine_sim_sum / len(sentences)
    kl_avg = kl_sum / len(sentences)

    # Calculate the average KL-divergence for 50 random vectors
    random_kl_avg = calculate_kl_with_random_injection(llm=llm, sentence=sentences[0])

    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot KL-divergence vs Cosine Similarity on the left x-axis
    ax1.plot(cosine_sim_avg, kl_avg, marker='o', color='green', label='KL-divergence vs Cosine Similarity')
    ax1.set_xlabel('Cosine Similarity', color='green')
    ax1.set_ylabel('KL-divergence')
    ax1.tick_params(axis='x', labelcolor='green')
    ax1.grid(True)

    # Create another x-axis on the right for MSE
    ax2 = ax1.twiny()
    ax2.plot(mse_avg, kl_avg, marker='s', color='blue', label='KL-divergence vs MSE')
    ax2.set_xlabel('MSE', color='blue')
    ax2.tick_params(axis='x', labelcolor='blue')

    # Add a horizontal line for the average KL-divergence of 50 random vectors
    ax1.axhline(y=random_kl_avg, color='red', linestyle='--', label=f'Average KL (Random Vectors) = {random_kl_avg:.2f}')

    # Combine legends from both axes
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.title('KL-divergence vs Cosine Similarity and MSE (Averaged over all sentences)')
    plt.show()
