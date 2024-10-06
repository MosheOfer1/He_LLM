import numpy as np
import torch
import os
import sys
import torch.nn.functional as F

from matplotlib import pyplot as plt

from custom_transformers.custom_trainer_trans1 import calculate_KL_div

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.opt_llm import OptLLM
from custom_transformers.transformer_1 import Transformer1, logits_from_first_layer
from translation.helsinki_translator import HelsinkiTranslator

translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
translator = HelsinkiTranslator(translator1_model_name,
                                translator2_model_name)

llm_model_name = "facebook/opt-350m"
llm = OptLLM(llm_model_name)


def plot_top_k_predictions(output_logits, target_hidden_logits, k, llm, sentence_idx):
    """
    Plot the top k word predictions from both the output logits and the target hidden states logits.
    """
    # Get the final predictions (last time step) for both output and target hidden logits
    output_last_logits = output_logits[:, -1, :]
    target_last_logits = target_hidden_logits[:, -1, :]

    # Get the top k predictions from the output logits
    top_output_probs, top_output_indices = torch.topk(torch.softmax(output_last_logits, dim=-1), k, dim=-1)
    top_output_tokens = [llm.tokenizer._output_decode([idx.item()]) for idx in top_output_indices[0]]

    # Get the top k predictions from the target hidden logits
    top_target_probs, top_target_indices = torch.topk(torch.softmax(target_last_logits, dim=-1), k, dim=-1)
    top_target_tokens = [llm.tokenizer._output_decode([idx.item()]) for idx in top_target_indices[0]]

    print(f"Top {k} Predictions from Output Logits for Sentence {sentence_idx + 1}:")
    for i, (prob, token) in enumerate(zip(top_output_probs[0], top_output_tokens)):
        print(f"{i + 1}: Token: {token}, Prob: {prob.item()}")

    print(f"\nTop {k} Predictions from Target Hidden States for Sentence {sentence_idx + 1}:")
    for i, (prob, token) in enumerate(zip(top_target_probs[0], top_target_tokens)):
        print(f"{i + 1}: Token: {token}, Prob: {prob.item()}")

    # Plot the probability distributions for the top k tokens
    plot_prob_comparison(
        top_output_probs[0],
        top_target_probs[0],
        list(zip(top_output_tokens, top_target_tokens)),
        k,
        sentence_idx
    )


def plot_prob_comparison(output_probs, target_probs, tokens, k, sentence_idx):
    """
    Helper function to plot the probability distributions of the top k tokens from both outputs and targets.
    """
    n_tokens = len(tokens)

    x = np.arange(n_tokens)
    width = 0.35

    output_probs_np = output_probs.detach().cpu().numpy()
    target_probs_np = target_probs.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot for the output probabilities
    ax.bar(x - width / 2, output_probs_np, width, label='Output Probs', color='blue')
    # Bar plot for the target hidden state probabilities
    ax.bar(x + width / 2, target_probs_np, width, label='Target Probs', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels([f'({o}, {t})' for o, t in tokens], rotation=45, ha='right')

    ax.set_ylabel('Probability')
    ax.set_title(f'Sentence {sentence_idx + 1} - Top {k} Token Probability Distribution')
    ax.legend()
    plt.show()


def load_and_evaluate_model(model_name):
    """Test loading a model by name and running a forward pass in evaluation mode."""

    # Load the model
    loaded_model = Transformer1.load_model(
        model_name=model_name,
        translator=translator,
        llm=llm
    )

    sentence = "ירושלים היא עיר הבירה של "

    # Step 1: Get the last hidden state from the first translator model
    with torch.no_grad():
        outputs = translator.get_output(from_first=True, text=sentence)
    input_hidden_states = outputs.decoder_hidden_states[-1]  # Shape: (seq_len, hidden_dim)

    # Step 2: Translate the sentence
    translated_text = translator.decode_logits(
        tokenizer=translator.src_to_target_tokenizer,
        logits=outputs.logits
    )
    print(translated_text)

    # Step 3: Pass the English translation through the LLM and get the first hidden state
    with torch.no_grad():
        with llm.injection_state():
            target_hidden_states = llm.text_to_hidden_states(
                tokenizer=llm.tokenizer,
                model=llm.model,
                text=translated_text,
                layer_num=0  # Assuming this returns a tensor of shape (seq_len, hidden_dim)
            )

    # Perform a forward pass
    with torch.no_grad():
        transformer1_outputs = loaded_model(input_hidden_states)

    min_len = min(transformer1_outputs.shape[1], target_hidden_states.shape[1])
    loss = calculate_KL_div(
        llm=llm,
        outputs=transformer1_outputs[:, : min_len, :],
        label=target_hidden_states[:, : min_len, :]
    )

    print(f"The KL-div is: {loss}")

    cosine_sim = F.cosine_similarity(transformer1_outputs[:, : min_len, :], target_hidden_states[:, : min_len, :], dim=-1)
    loss = 1 - cosine_sim.mean()
    print(f"The cosine_sim is: {loss}")

    output_logits = logits_from_first_layer(llm, transformer1_outputs)
    target_hidden_logits = logits_from_first_layer(llm, target_hidden_states)
    plot_top_k_predictions(output_logits=output_logits, target_hidden_logits=target_hidden_logits, k=5, llm=llm, sentence_idx=0)


if __name__ == '__main__':
    model_name = "models/transformer_1_Helsinki-NLP_opus-mt-tc-big-he-en_to_facebook_opt-350m_KL.pth"
    load_and_evaluate_model(model_name)
