import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, OPTForCausalLM

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.llm_wrapper import LLMWrapper
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize the LLM model once
model_name = "facebook/opt-350m"
llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = OPTForCausalLM.from_pretrained(model_name)

llm = LLMWrapper(model_name,
                 llm_tokenizer,
                 llm_model)

# Create the main window
root = tk.Tk()
root.title("Token Prediction")

# Create a figure and axis to update the plot
fig, ax = plt.subplots(figsize=(10, 6))


def predict_and_plot():
    # Get the input text from the GUI
    text = entry.get()

    # Get the hidden state and logits
    llm.model.base_model.decoder.layers[llm.injected_layer_num].set_injection_state(False)
    last_layer = llm.text_to_hidden_states(
        tokenizer=llm.tokenizer,
        model=llm.model,
        text=text,
        layer_num=-1
    )
    llm.model.base_model.decoder.layers[llm.injected_layer_num].set_injection_state(True)
    last_hidden_state = last_layer[0, -1, :].unsqueeze(0).unsqueeze(0)

    logits = llm.model.lm_head(last_hidden_state)
    probs = torch.softmax(logits, dim=-1).squeeze()

    top_probs, top_indices = torch.topk(probs, 5)
    rest_prob = probs.sum() - top_probs.sum()
    print(top_indices)
    top_tokens = [llm.tokenizer.decode([idx.item()], clean_up_tokenization_spaces=True, skip_special_tokens=True) for
                  idx in top_indices]
    print(top_tokens)
    labels = top_tokens + ["Rest"]
    probabilities = top_probs.tolist() + [rest_prob.item()]

    # Clear the previous plot
    ax.clear()

    # Plot the probabilities
    bars = ax.bar(labels, probabilities, color='skyblue')
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Probability')
    ax.set_title(f'Top 5 Predicted Tokens for the next token after "{text}"')
    ax.set_ylim(0, 1)

    for bar, prob in zip(bars, probabilities):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.4f}', ha='center', va='bottom')

    # Redraw the updated plot
    fig.canvas.draw()


# Create an input field
entry = tk.Entry(root, width=50)
entry.pack(pady=20)

# Create a button to trigger the prediction
button = tk.Button(root, text="Predict Next Token", command=predict_and_plot)
button.pack(pady=20)

# Embed the plot in the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Start the GUI event loop
root.mainloop()
