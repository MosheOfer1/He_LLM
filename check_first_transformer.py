import torch.nn as nn
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.opt_llm import OptLLM
from custom_transformers.transformer_1 import Transformer1
from translation.helsinki_translator import HelsinkiTranslator

translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
translator = HelsinkiTranslator(translator1_model_name,
                                translator2_model_name)

llm_model_name = "facebook/opt-125m"
llm = OptLLM(llm_model_name)


def load_and_evaluate_model(model_name):
    """Test loading a model by name and running a forward pass in evaluation mode."""

    # Load the model
    loaded_model = Transformer1.load_model(
        model_name=model_name,
        translator=translator,
        llm=llm
    )

    sentence = input("Enter Hebrew sentence: ")

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
        output = loaded_model(input_hidden_states)

    loss_fct = nn.MSELoss()
    loss = loss_fct(output, target_hidden_states)
    print(loss)

    llm.inject_hidden_states(output)
    outputs = llm.get_output_by_using_dummy(output.shape[1])
    print(llm.decode_logits(outputs.logits))

    llm.inject_hidden_states(target_hidden_states)
    outputs = llm.get_output_by_using_dummy(target_hidden_states.shape[1])
    print(llm.decode_logits(outputs.logits))


if __name__ == '__main__':
    model_name = input("Enter model name: ")
    load_and_evaluate_model(model_name)
