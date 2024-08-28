import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Transformers
from custom_transformers.transformer import Transformer
# Translators
from translation.helsinki_translator import HelsinkiTranslator
from llm.facebook_llm import FacebookLLM


class MyCustomModel(nn.Module):

    def __init__(self,
                 src_to_target_translator_model_name,
                 target_to_src_translator_model_name,
                 llm_model_name):
        super(MyCustomModel, self).__init__()

        # Custom Translator
        self.translator = HelsinkiTranslator(src_to_target_translator_model_name,
                                             target_to_src_translator_model_name)
        # Custom LLM
        self.llm = FacebookLLM(llm_model_name)

        self.transformer = Transformer(translator=self.translator,
                                       llm=self.llm)

        # Freeze Translator1 parameters
        self.translator.set_requires_grad(False)

        # Freeze LLM parameters
        self.llm.set_requires_grad(False)

    def forward(self, text: str) -> torch.Tensor:
        # Get hidden states from text
        translator_last_hs = self.translator.text_to_hidden_states(text, -1, self.translator.src_to_target_tokenizer,
                                                                   self.translator.src_to_target_model, False)

        # Transform to llm first hidden states
        transformed_to_llm_hs = self.transformer.transformer1.forward(translator_last_hs)

        # Inject the new hidden states to the llm first layer
        self.llm.inject_hidden_states(transformed_to_llm_hs)

        # Input dummy text but the it is ignored and uses the injected 
        llm_outputs = self.llm.get_output_by_using_dummy(transformed_to_llm_hs.shape[1])

        # Extract the last hidden states
        # llm_last_hidden_state = llm_outputs.hidden_states[-1]
        llm_last_hidden_state = llm_outputs.hidden_states[-1][:,-1,:].unsqueeze(0)

        # Transform to translator first hidden states
        transformed_to_translator_hs = self.transformer.transformer2.forward(llm_last_hidden_state)

        # Inject the new hidden states to translator2 first layer
        self.translator.inject_hidden_states(transformed_to_translator_hs)
                
        # Input dummy text but the it is ignored and uses the injected 
        translator_outputs = self.translator.get_output_by_using_dummy(transformed_to_translator_hs.shape[1])

        return translator_outputs.logits
