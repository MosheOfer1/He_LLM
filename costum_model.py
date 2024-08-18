from transformers import OPTForCausalLM, OPTConfig
import torch
import torch.nn as nn
from typing import Union, Tuple

from transformers.modeling_outputs import CausalLMOutputWithPast


class MyCustomModel(OPTForCausalLM):
    config_class = OPTConfig

    def __init__(self, config, additional_layer_size=512):
        super(MyCustomModel, self).__init__(config)
        self.additional_layer = nn.Linear(config.hidden_size, additional_layer_size)
        self.output_layer = nn.Linear(additional_layer_size, config.vocab_size)

    def forward(self, *args, **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        # Ensure hidden states are returned
        kwargs['output_hidden_states'] = True

        # Call the original forward method of OPTForCausalLM
        outputs = super(MyCustomModel, self).forward(*args, **kwargs)

        # Get the hidden states from the original output
        hidden_states = outputs.hidden_states[-1]  # Take the last hidden state

        # Apply additional custom layers
        x = self.additional_layer(hidden_states)
        x = torch.relu(x)
        logits = self.output_layer(x)

        # If the output is not a dict, return a tuple
        if not kwargs.get('return_dict', False):
            return (logits,) + outputs[1:]

        # Return the logits in the expected output format
        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Example usage with the generate function
model = MyCustomModel.from_pretrained('facebook/opt-125m')

input_ids = torch.tensor([[1, 2, 3, 4]])
generated_output = model.generate(input_ids, max_length=10)
print(generated_output)
