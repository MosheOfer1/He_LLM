import torch.nn as nn


class CustomLayerWrapper(nn.Module):
    def __init__(self, layer, hidden_states):
        super().__init__()
        self.layer = layer
        self.hs = hidden_states  # The injected hidden state layer

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None,
                past_key_value=None, output_attentions=None, use_cache=None):
        # Apply modifications to hidden_states here

        # Pass modified_hidden_states to the original layer
        return self.layer(self.hs, attention_mask, layer_head_mask,
                          past_key_value, output_attentions, use_cache)
