from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Injectable(ABC):
    @abstractmethod
    def inject_hidden_states(self, injected_hidden_state: torch.Tensor):
        """
        Abstract method to inject hidden states into the model.

        :param injected_hidden_state: The injected hidden states
        """
        pass

    @abstractmethod
    def get_output_by_using_dummy(self, token_num):
        """
        Abstract method to retrieve the output after injection.

        :return: The output tensor or processed result.
        """
        pass


class CustomLayerWrapper(nn.Module):
    def __init__(self, layer, hidden_states):
        super().__init__()
        self.layer = layer
        self.injected_hidden_state = hidden_states  # The injected hidden state layer
        self.injection_state = True

    def set_injection_state(self, injection_state: bool):
        self.injection_state = injection_state

    def forward(self, hidden_states, *args, **kwargs):
        """
        Forward pass for the wrapped layer. Injects hidden states if the injection is active.
        :param hidden_states: The original hidden states.
        :param *args: Additional positional arguments.
        :param **kwargs: Additional keyword arguments.
        """
        # Inject the hidden states if injection is enabled
        if self.injection_state and self.injected_hidden_state is not None:
            hidden_states = self.injected_hidden_state

        # Pass the modified or original hidden states along with the additional arguments to the original layer
        return self.layer(hidden_states, *args, **kwargs)
