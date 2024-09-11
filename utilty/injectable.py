import inspect
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

    def forward(self, *args, **kwargs):
        """
        Forward pass for the wrapped layer. Injects hidden states if the injection is active.
        :param *args: Positional arguments.
        :param **kwargs: Keyword arguments.
        """
        # Get the argument names of the layer's forward method
        arg_names = self._get_arg_names()

        # If injection is enabled, and we have an injected hidden state, prepare to replace it
        if self.injection_state and self.injected_hidden_state is not None:
            hidden_states = self.injected_hidden_state
        else:
            return self.layer(*args, **kwargs)

        # Look for 'hidden_states' in both *args and **kwargs and replace it with the injected hidden state
        new_args = list(args)  # Convert args to a list to modify
        for i, arg_name in enumerate(arg_names):
            if arg_name == 'hidden_states':
                # Replace in *args if 'hidden_states' is in positional arguments
                if i < len(new_args):
                    new_args[i] = hidden_states
                    break
        new_args = tuple(new_args)

        # If 'hidden_states' is in **kwargs, replace it there as well
        if 'hidden_states' in kwargs:
            kwargs['hidden_states'] = hidden_states

        # Call the layer's forward method with the modified arguments
        return self.layer(*new_args, **kwargs)

    def _get_arg_names(self):
        # Use inspect to get the function signature and parameter names for the layer's forward method
        signature = inspect.signature(self.layer.forward)
        return [param.name for param in signature.parameters.values()]
