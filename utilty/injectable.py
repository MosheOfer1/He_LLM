from abc import ABC, abstractmethod
import torch


class Injectable(ABC):
    @abstractmethod
    def inject_hidden_states(self, layer_num, hidden_states: torch.Tensor):
        """
        Abstract method to inject hidden states into the model.

        :param hidden_states: The hidden states tensor to be injected.
        """
        pass

    @abstractmethod
    def get_output_using_dummy(self, token_num: int):
        """
        Abstract method to retrieve the output after injection.

        :return: The output tensor or processed result.
        """
        pass