from abc import ABC, abstractmethod
import torch


class TransformerStrategy(ABC):
    @abstractmethod
    def transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Transform the hidden states and return the modified hidden states.

        :param hidden_states: A tensor containing the hidden states from the model.
        :return: A tensor containing the transformed hidden states.
        """
        pass
