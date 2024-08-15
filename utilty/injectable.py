from abc import ABC, abstractmethod


class Injectable(ABC):
    @abstractmethod
    def inject_hidden_states(self, hidden_states):
        """
        Abstract method to inject hidden states into the model.

        :param hidden_states: The hidden states tensor to be injected.
        """
        pass

    @abstractmethod
    def get_output(self):
        """
        Abstract method to retrieve the output after injection.

        :return: The output tensor or processed result.
        """
        pass

    @abstractmethod
    def decode_output(self, output):
        """
        Abstract method to decode the output into a human-readable format.

        :param output: The output tensor to decode.
        :return: The decoded string.
        """
        pass
