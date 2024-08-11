class InputHandler:
    def __init__(self):
        # Initialize any required variables here if needed
        pass

    def get_input(self) -> str:
        """
        Prompt the user for Hebrew input and return it as a string.
        """
        return input("Enter your Hebrew text: ")

    def display_output(self, text: str):
        """
        Display the translated output to the user.
        """
        print("Translated Output:", text)
