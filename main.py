from input_handler.input_handler import InputHandler
from facade_pipeline import Pipeline


def main():
    # Step 1: Get user input
    input_handler = InputHandler()
    user_input = input_handler.get_input()

    # Step 2: Configure custom_transformers based on user choice
    # For this example, we're hardcoding the choices. In a real-world application,
    # these could be obtained from user input or a config file.
    use_transformer_1 = True  # Set to True if you want to apply Transformer 1
    use_transformer_2 = True  # Set to True if you want to apply Transformer 2

    # Step 3: Create the facade with the selected custom_transformers
    translator = Pipeline(use_transformer_1, use_transformer_2)

    # Step 4: Process the input through the translation pipeline
    translated_output = translator.process_text(user_input)

    # Step 5: Display the output to the user
    input_handler.display_output(translated_output)


if __name__ == "__main__":
    main()
