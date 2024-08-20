from input_handler.input_handler import InputHandler
from facade_pipeline import Pipeline
from llm.llm_integration import LLMWrapper
from translation.helsinki_translator import HelsinkiTranslator


def main():
    # Get user input
    input_handler = InputHandler()
    user_input = input_handler.get_input()

    # Initialize the translator and LLMIntegration
    translator = HelsinkiTranslator()
    llm_integration = LLMWrapper()

    pipeline = Pipeline(
        translator=translator,
        llm=llm_integration,
        use_transformer_1=True,
        use_transformer_2=True
    )

    # Process the input through the pipeline
    processed_output = pipeline.process_text(user_input)

    # Display the output to the user
    input_handler.display_output(processed_output)


if __name__ == "__main__":
    main()
