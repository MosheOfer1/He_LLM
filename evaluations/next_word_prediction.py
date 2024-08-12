import csv
from facade_pipeline import Pipeline
from llm.llm_integration import LLMIntegration


def load_dataset(filepath: str):
    """
    Load the dataset of Hebrew sentences.

    :param filepath: Path to the CSV file containing Hebrew sentences.
    :return: A list of Hebrew sentences.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        sentences = [row[0] for row in reader]
    return sentences


def predict_next_word_straight_llm(input_sentence: str) -> str:
    llm = LLMIntegration()
    hidden_states = llm.process_text_input(input_sentence)
    next_word = llm.tokenizer.decode(hidden_states[0].argmax(-1))
    return next_word.split()[-1]


def predict_next_word(pipeline: Pipeline, input_sentence: str) -> str:
    result = pipeline.process_text(input_sentence)
    next_word = result.split()[-1]
    return next_word


def calculate_score(predictions, actual_words):
    correct = sum(1 for pred, actual in zip(predictions, actual_words) if pred == actual)
    return correct / len(actual_words) * 100


def run_tests(sentences):
    scores = {}
    predictions = {
        "Straight Hebrew to LLM": [],
        "Without any transforms": [],
        "Just the first transformer": [],
        "Just the second transformer": [],
        "Both transformers": []
    }

    actual_last_words = [sentence.split()[-1] for sentence in sentences]
    input_sentences = [" ".join(sentence.split()[:-1]) for sentence in sentences]

    # Test each configuration
    for input_sentence in input_sentences:
        predictions["Straight Hebrew to LLM"].append(predict_next_word_straight_llm(input_sentence))

        pipeline_2 = Pipeline(use_transformer_1=False, use_transformer_2=False)
        predictions["Without any transforms"].append(predict_next_word(pipeline_2, input_sentence))

        pipeline_3 = Pipeline(use_transformer_1=True, use_transformer_2=False)
        predictions["Just the first transformer"].append(predict_next_word(pipeline_3, input_sentence))

        pipeline_4 = Pipeline(use_transformer_1=False, use_transformer_2=True)
        predictions["Just the second transformer"].append(predict_next_word(pipeline_4, input_sentence))

        pipeline_5 = Pipeline(use_transformer_1=True, use_transformer_2=True)
        predictions["Both transformers"].append(predict_next_word(pipeline_5, input_sentence))

    # Calculate scores
    for test_case in predictions:
        scores[test_case] = calculate_score(predictions[test_case], actual_last_words)

    # Display the results
    for test_case, score in scores.items():
        print(f"{test_case}: {score:.2f}% correct")


if __name__ == "__main__":
    # Load the dataset of Hebrew sentences
    sentences = load_dataset('../datasets/hebrew_sentences.csv')
    run_tests(sentences)
