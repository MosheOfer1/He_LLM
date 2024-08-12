import csv
import mlflow
from facade_pipeline import Pipeline
from llm.llm_integration import LLMIntegration
from translation.helsinki_translator import HelsinkiTranslator


def load_dataset(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        sentences = [row[0] for row in reader]
    return sentences


def predict_next_word_straight_llm(input_sentence: str) -> str:
    llm = LLMIntegration()
    hidden_states = llm.process_text_input_to_logits(input_sentence)
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

    translator = HelsinkiTranslator()
    llm_integration = LLMIntegration()

    # Initialize pipelines with different configurations
    pipeline_2 = Pipeline(translator=translator, llm=llm_integration,
                          use_transformer_1=False, use_transformer_2=False)
    pipeline_3 = Pipeline(translator=translator, llm=llm_integration,
                          use_transformer_1=True, use_transformer_2=False)
    pipeline_4 = Pipeline(translator=translator, llm=llm_integration,
                          use_transformer_1=False, use_transformer_2=True)
    pipeline_5 = Pipeline(translator=translator, llm=llm_integration,
                          use_transformer_1=True, use_transformer_2=True)

    # Log experiment details with MLflow
    mlflow.log_param("dataset_size", len(sentences))

    for input_sentence in input_sentences:
        predictions["Straight Hebrew to LLM"].append(predict_next_word_straight_llm(input_sentence))
        predictions["Without any transforms"].append(predict_next_word(pipeline_2, input_sentence))
        predictions["Just the first transformer"].append(predict_next_word(pipeline_3, input_sentence))
        predictions["Just the second transformer"].append(predict_next_word(pipeline_4, input_sentence))
        predictions["Both transformers"].append(predict_next_word(pipeline_5, input_sentence))

    for test_case in predictions:
        score = calculate_score(predictions[test_case], actual_last_words)
        scores[test_case] = score
        mlflow.log_metric(f"{test_case}_score", score)

    for test_case, score in scores.items():
        print(f"{test_case}: {score:.2f}% correct")


if __name__ == "__main__":
    mlflow.start_run()  # Start a new MLflow run
    try:
        sentences = load_dataset('../datasets/hebrew_sentences.csv')
        run_tests(sentences)
    finally:
        mlflow.end_run()  # End the MLflow run
