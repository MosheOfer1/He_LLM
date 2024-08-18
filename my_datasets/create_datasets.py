import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, OPTForCausalLM

from llm.llm_integration import LLMIntegration
from translation.translator import Translator


def create_transformer1_dataset(translator: Translator, llm: LLMIntegration):
    """
    Create a dataset for training Transformer1.

    :param translator: The translator instance used to generate hidden states.
    :param llm: The LLM instance used to generate hidden states.
    :return: A TensorDataset containing input-output pairs for Transformer1.
    """
    hebrew_sentences = [
        "שלום, איך אתה?",
        "אני אוהב לקרוא ספרים.",
        "מזג האוויר היום מאוד נחמד.",
        "הכלב שלי אוהב לשחק בפארק.",
        "האם אתה יכול לעזור לי בבקשה?",
        "אני הולך לסופרמרקט לקנות כמה דברים.",
        "אני עובד על פרויקט חדש בעבודה.",
        "הילדים משחקים בחוץ בגינה.",
        "מה אתה חושב על הסרט שראינו אתמול?",
        "אני רוצה ללמוד לנגן בגיטרה.",
        "המשפחה שלי מתכננת טיול לשבוע הבא.",
        "אני מנסה להכין ארוחת ערב מיוחדת.",
        "האם אתה אוהב את העיר הזאת?",
        "אני צריך לקנות מתנה לחבר שלי.",
        "אני מתאמן כל בוקר בחדר הכושר.",
        "מה התוכניות שלך לסוף השבוע?",
        "אני מאוד עייף אחרי יום ארוך בעבודה.",
        "האוכל במסעדה היה מאוד טעים.",
        "אני מתכנן ללמוד קורס חדש באוניברסיטה.",
        "יש לי פגישה חשובה מחר בבוקר."
    ]

    llm_tokenizer = AutoTokenizer.from_pretrained(llm.model_name)
    llm_model = OPTForCausalLM.from_pretrained(llm.model_name)

    input_hidden_states_list = []
    target_hidden_states_list = []

    for sentence in hebrew_sentences:
        # Step 1: Get the last hidden state from the first translator model
        with torch.no_grad():
            input_hidden_states = Translator.text_to_hidden_states(
                text=sentence,
                tokenizer=translator.source_to_target_tokenizer,
                model=translator.source_to_target_model,
                layer_num=-1,
                from_encoder=False
            )

        # Step 2: Translate the sentence
        translated_text = translator.translate(
            from_first=True,
            text=sentence
        )

        # Step 3: Pass the English translation through the LLM and get the first hidden state
        with torch.no_grad():
            target_hidden_states = llm.text_to_hidden_states(
                tokenizer=llm_tokenizer,
                model=llm_model,
                text=translated_text,
                layer_num=0
            )

        input_hidden_states_list.append(input_hidden_states)
        target_hidden_states_list.append(target_hidden_states)

    # Convert lists to tensors
    input_hidden_states_tensor = torch.cat(input_hidden_states_list, dim=0)
    target_hidden_states_tensor = torch.cat(target_hidden_states_list, dim=0)

    return TensorDataset(input_hidden_states_tensor, target_hidden_states_tensor)


def create_transformer2_dataset(translator, llm):
    """
    Create a dataset for training Transformer2.

    :param translator: The translator instance used to generate hidden states.
    :param llm: The LLM instance used to generate hidden states.
    :return: A TensorDataset containing input-output pairs for Transformer2.
    """
    hebrew_sentences = load_hebrew_sentences(file_name)

    input_hidden_states_list = []
    target_hidden_states_list = []

    for sentence in hebrew_sentences:
        # Step 1: Pass the English translation through the LLM and get the last hidden state
        english_translation = translator.translate_to_target(sentence)
        with torch.no_grad():
            input_hidden_states = llm.process_text_input(english_translation)

        # Step 2: Translate English back to Hebrew and get the first hidden state
        target_hidden_states = translator.translate_hidden_to_source(input_hidden_states)

        input_hidden_states_list.append(input_hidden_states)
        target_hidden_states_list.append(target_hidden_states)

    # Convert lists to tensors
    input_hidden_states_tensor = torch.cat(input_hidden_states_list, dim=0)
    target_hidden_states_tensor = torch.cat(target_hidden_states_list, dim=0)

    return TensorDataset(input_hidden_states_tensor, target_hidden_states_tensor)
