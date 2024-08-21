from transformers import AutoTokenizer, OPTForCausalLM

from custom_transformers.transformer_1 import Transformer1
from llm.llm_integration import LLMWrapper
from my_datasets.create_datasets import create_transformer1_dataset
from translation.helsinki_translator import HelsinkiTranslator

translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
translator = HelsinkiTranslator(translator1_model_name, translator2_model_name)

llm_model_name = "facebook/opt-125m"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = OPTForCausalLM.from_pretrained(llm_model_name)
llm = LLMWrapper(llm_model_name, llm_tokenizer, llm_model)

transformer1 = Transformer1(translator, llm)

file_path = '../my_datasets/'
train_ds, test_ds = create_transformer1_dataset(
    translator,
    llm,
    file_path,
    dataset_name="SVLM_Hebrew_Wikipedia_Corpus.txt",
    sentence_num=300
)

transformer1.train_model(train_ds, test_ds)

