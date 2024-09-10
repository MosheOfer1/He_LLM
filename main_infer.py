import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Im working with: {device}")


translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
llm_model_name = "facebook/opt-125m"

# # Specify the full file path for the model
model_path = "/home/ddn1/Documents/GitHub/He_LLM/output_dir_orel_moshe/customLLM_batches.pth"

# model = load_model(model_path=model_path,
#                    translator1_model_name=translator2_model_name,
#                    translator2_model_name=translator2_model_name,
#                    llm_model_name=llm_model_name,
#                    device=device)
