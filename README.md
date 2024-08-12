# Project Structure Overview

This document provides an overview of the project's structure, outlining the purpose of each directory and file. This should help in dividing the work effectively among developers.

## Directory and File Structure

```
/He_LLM/
│
├── facade_pipeline.py                # Orchestrates the overall processing pipeline
├── main.py                           # A basic use Example
├── requirements.txt                  # Specifies project dependencies
│
├── /custom_transformers/             # Contains custom transformers for processing hidden states
│   ├── __init__.py                  
│   ├── base_transformer.py           # Base class for transformers (inherits from nn.Module)
│   ├── transformer_strategy.py       # Base strategy interface for transformers
│   ├── transformer_1.py              # Implementation of the first transformer
│   └── transformer_2.py              # Implementation of the second transformer
│
├── /datasets/                        # Directory for storing datasets
│   └── hebrew_sentences.csv          # Example dataset of Hebrew sentences
│
├── /input_handler/                   # Manages user input and output
│   ├── __init__.py                  
│   └── input_handler.py             
│
├── /llm/                             # Handles integration with the LLM (e.g., OPT)
│   ├── __init__.py                  
│   └── llm_integration.py           
│
├── /tests/                           # Contains unit tests for the various components
│   ├── __init__.py                  
│   ├── test_input_handler.py        
│   ├── test_helsinki_translator.py  
│   ├── test_transformers.py         
│   ├── test_llm_integration.py      
│   └── test_facade.py               # Unit test for the pipeline (facade_pipeline.py)
│
├── /evaluations/
│   ├── next_word_prediction.py       # Script to evaluate next-word prediction accuracy
│
├── /translation/                     # Handles translation logic
    ├── __init__.py
    └── helsinki_translator.py       
```

## Detailed Directory and File Descriptions

### **1. `facade_pipeline.py`**
- **Purpose:** This file contains the `Pipeline` class, which orchestrates the entire processing flow. It coordinates the interaction between input handling, translation, transformers, and LLM integration.

### **2. `/custom_transformers/`**
- **Purpose:** This directory holds custom transformers that process hidden states between translation and LLM interaction.
  - **`base_transformer.py`**: Base class for transformers, handles model loading and training.
  - **`transformer_strategy.py`**: Defines the interface that all transformers must implement.
  - **`transformer_1.py` & `transformer_2.py`**: Specific implementations of transformers. Developers focusing on model transformation will work here.

### **3. `/llm/`**
- **Purpose:** Handles integration with the Language Model (LLM), such as OPT.
  - **`llm_integration.py`**: Manages the processing of text and hidden states through the LLM. Developers working on LLM-related logic will focus here.

### **4. `/input_handler/`**
- **Purpose:** This directory contains the logic for handling user input and output.
  - **`input_handler.py`**: Manages input collection and output display. Developers working on user interaction will focus here.

### **5. `/translation/`**
- **Purpose:** Contains modules related to translating Hebrew text to English and vice versa.
  - **`helsinki_translator.py`**: Implements translation using the Helsinki-NLP models. Developers focusing on translation logic will work here.

### **6. `/tests/`**
- **Purpose:** Contains unit tests to ensure the reliability of each component.
  - **Unit Tests:** Each file tests a specific module or component of the project. Developers working on quality assurance and testing will focus here.

## Assigning Work to Developers

- **User Interaction (Input/Output):** Work on `/input_handler/` directory.
- **Translation Logic:** Focus on `/translation/` directory.
- **Transformer Development:** Work on `/custom_transformers/` directory and `/training/` scripts.
- **LLM Integration:** Focus on `/llm/` directory.
- **Testing:** Focus on `/tests/` directory and ensure all modules are covered.
- **Evaluation and Performance Testing:** Use and potentially extend `next_word_prediction_test.py`.
![](He_LLM.jpg)