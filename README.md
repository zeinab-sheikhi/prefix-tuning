# Prefix-Tuning for Natural Language Generation

This implementation is taken from the paper:
["Prefix-Tuning: Optimizing Continuous Prompts for Generation"](https://arxiv.org/abs/2101.00190) (Li & Liang, 2021)


## Overview
This project implements prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks. Prefix-tuning prepends trainable continuous prefixes to frozen language models, achieving comparable performance to full fine-tuning while only updating 0.1%-3% of the parameters.

## Tasks and Datasets

### Machine Translation (English to French)
- **Dataset**: [Helsinki-NLP/opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books)
- **Size**: Contains bilingual sentence pairs for English-to-French translation.
- **Description**: This task involves translating sentences from English to French.
- **Model**: T5 with prefix-tuning
- **Prefix Length**: 10 tokens



## Project Structure

```plaintext
├── README.md              # Documentation for the project
├── app.ipynb              # Jupyter Notebook for model experimentation
├── configs.py             # Configuration settings for the project
├── hyp_tuning_script.py   # Script for hyperparameter tuning
├── metrics.py             # Evaluation metrics implementation
├── prediction.py          # Prediction logic for the translation model
├── prefix_t5_model.py     # T5 model with prefix-tuning integration
├── requirements.txt       # List of project dependencies
├── trainer.py             # Training pipeline for the model
├── translation_dataset.py # Dataset preparation and loading utilities

### `configs.py`
This file contains configuration settings for the project, including:
- Model checkpoint: `google-t5/t5-small`
- Dataset: `Helsinki-NLP/opus_books`
- Hyperparameters: batch size, learning rate, number of epochs, warmup steps, and gradient clipping.
- Prefix-tuning parameters: prefix length and projection size.