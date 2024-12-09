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
```

#### `configs.py`
This file contains configuration settings for the project, including:
- Model checkpoint: `google-t5/t5-small`
- Dataset: `Helsinki-NLP/opus_books`
- Hyperparameters: batch size, learning rate, number of epochs, warmup steps, and gradient clipping.
- Prefix-tuning parameters: prefix length and projection size.

#### `translation_dataset.py`
This file defines a custom PyTorch `Dataset` class for translation tasks and a function to create dataloaders. It processes bilingual data by tokenizing input and target texts, adding task-specific prefixes, and preparing them for the T5 model. The `create_dataloader` function facilitates batching and shuffling for training.

#### `prefix_t5_model.py`
This file implements a custom `PrefixT5` model for prefix-tuning the T5 architecture. It includes:
- **PrefixT5 Class:** Modifies the T5 model by adding trainable prefix embeddings for both encoder and decoder while freezing the base model's parameters.
- **PrefixLayer Class:** Defines the prefix embeddings with reparameterization using a multi-layer perceptron (MLP) for dynamic projection.
The model optimizes resource usage by tuning only the prefix parameters while leveraging the T5 architecture for efficient translation tasks.

#### `trainer.py`
This file implements the training pipeline for the translation model. It includes:
- **Trainer Class:** Handles training, validation, and model saving. Supports gradient clipping, learning rate scheduling, and logging losses to CSV files.
- **TrainingHyperParams Class:** Stores hyperparameters like batch size, learning rate, and device settings, while dynamically generating model save paths.
- **Model Saving and Loading:** Provides utilities to save the model state and load pretrained models with optional prefix-tuning configurations.

#### `hyp_tuning_script.py`
This script performs hyperparameter tuning for the translation model using grid search. Key functionalities include:
- **Dataset Preparation:** Loads, shuffles, and splits the dataset into training, validation, and testing sets.
- **Model Customization:** Dynamically creates `PrefixT5` models with different prefix lengths, projection sizes, and learning rates.
- **Training and Evaluation:** Trains the model for each configuration and logs results, including training and validation losses, for comparison.
- **Results Storage:** Stores experiment configurations and performance metrics for further analysis.

#### `metrics.py`
This file provides a function to compute BLEU scores using the `sacrebleu` library. It takes predicted and reference sentences as input, processes them, and calculates the BLEU score to evaluate the quality of the translations.

#### `prediction.py`
This file provides utilities for generating translations and evaluating the model. It includes:
- **`generate_translation`:** Generates translations for input texts using the model and tokenizer. Supports iterative decoding with a customizable maximum length.
- **`evaluate`:** Loads a pretrained model, generates translations for a list of source sentences, and computes BLEU scores against reference translations to assess model performance.

#### `app.ipynb`
This notebook provides a comprehensive overview of the process, starting with data preprocessing, where the dataset is loaded, shuffled, and split into training, validation, and test sets. The model initialization phase includes setting up both the T5 base model and the prefix-tuned model, incorporating prefix embeddings to enhance performance with minimal parameter updates. The training pipeline trains both models, logging losses and fine-tuning hyperparameters to achieve optimal results. Evaluation compares the BLEU scores of the prefix-tuned model and the T5 base model, quantifying the improvements made by the prefix-tuning approach. This procedure highlights the efficiency of prefix-tuning in adapting pre-trained models for translation tasks.


### Evaluation Results on test dataset

The table below compares the BLEU scores of the prefix-tuned model across different prefix lengths and learning rates. These results highlight the performance improvements achieved through hyperparameter tuning.

| Prefix Length | Learning Rate | BLEU Score         |
|---------------|---------------|--------------------|
| 5             | 5e-05         | 7.5298            |
| 5             | 8e-05         | 8.9814            |
| 10            | 5e-05         | 10.0716           |
| 10            | 8e-05         | 9.3513            |
| 15            | 5e-05         | 9.1901            |
| 15            | 8e-05         | 10.1679           |

