# Prefix-Tuning for Natural Language Generation

This implementation is taken from the paper:
["Prefix-Tuning: Optimizing Continuous Prompts for Generation"](https://arxiv.org/abs/2101.00190) (Li & Liang, 2021)


## Overview
This project implements prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks. Prefix-tuning prepends trainable continuous prefixes to frozen language models, achieving comparable performance to full fine-tuning while only updating 0.1%-3% of the parameters.

## Tasks and Datasets

### 1. Table-to-Text Generation (WebNLG)
- **Dataset**: WebNLG v2.0
- **Size**: 22K examples
- **Description**: Contains subject-property-object triples and their corresponding natural language descriptions
- **Example**:
  ```
  Input (Triple): 
  The_Eagle | location | riverside_area, The_Eagle | type | coffee_shop
  Output (Text): 
  The Eagle is a coffee shop located in the riverside area.
  ```
- **Model**: GPT-2 with prefix-tuning
- **Prefix Length**: 10 tokens

### 2. Summarization (XSUM)
- **Dataset**: XSUM (Extreme Summarization)
- **Size**: 225K examples
- **Description**: News articles with single-sentence summaries
- **Statistics**:
  - Average article length: 431 words
  - Average summary length: 23.3 words
- **Model**: BART with prefix-tuning
- **Prefix Length**: 200 tokens

## Model Architecture

### GPT Prefix-Tuning