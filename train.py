import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from preprocessing import create_dataloader
from configs import CHECKPOINT, DATASET, BATCH_SIZE, MAX_LENGTH

t5_tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT)
t5_model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT)

# Load dataset
dataset = load_dataset(DATASET, "en-fr")['train']

# Shuffle the dataset
dataset = dataset.shuffle(seed=42)

# Split the dataset into train, valid, test (train_size=80%, valid_size=10%, test_size=10%)
train_valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
train_dataset, valid_dataset = torch.utils.data.random_split(train_valid_dataset, [0.9, 0.1])

# Prepare dataloaders for train, valid, and test datasets
train_loader = create_dataloader(train_dataset, t5_tokenizer, BATCH_SIZE, MAX_LENGTH)
valid_loader = create_dataloader(valid_dataset, t5_tokenizer, BATCH_SIZE, MAX_LENGTH)
test_loader = create_dataloader(test_dataset, t5_tokenizer, BATCH_SIZE, MAX_LENGTH)
