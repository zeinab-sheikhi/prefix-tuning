from torch.utils.data import Dataset, DataLoader, random_split
from transformers import T5Tokenizer, T5ForConditionalGeneration


# Define TranslationDataset class
class TranslationDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer: T5Tokenizer,
        max_length: int = 128,
        source_lang: str = "en",
        target_lang: str = "fr",
        task_prefix: str = "translate English to French: "
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.task_prefix = task_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single example from the dataset.

        Args:
            idx: Index of the example
            
        Returns:
            Dictionary containing input_ids, decoder_input_ids, labels, attention_mask
        """
        example = self.data[idx]
        source_text = self.task_prefix + example['translation'][self.source_lang]
        target_text = example['translation'][self.target_lang]
        
        # Tokenize inputs and targets
        encoded = self.tokenizer(
            text=source_text,
            text_target=target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        features = {
            'input_ids': encoded["input_ids"].squeeze(0),
            'attention_mask': encoded["attention_mask"].squeeze(0),
            'labels': encoded["labels"].squeeze(0),
        }
        return features

# Create data loaders
def create_dataloader(dataset, tokenizer, batch_size, max_length):
    """
    Create a DataLoader for a given dataset.
    """
    translation_dataset = TranslationDataset(dataset, tokenizer, max_length)
    return DataLoader(
        translation_dataset,
        batch_size=batch_size,
        shuffle=True
    )
