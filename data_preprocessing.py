from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer


# Define TranslationDataset class
class TranslationDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer: T5Tokenizer,
        shift_right_func: callable = None,
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
        self.shift_right_func = shift_right_func

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
        item = self.data[idx]
        source_text = self.task_prefix + item['translation'][self.source_lang]
        target_text = item['translation'][self.target_lang]
        
        # Tokenize inputs and targets
        encoded = self.tokenizer(
            text=source_text,
            text_target=target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        
        features = {
            'input_ids': encoded["input_ids"],
            'attention_mask': encoded["attention_mask"],
            'labels': encoded["labels"],
            'decoder_input_ids': self.shift_right_func(encoded['labels'])
        }
        return features


# Create data loaders
def create_dataloader(dataset, tokenizer, shift_right_func, batch_size, max_length):
    """
    Create a DataLoader for a given dataset.
    """
    translation_dataset = TranslationDataset(data=dataset, tokenizer=tokenizer, max_length=max_length, shift_right_func=shift_right_func)
    return DataLoader(
        translation_dataset,
        batch_size=batch_size,
        shuffle=True
    )
