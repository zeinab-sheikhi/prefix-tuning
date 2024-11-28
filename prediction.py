import torch
from prefix_t5_model import PrefixT5
from configs import PREFIX_LENGTH
from transformers import T5Tokenizer


def load_model(t5_model, model_path: str, device: torch.device):
    model = PrefixT5(model=t5_model, prefix_length=PREFIX_LENGTH).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print("Model Loaded Successfully")
    return model
            

def translate_sentence(english_sentence: str) -> str:
    # tokenize the english_sentence
    pass
    
