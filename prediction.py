import torch
from transformers import T5Tokenizer
from typing import List

from metrics import compute_blue
from trainer import load_model


@torch.no_grad()
def generate_translation(
    model: torch.nn.Module,
    tokenizer: T5Tokenizer,
    input_text,
    max_length: int = 128, 
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
        
    """Generate and print sample translations"""
    model.eval().to(device)

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    decoder_input_ids = torch.full(
        (input_ids.size(0), 1),  # batch_size x 1
        tokenizer.pad_token_id,  # Start with <pad> (can be adjusted for specific tasks)
        device=device,
    )

    finished_sequences = torch.zeros(input_ids.size(0), dtype=torch.bool, device=device)

    for _ in range(max_length):
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            decoder_input_ids=decoder_input_ids,
        )

        logits = outputs.logits[:, -1, :]
        next_token_id = logits.argmax(dim=-1, keepdim=True)

        decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)

        finished_sequences |= (next_token_id.squeeze(-1) == tokenizer.eos_token_id)

        if finished_sequences.all():
            break

    generated_texts = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in decoder_input_ids
    ]
    return generated_texts


def evaluate(
    model_path: str,     
    srcs: List[str], 
    refs: List[str],
    t5_model,
    t5_tokenizer: T5Tokenizer,
    device="cpu"
):
    
    parts = model_path.split('/')[-1]
    prefix_length = int(parts.split('-')[0].split('_')[-1])  
    prefix_projection_size = int(parts.split('-')[1].split('_')[-1]) 
    lr_start = model_path.find("lr") + 2  
    learning_rate = model_path[lr_start:].split(".")[0][1:]

    prefixed_t5_model = load_model(
        base_model=t5_model,
        save_path=model_path,
        device=device,
        prefix_length=prefix_length, 
        prefix_projection_size=prefix_projection_size
    )

    preds = generate_translation(
        model=prefixed_t5_model, 
        tokenizer=t5_tokenizer, 
        input_text=srcs,
        max_length=1000
    )

    bleu = compute_blue(refs=refs, preds=preds)

    return {
        "prefix_length": prefix_length,
        'learning_rate': learning_rate, 
        # "prefix_projection_size": prefix_projection_size, 
        "blue": bleu
    }
