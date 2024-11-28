import torch

from transformers import T5Tokenizer


@torch.no_grad()
def generate_translation(
    model: torch.nn.Module,
    tokenizer: T5Tokenizer,
    input_text, max_length: int = 128, 
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

    generated_ids = []

    for _ in range(max_length):
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            decoder_input_ids=decoder_input_ids,
        )

        print(outputs)
        print(outputs.shape)

        logits = outputs.logits[:, -1, :]
        next_token_id = logits.argmax(dim=-1)
        generated_ids.append(next_token_id)

        decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(-1)], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(torch.cat(generated_ids), skip_special_tokens=True)
    return generated_text
