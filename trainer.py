import os
import torch
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from dataclasses import dataclass, field
from typing import Dict, Any
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from configs import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WARMUP_STEPS, MAX_GRAD_NORM
from prefix_t5_model import PrefixT5


@dataclass
class TrainingHyperParams:
    num_epochs: int = NUM_EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    warmup_steps: int = WARMUP_STEPS
    max_grad_norm: float = MAX_GRAD_NORM
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = field(init=False) 

    def __post_init__(self):
        self.save_path = self._model_save_path(base_name="prefix_t5")

    def _model_save_path(self, base_name="prefix_t5") -> str:
        params_str = "-".join(f"{key}_{value}" for key, value in vars(self).items() if key != "save_path")
        return f"{base_name}-{params_str}.pth"


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        tokenizer: T5Tokenizer,
        args: TrainingHyperParams,
    ):
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.args = args

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        num_training_steps = len(train_loader) * args.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training", unit="batch"):
            assert "input_ids" in batch and\
                "attention_mask" in batch and\
                "labels" in batch and\
                "decoder_input_ids" in batch
            
            input_ids = batch["input_ids"].to(self.args.device)
            attention_mask = batch["attention_mask"].to(self.args.device)
            labels = batch["labels"].to(self.args.device)
            decoder_input_ids = batch["decoder_input_ids"].to(self.args.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Clip gradients
            clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _evaluate_one_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Validation", unit="batch"):
            assert "input_ids" in batch and\
                "attention_mask" in batch and\
                "labels" in batch and\
                "decoder_input_ids" in batch
            
            input_ids = batch["input_ids"].to(self.args.device)
            attention_mask = batch["attention_mask"].to(self.args.device)
            labels = batch["labels"].to(self.args.device)
            decoder_input_ids = batch["decoder_input_ids"].to(self.args.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
            )

            loss = outputs.loss
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self) -> Dict[str, Any]:
        best_valid_loss = float("inf")
        epoch_train_loss = []
        epoch_valid_loss = []

        for epoch in range(self.args.num_epochs):
            
            train_loss = self._train_one_epoch()
            valid_loss = self._evaluate_one_epoch(self.valid_loader)
            epoch_train_loss.append(train_loss)
            epoch_valid_loss.append(valid_loss)
            
            print(
                f"Epoch {epoch + 1}/{self.args.num_epochs}\n"
                f"\tTrain Loss: {train_loss:.4f}\n"
                f"\tValidation Loss: {valid_loss:.4f}\n"
            )   

            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_model(self.args.save_path)

        return {
            "train_losses": epoch_train_loss,
            "valid_losses": epoch_valid_loss,
        }
    
    # def decode_batch(self, batch, generated_ids):
    #     results = []
    #     for i in range(len(generated_ids)):
    #         source_text = self.tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
    #         target_text = self.tokenizer.decode(batch['labels'][i], skip_special_tokens=True)
    #         generated_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
    #         results.append({
    #             'source': source_text,
    #             'target': target_text,
    #             'generated': generated_text
    #         })
    #     return results

    def save_model(self, save_path: str) -> None:
        save_dir = os.path.join("models", os.path.basename(save_path))
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)        
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            save_dir,
        )

        print(f"Model saved at models/{save_path}")


def load_model(
    base_model: torch.nn.Module,
    save_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.nn.Module:
    
    model = PrefixT5(t5_model=base_model).to(device)
    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])    
    print(f"Model successfully loaded from {save_path}")
    return model
