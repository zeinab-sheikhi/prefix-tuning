import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from dataclasses import dataclass
from typing import Dict, Any
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from translation_dataset import create_dataloader
from prefix_t5_model import PrefixT5
from configs import CHECKPOINT, BATCH_SIZE, MAX_LENGTH, NUM_EPOCHS, LEARNING_RATE, PREFIX_LENGTH, PREFIX_PROJECTION_SIZE, WARMUP_STEPS, MAX_GRAD_NORM


@dataclass
class TrainingHyperParams:
    num_epochs: int = NUM_EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    prefix_length: int = PREFIX_LENGTH
    prefix_projection_size: int = PREFIX_PROJECTION_SIZE
    warmup_steps: int = WARMUP_STEPS
    max_grad_norm: float = MAX_GRAD_NORM
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "models/model.pth"


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
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
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
            valid_loss = self._evaluate_one_epoch(self.valid_loader, "valid")
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

        # # Final evaluation on test set
        # test_metrics = self.evaluate(self.test_loader, "test")
        # print(f"Test Metrics: {test_metrics}")

        return {
            "train_losses": epoch_train_loss,
            "valid_losses": epoch_valid_loss,
        }


    
    def save_model(self, save_path: str) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            save_path,
        )

        print(f"Model saved at {save_path}")


