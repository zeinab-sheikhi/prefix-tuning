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
    prefix_length: int = 0  # Add this line
    save_path: str = field(init=False) 

    def __post_init__(self):
        self.save_path = self._model_save_path(base_name="prefix_t5")

    def _model_save_path(self, base_name="prefix_t5") -> str:
        params_str = "-".join(f"{key}_{value}" for key, value in vars(self).items() if key != "save_path")
        return f"{base_name}-{params_str}.pth"
import csv

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        tokenizer: T5Tokenizer,
        args: TrainingHyperParams,
        experiment_id: str = None  # Add experiment_id parameter
    ):
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.args = args
        self.experiment_id = experiment_id  # Store experiment_id

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.learning_rate
        )
        num_training_steps = len(train_loader) * args.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", unit="batch")):
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

            # Log batch loss to CSV
            self._log_to_csv(epoch, batch_idx, loss.item(), "train")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _evaluate_one_epoch(self, epoch: int, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", unit="batch")):
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

            # Log batch loss to CSV
            self._log_to_csv(epoch, batch_idx, loss.item(), "valid")

        return total_loss / len(dataloader)

    def _log_to_csv(self, epoch: int, batch_idx: int, loss: float, phase: str) -> None:
        if self.experiment_id:
            csv_file = f"metrics_{self.experiment_id}.csv"
            with open(csv_file, mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, batch_idx, loss, phase, self.args.prefix_length])

    def train(self) -> Dict[str, Any]:
        best_valid_loss = float("inf")
        epoch_train_loss = []
        epoch_valid_loss = []

        # Create CSV file and write header
        if self.experiment_id:
            csv_file = f"metrics_{self.experiment_id}.csv"
            with open(csv_file, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "batch_idx", "loss", "phase", "prefix_length"])

        for epoch in range(self.args.num_epochs):
            train_loss = self._train_one_epoch(epoch)
            valid_loss = self._evaluate_one_epoch(epoch, self.valid_loader)
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
    
    
# def load_model(
#     base_model: torch.nn.Module,
#     save_path: str,
#     device: str = "cuda" if torch.cuda.is_available() else "cpu",
# ) -> torch.nn.Module:
    
#     model = PrefixT5(t5_model=base_model).to(device)
#     checkpoint = torch.load(save_path, map_location=device, weights_only=True)
#     model.load_state_dict(checkpoint["model_state_dict"])    
#     print(f"Model successfully loaded from {save_path}")
#     return model
def load_model(
    base_model: torch.nn.Module,
    save_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    prefix_length: int = None,
    prefix_projection_size: int = None
) -> torch.nn.Module:
    # If specific prefix parameters are provided, use them
    if prefix_length is not None and prefix_projection_size is not None:
        model = PrefixT5(
            t5_model=base_model, 
            prefix_length=prefix_length, 
            prefix_projection_size=prefix_projection_size
        ).to(device)
    else:
        # Fallback to default or existing configuration
        model = PrefixT5(t5_model=base_model).to(device)
    
    checkpoint = torch.load(save_path, map_location=device)
    
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        print(f"Warning: Could not load exact state dict. Error: {e}")
        # Optionally implement more sophisticated state dict loading
    
    print(f"Model successfully loaded from {save_path}")
    return model