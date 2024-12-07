import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from translation_dataset import create_dataloader, TranslationDataset
from prefix_t5_model import PrefixT5
from trainer import Trainer, TrainingHyperParams
from configs import CHECKPOINT, DATASET, BATCH_SIZE

def hyperparameter_tuning(
    prefix_lengths=[5, 10, 15],
    projection_sizes=[512],
    learning_rates=[8e-05, 5e-05],
    num_epochs=20
):

    # Load base models
    t5_tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT)
    t5_model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT)

    # Load and preprocess dataset
    dataset = load_dataset(DATASET, "en-fr")['train']
    dataset = dataset.shuffle(seed=42)

    # Split dataset
    train_valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_dataset, valid_dataset = torch.utils.data.random_split(train_valid_dataset, [0.9, 0.1])

    # Prepare datasets OUTSIDE the nested loop
    train_translation_dataset = TranslationDataset(
        data=train_dataset,
        tokenizer=t5_tokenizer,
        shift_right_func=t5_model._shift_right
    )

    valid_translation_dataset = TranslationDataset(
        data=valid_dataset,
        tokenizer=t5_tokenizer,
        shift_right_func=t5_model._shift_right
    )

    train_loader = create_dataloader(
        dataset=train_translation_dataset,
        batch_size=BATCH_SIZE
    )

    valid_loader = create_dataloader(
        dataset=valid_translation_dataset,
        batch_size=BATCH_SIZE
    )

    # Prepare results dictionary
    results = {}

    # Hyperparameter tuning grid search
    for prefix_length in prefix_lengths:
        for projection_size in projection_sizes:
            for learning_rate in learning_rates:
                # Create unique experiment identifier
                experiment_id = (
                    f"prefix_len_{prefix_length}-"
                    f"proj_size_{projection_size}-"
                    f"lr_{learning_rate}"
                )
                print(f"\nRunning Experiment: {experiment_id}")

                # Create PrefixT5 model with current hyperparameters
                prefixed_t5_model = PrefixT5(
                    t5_model=t5_model, 
                    prefix_length=prefix_length, 
                    prefix_projection_size=projection_size
                )

                # Customize training hyperparameters
                training_params = TrainingHyperParams(
                    num_epochs=num_epochs,
                    batch_size=BATCH_SIZE,
                    learning_rate=learning_rate,
                    warmup_steps=100,
                    max_grad_norm=1,
                    prefix_length=prefix_length  # Add this line
                )

                # Modify save path to include experiment details
                training_params.save_path = f"models/{experiment_id}.pth"

                # Create trainer
                trainer = Trainer(
                    model=prefixed_t5_model,
                    tokenizer=t5_tokenizer,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    test_loader=valid_loader,  # Using valid loader as test loader for this example
                    args=training_params,
                    experiment_id=experiment_id  # Pass experiment_id to Trainer
                )

                # Train and store results
                training_results = trainer.train()
                
                # Store results for this experiment
                results[experiment_id] = {
                    'train_losses': training_results['train_losses'],
                    'valid_losses': training_results['valid_losses'],
                    'config': {
                        'prefix_length': prefix_length,
                        'projection_size': projection_size,
                        'learning_rate': learning_rate
                    }
                }

    return results

# # Run hyperparameter tuning
tuning_results = hyperparameter_tuning()

# # Analyze and print results
# def print_best_configurations(results):
#     """
#     Print best configurations based on validation loss
#     """
#     print("\n--- Hyperparameter Tuning Results ---")
#     sorted_results = sorted(
#         results.items(), 
#         key=lambda x: min(x[1]['valid_losses'])
#     )

#     print("\nTop 3 Best Configurations:")
#     for i, (exp_id, result) in enumerate(sorted_results[:3], 1):
#         print(f"\n{i}. Experiment: {exp_id}")
#         print(f"   Final Validation Loss: {min(result['valid_losses']):.4f}")
#         print("   Configuration:")
#         for key, value in result['config'].items():
#             print(f"     {key}: {value}")

# # Analyze results
# print_best_configurations(tuning_results)
