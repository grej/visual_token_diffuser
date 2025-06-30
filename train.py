# visual_token_diffuser/train.py

"""
Training script for the Visual Token Diffusion Language Model.
Supports different training stages: 'reconstruction' (autoencoder) and 'diffusion'.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os
import json
import time
from tqdm import tqdm

# Import our components
from utils import create_simple_tokenizer, visualize_pattern, visualize_pattern_batch, analyze_pattern_space
from model import VisualTokenDiffusionLM


def load_data(file_path: str) -> List[str]:
    """
    Load text data from a file.

    Args:
        file_path: Path to the text file

    Returns:
        List of text samples
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Strip whitespace and filter empty lines
        samples = [line.strip() for line in lines if line.strip()]
        if not samples:
            print(f"Warning: No non-empty lines found in {file_path}")
        return samples
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        exit(1)


def create_vocabulary(samples: List[str], max_vocab_size: int = 1000) -> List[str]:
    """
    Create a vocabulary from text samples.

    Args:
        samples: List of text samples
        max_vocab_size: Maximum vocabulary size

    Returns:
        List of tokens (words) in the vocabulary
    """
    word_counts = {}

    for sample in samples:
        words = sample.lower().split()
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Take the most frequent words, subtract space for special tokens
    vocab = [word for word, count in sorted_words[:max_vocab_size - 3]] # Reserve space for <pad>, <unk>, <eos>

    return vocab


def get_temperature(epoch: int, num_epochs: int, start_temp: float = 5.0, end_temp: float = 0.5) -> float:
    """
    Calculate Gumbel-Softmax temperature with annealing schedule.
    
    Args:
        epoch: Current epoch (0-indexed)
        num_epochs: Total number of epochs
        start_temp: Starting temperature (higher = more uniform)
        end_temp: Ending temperature (lower = more focused)
        
    Returns:
        Current temperature value
    """
    if num_epochs <= 1:
        return start_temp
    progress = epoch / (num_epochs - 1)
    return start_temp * (end_temp / start_temp) ** progress


def create_batches(samples: List[str], batch_size: int) -> List[List[str]]:
    """
    Create batches from text samples. Handles potential empty list.

    Args:
        samples: List of text samples
        batch_size: Batch size

    Returns:
        List of batches
    """
    if not samples:
        return []
    # Shuffle samples
    indices = np.random.permutation(len(samples))

    # Create batches
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch = [samples[idx] for idx in batch_indices]
        batches.append(batch)

    return batches


def train(model: VisualTokenDiffusionLM,
          training_stage: str, # ADDED: Explicitly pass the stage
          train_samples: List[str],
          val_samples: List[str],
          batch_size: int = 32,
          num_epochs: int = 10,
          learning_rate: float = 0.001,
          save_dir: str = "checkpoints",
          log_interval: int = 10,
          load_checkpoint_path: Optional[str] = None, # ADDED: Path to load previous state
          fine_tune_autoencoder: bool = False): # ADDED: Option to fine-tune AE during diffusion stage
    """
    Train the Visual Token Diffusion LM for a specific stage.

    Args:
        model: VisualTokenDiffusionLM instance, initialized for the correct stage.
        training_stage: The stage to run ('reconstruction' or 'diffusion').
        train_samples: Training samples.
        val_samples: Validation samples.
        batch_size: Batch size.
        num_epochs: Number of training epochs for this stage.
        learning_rate: Learning rate.
        save_dir: Directory to save checkpoints and logs for this stage.
        log_interval: Interval (in batches) to log training progress.
        load_checkpoint_path: Path to a checkpoint to load weights from (e.g., AE weights for diffusion stage).
        fine_tune_autoencoder: If True and stage is 'diffusion', include AE params in optimizer.
    """
    print(f"--- Starting Training Stage: {training_stage.upper()} ---")

    # --- 1. Configure Optimizer based on Stage ---
    params_to_optimize = []
    if training_stage == 'reconstruction':
        if not model.is_encoder_learnable or not model.is_decoder_learnable:
             raise ValueError("Reconstruction stage requires LearnableEncoder and LearnableDecoder.")
        print("Optimizer: Training Encoder and Decoder.")
        params_to_optimize.extend(model.encoder.parameters())
        params_to_optimize.extend(model.decoder.parameters())
    elif training_stage == 'diffusion':
        print("Optimizer: Training Diffusion Model.")
        params_to_optimize.extend(model.diffusion.parameters())
        if fine_tune_autoencoder:
            if model.is_encoder_learnable:
                print("Optimizer: Fine-tuning Encoder.")
                params_to_optimize.extend(model.encoder.parameters())
            if model.is_decoder_learnable:
                 print("Optimizer: Fine-tuning Decoder.")
                 params_to_optimize.extend(model.decoder.parameters())
    else:
        raise ValueError(f"Unknown training stage: {training_stage}")

    if not params_to_optimize:
        print("Warning: No parameters selected for optimization in this stage. Check model configuration.")
        return # Nothing to train

    optimizer = optim.Adam(params_to_optimize, lr=learning_rate)

    # --- 2. Load Pre-trained Weights if specified ---
    # This is typically used to load AE weights when starting the diffusion stage.
    if load_checkpoint_path:
        if os.path.exists(load_checkpoint_path):
            print(f"Loading weights from checkpoint: {load_checkpoint_path}")
            try:
                # Use the model's load method, but we only care about the state dicts here
                # We need to load selectively based on the current stage's needs
                map_location = model.device
                loaded_state = torch.load(load_checkpoint_path, map_location=map_location)

                # Load diffusion state if present and relevant
                if 'diffusion_state' in loaded_state and training_stage == 'diffusion':
                     model.diffusion.load_state_dict(loaded_state['diffusion_state'])
                     print("  - Loaded Diffusion state.")

                # Load encoder state if learnable and present
                if model.is_encoder_learnable and loaded_state.get('encoder_state') is not None:
                    model.encoder.load_state_dict(loaded_state['encoder_state'])
                    print("  - Loaded Encoder state.")
                elif model.is_encoder_learnable:
                     print(f"  - Warning: Learnable encoder expected but no state found in {load_checkpoint_path}.")

                # Load decoder state if learnable and present
                if model.is_decoder_learnable and loaded_state.get('decoder_state') is not None:
                    model.decoder.load_state_dict(loaded_state['decoder_state'])
                    print("  - Loaded Decoder state.")
                elif model.is_decoder_learnable:
                     print(f"  - Warning: Learnable decoder expected but no state found in {load_checkpoint_path}.")

            except Exception as e:
                print(f"Error loading checkpoint from {load_checkpoint_path}: {e}")
                print("Continuing with initial weights.")
        else:
            print(f"Warning: Checkpoint path specified but not found: {load_checkpoint_path}")

    # --- 3. Prepare Save Directory and Logging ---
    stage_save_dir = os.path.join(save_dir, training_stage)
    os.makedirs(stage_save_dir, exist_ok=True)
    print(f"Checkpoints and logs for this stage will be saved in: {stage_save_dir}")

    # --- 4. Training Loop ---
    all_losses = []
    best_val_loss = float('inf') # Simple validation metric

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} [{training_stage.upper()}]")

        # Create batches
        batches = create_batches(train_samples, batch_size)
        if not batches:
            print("Warning: No training batches created. Check data.")
            continue

        # Set models to appropriate train mode (handled within model.train_step now)
        epoch_losses = []

        # Training iterations
        pbar = tqdm(batches, desc=f"Epoch {epoch+1} Training")
        for i, batch in enumerate(pbar):
            # Perform training step (model internally uses self.training_stage)
            losses = model.train_step(batch, optimizer, epoch=epoch, num_epochs=num_epochs)
            epoch_losses.append(losses)

            # Log progress
            if (i + 1) % log_interval == 0 or i == len(batches) - 1:
                avg_loss = np.mean([l.get("total_loss", 0) for l in epoch_losses[-log_interval:]])
                pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

        # Calculate average epoch loss
        avg_epoch_loss = np.mean([l.get("total_loss", 0) for l in epoch_losses])
        print(f"  Epoch {epoch+1} Training Completed. Avg Loss: {avg_epoch_loss:.4f}")

        # --- 5. Validation ---
        print("  Running Validation...")
        model.diffusion.eval() # Set diffusion to eval mode
        if model.is_encoder_learnable: model.encoder.eval()
        if model.is_decoder_learnable: model.decoder.eval()

        val_batches = create_batches(val_samples, batch_size)
        total_val_loss = 0
        num_val_tokens = 0
        correct_reconstructions = 0 # For reconstruction stage

        with torch.no_grad():
            for val_batch in tqdm(val_batches, desc="Validation", leave=False):
                if not val_batch: continue

                # --- Validation Logic specific to Stage ---
                if training_stage == 'reconstruction':
                    token_ids_tensor = model._get_token_ids_from_text_batch(val_batch)
                    if token_ids_tensor.numel() == 0: continue

                    # Use Gumbel-Softmax for validation consistency with training
                    # Fixed temperature for deterministic validation
                    pattern_gumbel = model.encoder.forward_gumbel(token_ids_tensor, temperature=1.0)
                    predicted_token_probs = model.decoder(pattern_gumbel)
                    val_loss = F.cross_entropy(predicted_token_probs, token_ids_tensor, reduction='sum')
                    total_val_loss += val_loss.item()
                    num_val_tokens += token_ids_tensor.numel()

                    # Calculate reconstruction accuracy
                    predicted_token_ids = torch.argmax(predicted_token_probs, dim=-1)
                    correct_reconstructions += (predicted_token_ids == token_ids_tensor).sum().item()
                    
                    # DEBUG: First epoch detailed analysis
                    if epoch == 0 and len(val_batches) == 1:  # First validation batch of first epoch
                        print(f"\n=== DEBUGGING RECONSTRUCTION ACCURACY ===")
                        print(f"Sample predictions vs targets (first 15):")
                        print(f"Predicted tokens: {predicted_token_ids[:15].tolist()}")
                        print(f"Target tokens:    {token_ids_tensor[:15].tolist()}")
                        
                        # Check if decoder is outputting the same token
                        unique_predictions = torch.unique(predicted_token_ids)
                        print(f"Unique predicted token IDs: {unique_predictions.tolist()}")
                        print(f"Total unique predictions: {len(unique_predictions)} out of {len(predicted_token_ids)} tokens")
                        
                        # Which tokens are being predicted correctly?
                        correct_mask = (predicted_token_ids == token_ids_tensor)
                        correct_indices = torch.where(correct_mask)[0]
                        if len(correct_indices) > 0:
                            print(f"Correctly predicted token IDs: {token_ids_tensor[correct_indices].tolist()}")
                            print(f"Number of correct predictions: {len(correct_indices)}")
                        else:
                            print("No correct predictions!")
                        
                        # Check target token distribution
                        unique_targets = torch.unique(token_ids_tensor)
                        print(f"Unique target token IDs: {unique_targets.tolist()}")
                        print(f"Target distribution: {len(unique_targets)} unique out of {len(token_ids_tensor)} tokens")
                        print("="*50)

                elif training_stage == 'diffusion':
                    # Get target patterns
                    all_patterns_list = []
                    for text in val_batch:
                         patterns_np = model.encode_text(text) # Uses encoder in eval mode
                         patterns = torch.tensor(patterns_np, dtype=torch.long, device=model.device)
                         all_patterns_list.append(patterns)
                    if not all_patterns_list: continue
                    patterns_tensor = torch.cat(all_patterns_list, dim=0)

                    # Calculate diffusion loss (average over random timesteps)
                    # Simulating multiple timesteps for a more stable validation loss
                    val_diff_loss_sum = 0
                    num_t_samples = 5 # Sample a few timesteps per batch item for validation
                    for _ in range(num_t_samples):
                         t = torch.randint(0, model.diffusion.timesteps, (patterns_tensor.shape[0],), device=model.device).long()
                         noisy_patterns = model.diffusion.add_noise(patterns_tensor, t)
                         predicted_pattern_probs = model.diffusion(noisy_patterns, t) # Diffusion model is already in eval mode
                         val_diff_loss = F.cross_entropy(
                             predicted_pattern_probs.reshape(-1, model.num_colors),
                             patterns_tensor.reshape(-1).long(),
                             reduction='sum'
                         )
                         val_diff_loss_sum += val_diff_loss.item()

                    total_val_loss += (val_diff_loss_sum / num_t_samples) # Average loss across t samples
                    num_val_tokens += patterns_tensor.numel() # Count total pattern cells validated

            # Calculate average validation loss
            avg_val_loss = total_val_loss / num_val_tokens if num_val_tokens > 0 else 0
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            if training_stage == 'reconstruction' and num_val_tokens > 0:
                 recon_accuracy = correct_reconstructions / num_val_tokens
                 print(f"  Reconstruction Accuracy: {recon_accuracy:.4f}")

        # --- 6. Save Checkpoint (Best and Last) ---
        # Save based on validation loss improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_save_path = os.path.join(stage_save_dir, f"model_best.pt")
            model.save(best_save_path)
            print(f"  New best validation loss. Model saved to {best_save_path}")

        # Save last epoch checkpoint
        last_save_path = os.path.join(stage_save_dir, f"model_epoch_{epoch+1}.pt")
        model.save(last_save_path)
        print(f"  Epoch {epoch+1} checkpoint saved to {last_save_path}")

        # Append epoch losses to overall history
        all_losses.extend(epoch_losses)


    # --- 7. Save Final Loss History for the Stage ---
    # Restructure loss history slightly
    loss_history = {}
    if all_losses:
        keys = all_losses[0].keys()
        for key in keys:
            loss_history[key] = [l.get(key, 0) for l in all_losses] # Collect losses for each key

        loss_hist_path = os.path.join(stage_save_dir, "loss_history.json")
        try:
            with open(loss_hist_path, "w") as f:
                json.dump(loss_history, f, indent=2)
            print(f"Loss history saved to {loss_hist_path}")
        except Exception as e:
             print(f"Error saving loss history: {e}")

        # Plot loss for the stage
        plot_path = os.path.join(stage_save_dir, "loss_plot.png")
        try:
            plt.figure(figsize=(12, 6))
            for key, values in loss_history.items():
                 # Only plot if there are non-zero values
                 if any(v != 0 for v in values):
                     plt.plot(values, label=key.replace('_', ' ').title())
            plt.xlabel("Batch Step")
            plt.ylabel("Loss")
            plt.title(f"Training Loss ({training_stage.title()} Stage)")
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_path)
            plt.close()
            print(f"Loss plot saved to {plot_path}")
        except Exception as e:
            print(f"Error plotting losses: {e}")

    print(f"--- Training Stage {training_stage.upper()} Completed ---")


def main():
    """Main function to parse arguments and run the training script."""
    parser = argparse.ArgumentParser(description="Train a Visual Token Diffusion Language Model")

    # Data and Vocab
    parser.add_argument("--data", type=str, required=True, help="Path to training data file (.txt)")
    parser.add_argument("--val_data", type=str, help="Path to validation data file (.txt). If not provided, uses 10% of training data.")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Maximum vocabulary size.")

    # Model Configuration
    parser.add_argument("--encoder_type", type=str, default="deterministic",
                        choices=["deterministic", "learnable"], help="Type of encoder.")
    parser.add_argument("--diffusion_type", type=str, default="simple",
                        choices=["simple", "advanced"], help="Type of diffusion model.")
    parser.add_argument("--decoder_type", type=str, default="deterministic",
                        choices=["deterministic", "learnable"], help="Type of decoder.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size for learnable components.")
    parser.add_argument("--diffusion_timesteps", type=int, default=50, help="Number of timesteps for diffusion.")
    # Grid size and num colors could be args too, but keeping fixed for simplicity now

    # Training Control
    parser.add_argument("--stage", type=str, default="diffusion", required=True,
                        choices=["reconstruction", "diffusion"], help="Specify the training stage to run.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs for the specified stage.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Base directory to save checkpoints and logs.")
    parser.add_argument("--log_interval", type=int, default=20, help="Log training progress every N batches.")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to a checkpoint (.pt file) to load weights from before starting training (e.g., load AE weights for diffusion stage).")
    parser.add_argument("--fine_tune_autoencoder", action="store_true",
                        help="If set, allows fine-tuning of encoder/decoder during the diffusion stage (only applies if --stage diffusion and encoder/decoder are learnable).")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available.")

    args = parser.parse_args()

    # Validate args
    if args.stage == "reconstruction" and (args.encoder_type != "learnable" or args.decoder_type != "learnable"):
         parser.error("--stage reconstruction requires --encoder_type learnable and --decoder_type learnable.")
    if args.fine_tune_autoencoder and args.stage != "diffusion":
        parser.error("--fine_tune_autoencoder only applies when --stage is diffusion.")

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_samples = load_data(args.data)
    if args.val_data:
        val_samples = load_data(args.val_data)
    else:
        print("No validation data provided, using 10% of training data for validation.")
        split_idx = int(0.9 * len(train_samples))
        if split_idx == 0 and len(train_samples) > 0: split_idx = 1 # Ensure at least one val sample if possible
        val_samples = train_samples[split_idx:]
        train_samples = train_samples[:split_idx]

    if not train_samples:
        print("Error: No training samples loaded. Exiting.")
        exit(1)
    print(f"Loaded {len(train_samples)} training samples and {len(val_samples)} validation samples.")

    # Create vocabulary
    print("Creating vocabulary...")
    vocab = create_vocabulary(train_samples, args.vocab_size)
    token_to_id = create_simple_tokenizer(vocab)
    actual_vocab_size = len(token_to_id)
    print(f"Created vocabulary with {actual_vocab_size} tokens (including special tokens).")

    # Create model instance for the specified stage
    print(f"Creating model for stage: {args.stage}...")
    # Pass the selected stage to the model constructor
    model = VisualTokenDiffusionLM(
        token_to_id=token_to_id,
        encoder_type=args.encoder_type,
        diffusion_type=args.diffusion_type,
        decoder_type=args.decoder_type,
        hidden_dim=args.hidden_dim,
        diffusion_timesteps=args.diffusion_timesteps,
        device=device,
        initial_training_stage=args.stage # Set the stage here
    )

    # Train model for the specified stage
    print(f"Starting training for stage: {args.stage}...")
    train(
        model=model,
        training_stage=args.stage, # Pass stage to train function
        train_samples=train_samples,
        val_samples=val_samples,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        load_checkpoint_path=args.load_checkpoint, # Pass loading path
        fine_tune_autoencoder=args.fine_tune_autoencoder # Pass fine-tuning flag
    )

    print("\nTraining script finished.")


if __name__ == "__main__":
    main()
