"""
Training script for the Visual Token Diffusion Language Model.
"""

import argparse
import numpy as np
import torch
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
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Strip whitespace and filter empty lines
    samples = [line.strip() for line in lines if line.strip()]
    return samples


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
    
    # Take the most frequent words
    vocab = [word for word, count in sorted_words[:max_vocab_size]]
    
    return vocab


def create_batches(samples: List[str], batch_size: int) -> List[List[str]]:
    """
    Create batches from text samples.
    
    Args:
        samples: List of text samples
        batch_size: Batch size
    
    Returns:
        List of batches
    """
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
          train_samples: List[str], 
          val_samples: List[str],
          batch_size: int = 32, 
          num_epochs: int = 10, 
          learning_rate: float = 0.001,
          save_dir: str = "checkpoints",
          log_interval: int = 10):
    """
    Train the Visual Token Diffusion LM.
    
    Args:
        model: VisualTokenDiffusionLM instance
        train_samples: Training samples
        val_samples: Validation samples
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
        log_interval: Interval (in batches) to log training progress
    """
    # Create optimizer
    params = []
    if hasattr(model.diffusion, "parameters"):
        params.extend(model.diffusion.parameters())
    if model.is_encoder_learnable:
        params.extend(model.encoder.parameters())
    if model.is_decoder_learnable:
        params.extend(model.decoder.parameters())
    
    optimizer = optim.Adam(params, lr=learning_rate)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    all_losses = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Create batches
        batches = create_batches(train_samples, batch_size)
        
        # Training
        model.diffusion.train()
        if model.is_encoder_learnable:
            model.encoder.train()
        if model.is_decoder_learnable:
            model.decoder.train()
        
        epoch_losses = []
        
        for i, batch in enumerate(tqdm(batches)):
            # Perform training step
            losses = model.train_step(batch, optimizer)
            epoch_losses.append(losses)
            
            # Log progress
            if (i + 1) % log_interval == 0:
                avg_loss = np.mean([l["total_loss"] for l in epoch_losses[-log_interval:]])
                print(f"  Batch {i+1}/{len(batches)}, Avg Loss: {avg_loss:.4f}")
        
        # Calculate average epoch loss
        avg_epoch_loss = np.mean([l["total_loss"] for l in epoch_losses])
        print(f"  Epoch {epoch+1} completed, Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save losses
        all_losses.extend(epoch_losses)
        
        # Validation (simplified)
        with torch.no_grad():
            model.diffusion.eval()
            if model.is_encoder_learnable:
                model.encoder.eval()
            if model.is_decoder_learnable:
                model.decoder.eval()
            
            # Sample a few examples for validation
            val_batch = val_samples[:5]
            for text in val_batch:
                # Encode
                patterns = model.encode_text(text)
                patterns_tensor = torch.tensor(patterns, device=model.device)
                
                # Sample from diffusion
                sampled_patterns = model.diffusion.sample(1, model.device)
                
                # Decode
                decoded_text = model.decode_patterns(patterns.reshape(-1, model.grid_size, model.grid_size))
                sampled_text = model.decode_patterns(sampled_patterns.cpu().numpy())
                
                print(f"Original: {text}")
                print(f"Reconstructed: {decoded_text}")
                print(f"Sampled: {sampled_text}")
                print("-" * 50)
        
        # Save checkpoint
        save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        model.save(save_path)
        print(f"Model saved to {save_path}")
    
    # Save loss history
    loss_history = {
        "total_loss": [l["total_loss"] for l in all_losses],
        "diffusion_loss": [l["diffusion_loss"] for l in all_losses],
        "decoder_loss": [l["decoder_loss"] for l in all_losses],
        "encoder_loss": [l["encoder_loss"] for l in all_losses]
    }
    
    with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f)
    
    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history["total_loss"], label="Total Loss")
    plt.plot(loss_history["diffusion_loss"], label="Diffusion Loss")
    if any(l > 0 for l in loss_history["decoder_loss"]):
        plt.plot(loss_history["decoder_loss"], label="Decoder Loss")
    if any(l > 0 for l in loss_history["encoder_loss"]):
        plt.plot(loss_history["encoder_loss"], label="Encoder Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()


def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description="Train a Visual Token Diffusion Language Model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data file")
    parser.add_argument("--val_data", type=str, help="Path to validation data file")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--encoder_type", type=str, default="deterministic", 
                        choices=["deterministic", "learnable"], help="Type of encoder")
    parser.add_argument("--diffusion_type", type=str, default="simple", 
                        choices=["simple", "advanced"], help="Type of diffusion model")
    parser.add_argument("--decoder_type", type=str, default="deterministic", 
                        choices=["deterministic", "learnable"], help="Type of decoder")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_samples = load_data(args.data)
    if args.val_data:
        val_samples = load_data(args.val_data)
    else:
        # Use a portion of training data for validation
        split = int(0.9 * len(train_samples))
        val_samples = train_samples[split:]
        train_samples = train_samples[:split]
    
    print(f"Loaded {len(train_samples)} training samples and {len(val_samples)} validation samples")
    
    # Create vocabulary
    print("Creating vocabulary...")
    vocab = create_vocabulary(train_samples, args.vocab_size)
    token_to_id = create_simple_tokenizer(vocab)
    print(f"Created vocabulary with {len(token_to_id)} tokens")
    
    # Create model
    print("Creating model...")
    model = VisualTokenDiffusionLM(
        token_to_id=token_to_id,
        encoder_type=args.encoder_type,
        diffusion_type=args.diffusion_type,
        decoder_type=args.decoder_type,
        device=device
    )
    
    # Train model
    print("Starting training...")
    train(
        model=model,
        train_samples=train_samples,
        val_samples=val_samples,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()