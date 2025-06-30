#!/usr/bin/env python3
"""
Text generation script for the Visual Token Diffusion Language Model.
Loads a trained model and generates text using the diffusion process.
"""

import argparse
import torch
from model import VisualTokenDiffusionLM


def generate_text(model_path: str, prompt: str = "", num_tokens: int = 50, temperature: float = 1.0, device: str = "cpu") -> str:
    """
    Generate text using a trained Visual Token Diffusion Language Model.
    
    Args:
        model_path: Path to the trained model checkpoint
        prompt: Optional text prompt to seed generation
        num_tokens: Number of tokens to generate
        temperature: Temperature for sampling (higher = more random)
        device: Device to run on ("cpu" or "cuda")
        
    Returns:
        Generated text string
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        model = VisualTokenDiffusionLM.load(model_path, device=device)
        model.diffusion.eval()
        if model.is_encoder_learnable:
            model.encoder.eval()
        if model.is_decoder_learnable:
            model.decoder.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return ""
    
    # Generate text
    print(f"Generating {num_tokens} tokens with temperature {temperature}...")
    if prompt:
        print(f"Prompt: '{prompt}'")
    
    try:
        generated = model.generate(prompt=prompt, max_length=num_tokens, temperature=temperature)
        return generated
    except Exception as e:
        print(f"Error during generation: {e}")
        return ""


def main():
    """Main function to parse arguments and run text generation."""
    parser = argparse.ArgumentParser(description="Generate text using Visual Token Diffusion Language Model")
    
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint file (.pt)")
    parser.add_argument("--prompt", type=str, default="", 
                       help="Text prompt to seed generation (optional)")
    parser.add_argument("--length", type=int, default=50, 
                       help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, 
                       help="Sampling temperature (higher = more random)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run on")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.length <= 0:
        print("Error: Length must be positive")
        return
    if args.temperature <= 0:
        print("Error: Temperature must be positive")
        return
    
    # Generate samples
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n--- Sample {i+1}/{args.num_samples} ---")
        
        generated_text = generate_text(
            model_path=args.checkpoint,
            prompt=args.prompt,
            num_tokens=args.length,
            temperature=args.temperature,
            device=args.device
        )
        
        if generated_text:
            print(f"Generated: {generated_text}")
        else:
            print("Generation failed!")


if __name__ == "__main__":
    main()