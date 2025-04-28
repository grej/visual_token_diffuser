"""
Main model for the Visual Token Diffusion Language Model.
Integrates the encoder, diffusion model, and decoder components.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# Import components
from encoder import DeterministicEncoder, LearnableEncoder
from diffusion import SimpleDiffusionModel, AdvancedDiffusionModel
from decoder import DeterministicDecoder, LearnableDecoder


class VisualTokenDiffusionLM:
    """
    Complete Visual Token Diffusion Language Model.
    Combines encoder, diffusion, and decoder components.
    """

    def __init__(self, token_to_id: Dict[str, int],
                 encoder_type: str = "deterministic",
                 diffusion_type: str = "simple",
                 decoder_type: str = "deterministic",
                 num_colors: int = 3,
                 grid_size: int = 5,
                 diffusion_timesteps: int = 20,
                 hidden_dim: int = 256,
                 device: Optional[torch.device] = None):
        """
        Initialize the Visual Token Diffusion LM.

        Args:
            token_to_id: Dictionary mapping tokens to their IDs
            encoder_type: Type of encoder ("deterministic" or "learnable")
            diffusion_type: Type of diffusion model ("simple" or "advanced")
            decoder_type: Type of decoder ("deterministic" or "learnable")
            num_colors: Number of colors in the visual patterns
            grid_size: Size of the grid for visual patterns
            diffusion_timesteps: Number of diffusion timesteps
            hidden_dim: Dimension of hidden layers
            device: Device to run the model on (CPU or GPU)
        """
        self.token_to_id = token_to_id
        self.id_to_token = {id: token for token, id in token_to_id.items()}
        self.vocab_size = len(token_to_id)
        self.num_colors = num_colors
        self.grid_size = grid_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize encoder
        if encoder_type == "deterministic":
            self.encoder = DeterministicEncoder(
                token_to_id, num_colors=num_colors, grid_size=grid_size
            )
            self.is_encoder_learnable = False
        else:
            self.encoder = LearnableEncoder(
                token_to_id, hidden_dim=hidden_dim,
                num_colors=num_colors, grid_size=grid_size
            ).to(self.device)
            self.is_encoder_learnable = True

        # Initialize diffusion model
        if diffusion_type == "simple":
            self.diffusion = SimpleDiffusionModel(
                num_colors=num_colors, grid_size=grid_size,
                timesteps=diffusion_timesteps, hidden_dim=hidden_dim
            ).to(self.device)
        else:
            self.diffusion = AdvancedDiffusionModel(
                num_colors=num_colors, grid_size=grid_size,
                timesteps=diffusion_timesteps, hidden_dim=hidden_dim
            ).to(self.device)

        # Initialize decoder
        if decoder_type == "deterministic" and encoder_type == "deterministic":
            # If using deterministic encoder, we can create a corresponding decoder
            self.decoder = DeterministicDecoder.from_encoder(self.encoder)
            self.is_decoder_learnable = False
        else:
            # Otherwise, use a learnable decoder
            self.decoder = LearnableDecoder(
                self.id_to_token, self.vocab_size,
                hidden_dim=hidden_dim, num_colors=num_colors, grid_size=grid_size
            ).to(self.device)
            self.is_decoder_learnable = True

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to visual patterns.

        Args:
            text: Input text

        Returns:
            Array of visual patterns
        """
        # Tokenize
        words = text.lower().split()
        token_ids = []

        for word in words:
            if word in self.token_to_id:
                token_ids.append(self.token_to_id[word])
            else:
                # Use unknown token
                token_ids.append(self.token_to_id.get("<unk>", 0))

        # Encode
        if self.is_encoder_learnable:
            patterns = self.encoder.encode(token_ids, self.device)
        else:
            patterns = self.encoder.encode(token_ids)

        return patterns

    def decode_patterns(self, patterns: np.ndarray) -> str:
        """
        Decode visual patterns back to text.

        Args:
            patterns: Array of visual patterns

        Returns:
            Decoded text
        """
        if self.is_decoder_learnable:
            # Convert to tensor
            patterns_tensor = torch.tensor(patterns, device=self.device)
            tokens = self.decoder.decode(patterns_tensor, greedy=True)
        else:
            tokens = self.decoder.decode(patterns)

        # Join tokens to form text
        return " ".join(tokens)

    def train_step(self, text_batch: List[str], optimizer) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            text_batch: Batch of text samples
            optimizer: PyTorch optimizer

        Returns:
            Dictionary of loss values
        """
        # Process batch
        all_patterns = []
        for text in text_batch:
            patterns = self.encode_text(text)
            all_patterns.append(patterns)

        # Stack patterns
        patterns_batch = np.vstack(all_patterns)
        patterns_tensor = torch.tensor(patterns_batch, device=self.device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass through diffusion model
        noisy_patterns, predicted_patterns, t = self.diffusion(patterns_tensor)

        # Calculate diffusion loss
        diff_loss = F.cross_entropy(
            predicted_patterns.reshape(-1, self.num_colors),
            patterns_tensor.reshape(-1).long()
        )

        # If using learnable encoder/decoder, calculate reconstruction loss
        if self.is_encoder_learnable or self.is_decoder_learnable:
            # Get token IDs from text
            all_token_ids = []
            for text in text_batch:
                words = text.lower().split()
                token_ids = []
                for word in words:
                    if word in self.token_to_id:
                        token_ids.append(self.token_to_id[word])
                    else:
                        token_ids.append(self.token_to_id.get("<unk>", 0))
                all_token_ids.extend(token_ids)

            token_ids_tensor = torch.tensor(all_token_ids, device=self.device)

            # If using learnable decoder, compute decoder loss
            if self.is_decoder_learnable:
                token_probs = self.decoder(patterns_tensor)
                decoder_loss = F.cross_entropy(token_probs, token_ids_tensor)
            else:
                decoder_loss = torch.tensor(0.0, device=self.device)

            # If using learnable encoder, compute encoder loss
            if self.is_encoder_learnable:
                # This would be more complex, involving both the encoder and decoder
                # For simplicity, we'll skip it in this implementation
                encoder_loss = torch.tensor(0.0, device=self.device)
            else:
                encoder_loss = torch.tensor(0.0, device=self.device)

            # Combine losses
            total_loss = diff_loss + decoder_loss + encoder_loss
        else:
            # Only diffusion loss
            total_loss = diff_loss
            decoder_loss = torch.tensor(0.0, device=self.device)
            encoder_loss = torch.tensor(0.0, device=self.device)

        # Backward pass
        total_loss.backward()

        # Update weights
        optimizer.step()

        # Return losses
        return {
            "total_loss": total_loss.item(),
            "diffusion_loss": diff_loss.item(),
            "decoder_loss": decoder_loss.item(),
            "encoder_loss": encoder_loss.item()
        }

    def generate(self, prompt: Optional[str] = None, max_length: int = 10,
                 temperature: float = 1.0) -> str:
        """
        Generate text using the diffusion model.

        Args:
            prompt: Optional text prompt to condition the generation
            max_length: Maximum number of tokens to generate
            temperature: Temperature for sampling (higher = more random)

        Returns:
            Generated text
        """
        # If prompt is provided, encode it
        if prompt:
            prompt_patterns = self.encode_text(prompt)
            prompt_tokens = prompt.lower().split()
            generated_tokens = prompt_tokens.copy()
        else:
            prompt_patterns = None
            generated_tokens = []

        # Generate new tokens until max_length
        for _ in range(max_length):
            # Sample a new pattern from the diffusion model
            if prompt_patterns is not None:
                # TODO: Implement conditional generation
                # For now, we'll just generate unconditionally
                patterns = self.diffusion.sample(1, self.device, temperature)
            else:
                patterns = self.diffusion.sample(1, self.device, temperature)

            # Decode to tokens
            if self.is_decoder_learnable:
                new_token = self.decoder.decode(patterns, temperature)[0]
            else:
                new_token = self.decoder.decode(patterns.cpu().numpy())[0]

            # Add to generated tokens
            generated_tokens.append(new_token)

            # Stop if EOS token is generated
            if new_token == "<eos>":
                break

        # Join tokens to form text
        return " ".join(generated_tokens)

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        model_state = {
            "token_to_id": self.token_to_id,
            "encoder_type": "deterministic" if not self.is_encoder_learnable else "learnable",
            "decoder_type": "deterministic" if not self.is_decoder_learnable else "learnable",
            "diffusion_state": self.diffusion.state_dict(),
            "num_colors": self.num_colors,
            "grid_size": self.grid_size,
        }

        if self.is_encoder_learnable:
            model_state["encoder_state"] = self.encoder.state_dict()

        if self.is_decoder_learnable:
            model_state["decoder_state"] = self.decoder.state_dict()

        torch.save(model_state, path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'VisualTokenDiffusionLM':
        """
        Load a model from disk.

        Args:
            path: Path to load the model from
            device: Device to load the model on

        Returns:
            Loaded model
        """
        model_state = torch.load(path, map_location=device)

        # Create model
        model = cls(
            token_to_id=model_state["token_to_id"],
            encoder_type=model_state["encoder_type"],
            decoder_type=model_state["decoder_type"],
            num_colors=model_state["num_colors"],
            grid_size=model_state["grid_size"],
            device=device
        )

        # Load component states
        model.diffusion.load_state_dict(model_state["diffusion_state"])

        if model.is_encoder_learnable and "encoder_state" in model_state:
            model.encoder.load_state_dict(model_state["encoder_state"])

        if model.is_decoder_learnable and "decoder_state" in model_state:
            model.decoder.load_state_dict(model_state["decoder_state"])

        return model
