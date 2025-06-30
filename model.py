# visual_token_diffuser/model.py

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
                 device: Optional[torch.device] = None,
                 # ADDED: Parameter to control training stage
                 initial_training_stage: str = "diffusion"):
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
            initial_training_stage: The stage to initialize the model in ('reconstruction' or 'diffusion')
        """
        self.token_to_id = token_to_id
        self.id_to_token = {id: token for token, id in token_to_id.items()}
        self.vocab_size = len(token_to_id)
        self.num_colors = num_colors
        self.grid_size = grid_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ADDED: Store the current training stage
        self.training_stage = initial_training_stage
        print(f"Initializing VisualTokenDiffusionLM in '{self.training_stage}' stage.")

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
        # Ensure decoder is learnable if encoder is, or if explicitly requested
        effective_decoder_type = decoder_type
        if encoder_type == "learnable" and decoder_type == "deterministic":
            print("Warning: LearnableEncoder selected but DeterministicDecoder requested. Forcing LearnableDecoder for reconstruction.")
            effective_decoder_type = "learnable"

        if effective_decoder_type == "deterministic" and encoder_type == "deterministic":
            # If using deterministic encoder, we can create a corresponding decoder
            self.decoder = DeterministicDecoder.from_encoder(self.encoder)
            self.is_decoder_learnable = False
        else:
            # Use a learnable decoder if encoder is learnable or decoder is explicitly learnable
            self.decoder = LearnableDecoder(
                self.id_to_token, self.vocab_size,
                hidden_dim=hidden_dim, num_colors=num_colors, grid_size=grid_size
            ).to(self.device)
            self.is_decoder_learnable = True

        # Assert compatibility for reconstruction stage
        if self.training_stage == 'reconstruction':
            assert self.is_encoder_learnable and self.is_decoder_learnable, \
                "Reconstruction stage requires both LearnableEncoder and LearnableDecoder."


    # ADDED: Method to change the training stage
    def set_training_stage(self, stage: str):
        """Sets the training stage ('reconstruction' or 'diffusion')."""
        assert stage in ['reconstruction', 'diffusion'], f"Invalid stage: {stage}"
        print(f"Switching training stage to '{stage}'")
        self.training_stage = stage
        if stage == 'reconstruction':
             assert self.is_encoder_learnable and self.is_decoder_learnable, \
                "Reconstruction stage requires both LearnableEncoder and LearnableDecoder."


    def _get_token_ids_from_text_batch(self, text_batch: List[str]) -> torch.Tensor:
        """Helper to convert a batch of text strings to a tensor of token IDs."""
        all_token_ids = []
        max_len = 0
        # Find max sequence length in batch for potential padding if needed later
        # Also collect all token IDs
        for text in text_batch:
            words = text.lower().split()
            token_ids = []
            for word in words:
                token_id = self.token_to_id.get(word, self.token_to_id.get("<unk>", 1)) # Default to <unk> ID 1
                token_ids.append(token_id)
            if not token_ids: # Handle empty strings if they occur
                token_ids.append(self.token_to_id.get("<pad>", 0)) # Use <pad> ID 0
            all_token_ids.append(torch.tensor(token_ids, dtype=torch.long))
            max_len = max(max_len, len(token_ids))

        # Combine into a single list (for now, assuming non-sequential processing)
        # Note: This flattens the batch structure, which might be okay for independent
        # token processing but bad for sequence modeling later.
        flat_token_ids = torch.cat(all_token_ids)
        return flat_token_ids.to(self.device)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to visual patterns (handling batching implicitly if learnable).

        Args:
            text: Input text

        Returns:
            Array of visual patterns
        """
        words = text.lower().split()
        token_ids = [self.token_to_id.get(word, self.token_to_id.get("<unk>", 1)) for word in words]
        if not token_ids: # Handle empty string
             token_ids = [self.token_to_id.get("<pad>", 0)]

        if self.is_encoder_learnable:
            # Ensure encoder is in eval mode for deterministic encoding if needed
            # Pass a batch of size 1
            initial_mode = self.encoder.training
            self.encoder.eval()
            with torch.no_grad():
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
                # Use sampling method from encoder
                patterns = self.encoder.sample_patterns(token_ids_tensor).cpu().numpy()
            self.encoder.train(initial_mode) # Restore previous mode
        else:
            patterns = self.encoder.encode(token_ids)

        return patterns # Shape [seq_len, grid_size, grid_size]

    def decode_patterns(self, patterns: np.ndarray) -> str:
        """
        Decode visual patterns back to text.

        Args:
            patterns: Array of visual patterns, shape [num_patterns, grid_size, grid_size]

        Returns:
            Decoded text
        """
        if not self.is_decoder_learnable:
            # Use DeterministicDecoder
            tokens = self.decoder.decode(patterns)
        else:
            # Use LearnableDecoder
            initial_mode = self.decoder.training
            self.decoder.eval()
            with torch.no_grad():
                # Process the whole sequence/batch at once
                patterns_tensor = torch.tensor(patterns, dtype=torch.long, device=self.device) # Decoder expects long if input is [B,G,G]
                # Use greedy decoding for simplicity here
                tokens = self.decoder.decode(patterns_tensor, greedy=True)
            self.decoder.train(initial_mode) # Restore previous mode

        return " ".join(tokens)

    def train_step(self, text_batch: List[str], optimizer, epoch: int = 0, num_epochs: int = 10) -> Dict[str, float]:
        """
        Perform a single training step based on the current training stage.

        Args:
            text_batch: Batch of text samples
            optimizer: PyTorch optimizer (assumed to be configured for the current stage)
            epoch: Current epoch for temperature annealing
            num_epochs: Total epochs for temperature annealing

        Returns:
            Dictionary of loss values for logging
        """
        def get_gumbel_temperature(epoch: int, num_epochs: int, start_temp: float = 5.0, end_temp: float = 0.5) -> float:
            """Calculate Gumbel-Softmax temperature with annealing schedule."""
            if num_epochs <= 1:
                return start_temp
            progress = epoch / (num_epochs - 1)
            return start_temp * (end_temp / start_temp) ** progress
        # Make sure models are in training mode
        if self.is_encoder_learnable: self.encoder.train()
        if self.is_decoder_learnable: self.decoder.train()
        self.diffusion.train() # Diffusion model should always be trainable if stage is diffusion

        optimizer.zero_grad()
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # --- Stage 1: Reconstruction (Autoencoder Training) ---
        if self.training_stage == 'reconstruction':
            if not (self.is_encoder_learnable and self.is_decoder_learnable):
                 raise RuntimeError("Reconstruction stage requires LearnableEncoder and LearnableDecoder.")

            # 1. Get Token IDs
            token_ids_tensor = self._get_token_ids_from_text_batch(text_batch) # Shape [total_tokens_in_batch]
            if token_ids_tensor.numel() == 0: # Skip empty batches
                return {"total_loss": 0.0}

            # 2. Encode Tokens to Differentiable Patterns using Gumbel-Softmax
            # Calculate temperature for current epoch (anneals from 5.0 to 0.5)
            temperature = get_gumbel_temperature(epoch, num_epochs)
            
            # Use Gumbel-Softmax for differentiable sampling - this maintains gradient flow!
            # Shape: [batch_size, grid_size, grid_size, num_colors]
            pattern_gumbel = self.encoder.forward_gumbel(token_ids_tensor, temperature=temperature)
            
            # Convert one-hot patterns to indices for decoder input
            # Shape: [batch_size, grid_size, grid_size]
            sampled_patterns = torch.argmax(pattern_gumbel, dim=-1)


            # 4. Decode Sampled Patterns to Token Probabilities
            # Decoder takes [batch, grid, grid] or [batch, grid, grid, num_colors]
            # The current decoder handles [batch, grid, grid] by applying one-hot itself.
            predicted_token_probs = self.decoder(sampled_patterns) # Shape: [total_tokens_in_batch, vocab_size]

            # 5. Calculate Reconstruction Loss
            reconstruction_loss = F.cross_entropy(predicted_token_probs, token_ids_tensor)
            total_loss = reconstruction_loss

            loss_dict["reconstruction_loss"] = reconstruction_loss.item()
            loss_dict["diffusion_loss"] = 0.0 # No diffusion loss in this stage


        # --- Stage 2: Diffusion Training (Original Logic) ---
        elif self.training_stage == 'diffusion':
            # 1. Encode Text Batch to Target Patterns (x_0)
            # For diffusion, we need the 'true' target patterns.
            # If using DeterministicEncoder, these are fixed.
            # If using LearnableEncoder, these are the patterns it *currently* produces.
            # We might want to use the encoder in eval mode and potentially detach patterns
            # if we are only training the diffusion model on fixed targets.
            # Let's assume we use the current encoder output as the target x_0.
            all_patterns_list = []
            all_token_ids_list = [] # Keep track of original tokens if decoder is learnable
            with torch.no_grad() if not self.is_encoder_learnable else torch.enable_grad(): # Only track grads if encoder is learnable and *we want to fine-tune it*
                for text in text_batch:
                    words = text.lower().split()
                    token_ids = [self.token_to_id.get(word, self.token_to_id.get("<unk>", 1)) for word in words]
                    if not token_ids: continue # Skip empty lines

                    if self.is_encoder_learnable:
                         # Use sampling to get discrete patterns (as done in encode_text)
                        patterns_np = self.encode_text(text) # Gets numpy array
                        patterns = torch.tensor(patterns_np, dtype=torch.long, device=self.device)
                    else:
                        # Deterministic encoder gives numpy array directly
                        patterns_np = self.encoder.encode(token_ids)
                        patterns = torch.tensor(patterns_np, dtype=torch.long, device=self.device)

                    all_patterns_list.append(patterns)
                    all_token_ids_list.extend(token_ids) # Store corresponding token IDs

            if not all_patterns_list: # Skip if batch resulted in no patterns
                 return {"total_loss": 0.0}

            # Stack patterns: shape [total_tokens_in_batch, grid_size, grid_size]
            patterns_tensor = torch.cat(all_patterns_list, dim=0)
            token_ids_tensor = torch.tensor(all_token_ids_list, dtype=torch.long, device=self.device)

            # 2. Forward pass through diffusion model
            # Diffusion model's forward now typically takes x_0 and returns predicted x_0 and noise/t
            # Let's assume diffusion forward does the noising and denoising prediction step:
            # predicted_probs_or_patterns = self.diffusion(patterns_tensor) # Adapting to AdvancedDiffusion output
            # Need to adjust based on specific diffusion model forward signature

            # Using the signature from AdvancedDiffusionModel's forward:
            # It needs patterns (x_0) and optionally timesteps t. It returns predicted *probabilities* for x_0.
            batch_size = patterns_tensor.shape[0]
            t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device)
            noisy_patterns = self.diffusion.add_noise(patterns_tensor, t) # Need add_noise separate or internal? Assume internal if not present.
            # Let's assume diffusion.forward takes original patterns and t, returns predicted x_0 probabilities
            # Need to simulate the typical diffusion training: sample t, noise x_0 -> x_t, predict x_0 from x_t
            # Reworking based on typical diffusion training loop structure:

            # a. Sample timesteps
            t = torch.randint(0, self.diffusion.timesteps, (patterns_tensor.shape[0],), device=self.device).long()

            # b. Noise the data to time t (using diffusion's add_noise method)
            noisy_patterns = self.diffusion.add_noise(patterns_tensor, t) # shape [B, G, G]

            # c. Get the model prediction (predicts probabilities of original patterns x_0)
            # The Advanced model forward expects patterns and t, let's pass the NOISY patterns and t
            # to predict the ORIGINAL pattern probabilities.
            predicted_pattern_probs = self.diffusion(noisy_patterns, t) # shape [B, G, G, C]

            # d. Calculate diffusion loss (compare predicted probs with original patterns)
            # Target should be the original patterns_tensor
            # Reshape for cross_entropy: prediction [N, C], target [N]
            diff_loss = F.cross_entropy(
                predicted_pattern_probs.reshape(-1, self.num_colors), # [B*G*G, C]
                patterns_tensor.reshape(-1).long() # [B*G*G]
            )
            total_loss += diff_loss
            loss_dict["diffusion_loss"] = diff_loss.item()

            # 3. Optional: Add Decoder Reconstruction Loss on Diffusion Output
            # This could help ensure generated patterns are decodable.
            # We could decode the *predicted* patterns from the diffusion model.
            if self.is_decoder_learnable:
                 # Option A: Decode the predicted probabilities (requires decoder to handle probs or sampling)
                 # Option B: Sample from predicted probabilities and decode samples
                 # Let's use sampling (again, non-differentiable link here if fine-tuning diffusion model based on this)
                 predicted_patterns_sampled = torch.distributions.Categorical(probs=predicted_pattern_probs).sample() # [B, G, G]
                 predicted_token_probs = self.decoder(predicted_patterns_sampled) # [B, vocab_size]
                 decoder_loss = F.cross_entropy(predicted_token_probs, token_ids_tensor) # Compare with original tokens
                 # Weight this loss? e.g., total_loss += 0.1 * decoder_loss
                 # For now, just log it. We could add it to total_loss later.
                 loss_dict["decoder_loss_from_diffusion"] = decoder_loss.item()
            else:
                loss_dict["decoder_loss_from_diffusion"] = 0.0


        else:
            raise ValueError(f"Unknown training stage: {self.training_stage}")


        # Backward pass and optimizer step (only if loss is calculated)
        if total_loss.requires_grad:
             # Check for NaN/inf gradients before stepping
             total_loss.backward()
             # Optional: Gradient clipping
             # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) # Need to define self.parameters() or pass relevant params
             optimizer.step()

        loss_dict["total_loss"] = total_loss.item()
        return loss_dict


    # --- Other methods (generate, save, load) remain largely the same ---
    # Note: Generate might need adjustment depending on whether encoder/decoder are learnable
    # and how conditioning is implemented. Save/load needs to handle the stage? Not necessarily.

    def generate(self, prompt: Optional[str] = None, max_length: int = 10,
                 temperature: float = 1.0) -> str:
        """
        Generate text using the diffusion model.
        (Note: Conditional generation from prompt is NOT implemented yet)

        Args:
            prompt: Optional text prompt (currently ignored for generation logic)
            max_length: Maximum number of *new* tokens to generate
            temperature: Temperature for sampling (higher = more random)

        Returns:
            Generated text
        """
        # Ensure models are in eval mode
        initial_modes = {}
        if self.is_encoder_learnable:
            initial_modes['encoder'] = self.encoder.training
            self.encoder.eval()
        if self.is_decoder_learnable:
            initial_modes['decoder'] = self.decoder.training
            self.decoder.eval()
        initial_modes['diffusion'] = self.diffusion.training
        self.diffusion.eval()

        generated_tokens = []
        if prompt:
            # TODO: Implement actual conditioning based on prompt encoding
            prompt_tokens = prompt.lower().split()
            print(f"Prompt received: '{prompt}' (Note: Conditioning not implemented, generating unconditionally)")
            generated_tokens.extend(prompt_tokens) # Start with prompt tokens for display only

        current_length = 0
        with torch.no_grad():
            for _ in range(max_length):
                # 1. Sample a new pattern from the diffusion model
                # Diffusion sample method returns discrete patterns [batch_size, grid_size, grid_size]
                sampled_patterns_tensor = self.diffusion.sample(batch_size=1, device=self.device, temperature=temperature)

                # 2. Decode the pattern to a token
                # Use decode_patterns which handles both decoder types
                # Need numpy array for decode_patterns
                sampled_patterns_np = sampled_patterns_tensor.cpu().numpy()
                # decode_patterns expects [num_patterns, G, G]
                decoded_token = self.decode_patterns(sampled_patterns_np)[0] # Decode the single pattern

                # Add to generated tokens
                generated_tokens.append(decoded_token)
                current_length += 1

                # Stop if EOS token is generated
                if decoded_token == "<eos>":
                    break

        # Restore initial modes
        if 'encoder' in initial_modes: self.encoder.train(initial_modes['encoder'])
        if 'decoder' in initial_modes: self.decoder.train(initial_modes['decoder'])
        if 'diffusion' in initial_modes: self.diffusion.train(initial_modes['diffusion'])

        # Join tokens to form text
        return " ".join(generated_tokens)


    def save(self, path: str) -> None:
        """
        Save the model state to disk. Includes component states and config.

        Args:
            path: Path to save the model
        """
        model_state = {
            "config": {
                "token_to_id": self.token_to_id,
                "encoder_type": "learnable" if self.is_encoder_learnable else "deterministic",
                "diffusion_type": type(self.diffusion).__name__, # Store class name
                "decoder_type": "learnable" if self.is_decoder_learnable else "deterministic",
                "num_colors": self.num_colors,
                "grid_size": self.grid_size,
                "diffusion_timesteps": getattr(self.diffusion, 'timesteps', None), # Get timesteps if available
                "hidden_dim": getattr(self.diffusion, 'hidden_dim', None) # Get hidden_dim if available
            },
            "diffusion_state": self.diffusion.state_dict(),
            # Save learnable components only if they exist
            "encoder_state": self.encoder.state_dict() if self.is_encoder_learnable else None,
            "decoder_state": self.decoder.state_dict() if self.is_decoder_learnable else None,
        }
        torch.save(model_state, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'VisualTokenDiffusionLM':
        """
        Load a model from disk.

        Args:
            path: Path to load the model from
            device: Device to load the model on

        Returns:
            Loaded model instance
        """
        map_location = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_state = torch.load(path, map_location=map_location)

        config = model_state["config"]

        # Determine hidden_dim and timesteps (handle potential missing keys for older saves)
        hidden_dim = config.get('hidden_dim', 256) # Default if missing
        diffusion_timesteps = config.get('diffusion_timesteps', 20 if 'Simple' in config['diffusion_type'] else 50) # Default based on type

        # Create model instance using saved config
        model = cls(
            token_to_id=config["token_to_id"],
            encoder_type=config["encoder_type"],
            diffusion_type='simple' if 'Simple' in config['diffusion_type'] else 'advanced', # Map class name back
            decoder_type=config["decoder_type"],
            num_colors=config["num_colors"],
            grid_size=config["grid_size"],
            diffusion_timesteps=diffusion_timesteps,
            hidden_dim=hidden_dim,
            device=map_location,
            # Load in diffusion stage by default, can be changed later
            initial_training_stage='diffusion'
        )

        # Load component states carefully
        model.diffusion.load_state_dict(model_state["diffusion_state"])

        if model.is_encoder_learnable and model_state["encoder_state"] is not None:
            model.encoder.load_state_dict(model_state["encoder_state"])
        elif model.is_encoder_learnable and model_state["encoder_state"] is None:
            print(f"Warning: Model configured with learnable encoder but no encoder state found in {path}.")

        if model.is_decoder_learnable and model_state["decoder_state"] is not None:
            model.decoder.load_state_dict(model_state["decoder_state"])
        elif model.is_decoder_learnable and model_state["decoder_state"] is None:
             print(f"Warning: Model configured with learnable decoder but no decoder state found in {path}.")

        print(f"Model loaded from {path}")
        return model
