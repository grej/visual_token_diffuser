"""
Diffusion model component for the Visual Token Diffusion Language Model.
Implements a discrete diffusion process for the visual token space.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class SimpleDiffusionModel(nn.Module):
    """
    A simple discrete diffusion model for visual tokens.
    Implements a Markovian diffusion process for discrete data.
    """
    
    def __init__(self, num_colors: int = 3, grid_size: int = 5, 
                 timesteps: int = 20, hidden_dim: int = 256):
        """
        Initialize the diffusion model.
        
        Args:
            num_colors: Number of colors in the visual patterns
            grid_size: Size of the grid for visual patterns
            timesteps: Number of diffusion timesteps
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        self.num_colors = num_colors
        self.grid_size = grid_size
        self.timesteps = timesteps
        
        # Define the noise schedule (linear beta schedule)
        self.beta = torch.linspace(0.1, 0.9, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
        # Define the transition matrix
        self.transition_matrix = self._create_transition_matrix()
        
        # Define the denoising network
        self.denoising_network = self._create_denoising_network(hidden_dim)
    
    def _create_transition_matrix(self) -> torch.Tensor:
        """
        Create the transition matrix for the forward process.
        Each row sums to 1 and represents the probability of transitioning
        from one color to another when adding noise.
        
        Returns:
            Transition matrix of shape [timesteps, num_colors, num_colors]
        """
        transition_matrix = torch.zeros(self.timesteps, self.num_colors, self.num_colors)
        
        for t in range(self.timesteps):
            # As t increases, we move closer to a uniform distribution
            p_stay = self.alpha_cumprod[t]
            p_change = (1 - p_stay) / (self.num_colors - 1)
            
            # Fill the transition matrix
            for i in range(self.num_colors):
                for j in range(self.num_colors):
                    if i == j:
                        transition_matrix[t, i, j] = p_stay
                    else:
                        transition_matrix[t, i, j] = p_change
        
        return transition_matrix
    
    def _create_denoising_network(self, hidden_dim: int) -> nn.Module:
        """
        Create the denoising network that predicts the original pattern
        from a noisy pattern at a given timestep.
        
        Args:
            hidden_dim: Dimension of hidden layers
        
        Returns:
            Neural network module
        """
        denoising_net = nn.Sequential(
            # Input: noisy pattern flattened + timestep embedding
            nn.Linear(self.grid_size * self.grid_size * self.num_colors + self.timesteps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.grid_size * self.grid_size * self.num_colors)
        )
        
        return denoising_net
    
    def add_noise(self, patterns: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the patterns according to the forward process.
        
        Args:
            patterns: Original patterns, shape [batch_size, grid_size, grid_size]
            t: Timesteps for each sample in the batch, shape [batch_size]
        
        Returns:
            Noisy patterns, shape [batch_size, grid_size, grid_size]
        """
        batch_size = patterns.shape[0]
        device = patterns.device
        
        # Create one-hot encoded patterns
        patterns_one_hot = F.one_hot(patterns.long(), num_classes=self.num_colors)
        # Shape: [batch_size, grid_size, grid_size, num_colors]
        
        # Get transition matrices for each timestep in the batch
        batch_transitions = torch.stack([self.transition_matrix[t[i]] for i in range(batch_size)])
        # Shape: [batch_size, num_colors, num_colors]
        
        # Apply transition probabilities
        patterns_flat = patterns_one_hot.view(batch_size, -1, self.num_colors)
        # Shape: [batch_size, grid_size*grid_size, num_colors]
        
        # For each position, multiply by the transition matrix
        # This gives the probability of transitioning to each color
        transition_probs = torch.bmm(patterns_flat, batch_transitions)
        # Shape: [batch_size, grid_size*grid_size, num_colors]
        
        # Sample from the categorical distribution
        noisy_patterns_flat = torch.multinomial(
            transition_probs.view(-1, self.num_colors),
            num_samples=1
        ).view(batch_size, self.grid_size * self.grid_size)
        
        # Reshape back to grid
        noisy_patterns = noisy_patterns_flat.view(batch_size, self.grid_size, self.grid_size)
        
        return noisy_patterns
    
    def denoise(self, noisy_patterns: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Denoise the patterns according to the reverse process.
        
        Args:
            noisy_patterns: Noisy patterns, shape [batch_size, grid_size, grid_size]
            t: Timesteps for each sample in the batch, shape [batch_size]
        
        Returns:
            Predicted original patterns, shape [batch_size, grid_size, grid_size]
        """
        batch_size = noisy_patterns.shape[0]
        device = noisy_patterns.device
        
        # One-hot encode noisy patterns
        noisy_one_hot = F.one_hot(noisy_patterns.long(), num_classes=self.num_colors)
        # Shape: [batch_size, grid_size, grid_size, num_colors]
        
        # Flatten the one-hot patterns
        noisy_flat = noisy_one_hot.view(batch_size, -1)
        # Shape: [batch_size, grid_size*grid_size*num_colors]
        
        # Create timestep embeddings (one-hot)
        t_emb = F.one_hot(t.long(), num_classes=self.timesteps).float()
        # Shape: [batch_size, timesteps]
        
        # Concatenate noisy patterns and timestep embeddings
        network_input = torch.cat([noisy_flat, t_emb], dim=1)
        
        # Pass through the denoising network
        network_output = self.denoising_network(network_input)
        # Shape: [batch_size, grid_size*grid_size*num_colors]
        
        # Reshape to [batch_size, grid_size, grid_size, num_colors]
        predicted_probs = network_output.view(
            batch_size, self.grid_size, self.grid_size, self.num_colors
        )
        
        # Apply softmax to get probabilities
        predicted_probs = F.softmax(predicted_probs, dim=-1)
        
        # Get the most likely color at each position
        predicted_patterns = torch.argmax(predicted_probs, dim=-1)
        
        return predicted_patterns
    
    def forward(self, patterns: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the diffusion model.
        
        Args:
            patterns: Original patterns, shape [batch_size, grid_size, grid_size]
        
        Returns:
            Tuple of (noisy_patterns, predicted_patterns, t)
        """
        batch_size = patterns.shape[0]
        device = patterns.device
        
        # Sample random timesteps for each sample in the batch
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Add noise to the patterns
        noisy_patterns = self.add_noise(patterns, t)
        
        # Predict original patterns from noisy ones
        predicted_patterns = self.denoise(noisy_patterns, t)
        
        return noisy_patterns, predicted_patterns, t
    
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample new patterns from random noise using the reverse process.
        
        Args:
            batch_size: Number of patterns to sample
            device: Device to run the sampling on
        
        Returns:
            Sampled patterns, shape [batch_size, grid_size, grid_size]
        """
        # Start with random patterns
        patterns = torch.randint(
            0, self.num_colors, 
            (batch_size, self.grid_size, self.grid_size),
            device=device
        )
        
        # Iteratively denoise
        for t in range(self.timesteps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device)
            patterns = self.denoise(patterns, t_batch)
        
        return patterns


class AdvancedDiffusionModel(nn.Module):
    """
    More advanced discrete diffusion model for visual tokens.
    Includes attention mechanisms and improved network architecture.
    """
    
    def __init__(self, num_colors: int = 3, grid_size: int = 5, 
                 timesteps: int = 50, hidden_dim: int = 512, 
                 num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize the advanced diffusion model.
        
        Args:
            num_colors: Number of colors in the visual patterns
            grid_size: Size of the grid for visual patterns
            timesteps: Number of diffusion timesteps
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.num_colors = num_colors
        self.grid_size = grid_size
        self.timesteps = timesteps
        self.hidden_dim = hidden_dim
        
        # Define the noise schedule (cosine schedule)
        self.beta = self._cosine_beta_schedule(timesteps)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
        # Define the transition matrix
        self.register_buffer('transition_matrix', self._create_transition_matrix())
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(timesteps, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Input projection
        self.input_proj = nn.Linear(self.grid_size * self.grid_size * self.num_colors, hidden_dim)
        
        # Transformer blocks
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.grid_size * self.grid_size * self.num_colors)
    
    def _cosine_beta_schedule(self, timesteps: int) -> torch.Tensor:
        """
        Create a cosine noise schedule.
        
        Args:
            timesteps: Number of diffusion timesteps
        
        Returns:
            Beta schedule of shape [timesteps]
        """
        steps = timesteps + 1
        s = 0.008
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _create_transition_matrix(self) -> torch.Tensor:
        """
        Create the transition matrix for the forward process.
        
        Returns:
            Transition matrix of shape [timesteps, num_colors, num_colors]
        """
        transition_matrix = torch.zeros(self.timesteps, self.num_colors, self.num_colors)
        
        for t in range(self.timesteps):
            # As t increases, we move closer to a uniform distribution
            p_stay = self.alpha_cumprod[t]
            p_change = (1 - p_stay) / (self.num_colors - 1)
            
            # Fill the transition matrix
            for i in range(self.num_colors):
                for j in range(self.num_colors):
                    if i == j:
                        transition_matrix[t, i, j] = p_stay
                    else:
                        transition_matrix[t, i, j] = p_change
        
        return transition_matrix
    
    def add_noise(self, patterns: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the patterns according to the forward process.
        
        Args:
            patterns: Original patterns, shape [batch_size, grid_size, grid_size]
            t: Timesteps for each sample in the batch, shape [batch_size]
        
        Returns:
            Noisy patterns, shape [batch_size, grid_size, grid_size]
        """
        batch_size = patterns.shape[0]
        device = patterns.device
        
        # Create one-hot encoded patterns
        patterns_one_hot = F.one_hot(patterns.long(), num_classes=self.num_colors)
        # Shape: [batch_size, grid_size, grid_size, num_colors]
        
        # Get transition matrices for each timestep in the batch
        batch_transitions = torch.stack([self.transition_matrix[t[i]] for i in range(batch_size)])
        # Shape: [batch_size, num_colors, num_colors]
        
        # Apply transition probabilities
        patterns_flat = patterns_one_hot.reshape(batch_size, -1, self.num_colors)
        # Shape: [batch_size, grid_size*grid_size, num_colors]
        
        # For each position, multiply by the transition matrix
        transition_probs = torch.bmm(patterns_flat, batch_transitions)
        # Shape: [batch_size, grid_size*grid_size, num_colors]
        
        # Sample from the categorical distribution
        noisy_patterns_flat = torch.multinomial(
            transition_probs.reshape(-1, self.num_colors),
            num_samples=1
        ).reshape(batch_size, self.grid_size * self.grid_size)
        
        # Reshape back to grid
        noisy_patterns = noisy_patterns_flat.reshape(batch_size, self.grid_size, self.grid_size)
        
        return noisy_patterns
    
    def forward(self, patterns: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the diffusion model.
        
        Args:
            patterns: Original patterns, shape [batch_size, grid_size, grid_size]
            t: Optional timesteps for each sample in the batch, shape [batch_size]
                If None, random timesteps will be sampled.
        
        Returns:
            Predicted denoised patterns, shape [batch_size, grid_size, grid_size]
        """
        batch_size = patterns.shape[0]
        device = patterns.device
        
        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Add noise to the patterns
        noisy_patterns = self.add_noise(patterns, t)
        
        # One-hot encode noisy patterns
        noisy_one_hot = F.one_hot(noisy_patterns.long(), num_classes=self.num_colors).float()
        # Shape: [batch_size, grid_size, grid_size, num_colors]
        
        # Flatten the one-hot patterns
        noisy_flat = noisy_one_hot.reshape(batch_size, -1)
        # Shape: [batch_size, grid_size*grid_size*num_colors]
        
        # Create timestep embeddings (one-hot)
        t_emb = F.one_hot(t.long(), num_classes=self.timesteps).float()
        # Shape: [batch_size, timesteps]
        
        # Embed time
        time_emb = self.time_embed(t_emb)
        # Shape: [batch_size, hidden_dim]
        
        # Project input to hidden dimension
        x = self.input_proj(noisy_flat)
        # Shape: [batch_size, hidden_dim]
        
        # Add time embedding
        x = x + time_emb
        
        # Self-attention
        x_norm = self.norm1(x.unsqueeze(0))  # [1, batch_size, hidden_dim]
        attention_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attention_output.squeeze(0)
        
        # Feed-forward
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        x = x + ff_output
        
        # Project to output
        output = self.output_proj(x)
        # Shape: [batch_size, grid_size*grid_size*num_colors]
        
        # Reshape to [batch_size, grid_size, grid_size, num_colors]
        predicted_probs = output.reshape(batch_size, self.grid_size, self.grid_size, self.num_colors)
        
        # Apply softmax to get probabilities
        predicted_probs = F.softmax(predicted_probs, dim=-1)
        
        return predicted_probs
    
    def sample(self, batch_size: int, device: torch.device, 
               temperature: float = 1.0) -> torch.Tensor:
        """
        Sample new patterns from random noise using the reverse process.
        
        Args:
            batch_size: Number of patterns to sample
            device: Device to run the sampling on
            temperature: Temperature for sampling (higher = more random)
        
        Returns:
            Sampled patterns, shape [batch_size, grid_size, grid_size]
        """
        # Start with random patterns
        patterns = torch.randint(
            0, self.num_colors, 
            (batch_size, self.grid_size, self.grid_size),
            device=device
        )
        
        # Iteratively denoise
        for t in range(self.timesteps - 1, -1, -1):
            # Create timestep batch
            t_batch = torch.full((batch_size,), t, device=device)
            
            # Get predicted probabilities
            pred_probs = self.forward(patterns, t_batch)
            
            # Apply temperature
            if temperature != 1.0:
                pred_probs = pred_probs.pow(1.0 / temperature)
                pred_probs = pred_probs / pred_probs.sum(dim=-1, keepdim=True)
            
            # Sample from the predicted distribution
            if t > 0:  # For intermediate steps, sample
                patterns_flat = pred_probs.reshape(-1, self.num_colors)
                sampled_flat = torch.multinomial(patterns_flat, num_samples=1)
                patterns = sampled_flat.reshape(batch_size, self.grid_size, self.grid_size)
            else:  # For the final step, take the most likely color
                patterns = torch.argmax(pred_probs, dim=-1)
        
        return patterns