"""
Encoder module for the Visual Token Diffusion Language Model.
Maps text tokens to visual patterns (5x5 grids with 3 colors).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import hashlib


class DeterministicEncoder:
    """
    Simple deterministic encoder that maps tokens to fixed visual patterns.
    This is a baseline approach for initial testing.
    """

    def __init__(self, token_to_id: Dict[str, int], num_colors: int = 3, grid_size: int = 5):
        """
        Initialize the deterministic encoder.

        Args:
            token_to_id: Dictionary mapping tokens to their IDs
            num_colors: Number of colors to use in the visual patterns
            grid_size: Size of the square grid for visual patterns
        """
        self.token_to_id = token_to_id
        self.id_to_token = {id: token for token, id in token_to_id.items()}
        self.num_colors = num_colors
        self.grid_size = grid_size

        # Generate fixed patterns for each token ID
        self.token_patterns = self._generate_token_patterns()

    def _generate_token_patterns(self) -> Dict[int, np.ndarray]:
        """
        Generate unique visual patterns for each token.
        Uses a hash-based approach to ensure patterns are deterministic and diverse.

        Returns:
            Dictionary mapping token IDs to their visual patterns
        """
        patterns = {}

        for token_id in self.id_to_token.keys():
            # Generate a seed based on the token ID
            seed = int(hashlib.md5(str(token_id).encode()).hexdigest(), 16) % 10000
            np.random.seed(seed)

            # Generate a random pattern
            pattern = np.random.randint(0, self.num_colors,
                                       (self.grid_size, self.grid_size))

            # Ensure special tokens have special patterns
            if token_id < 3:  # Special tokens: pad, unk, eos
                if token_id == 0:  # <pad>
                    pattern = np.zeros((self.grid_size, self.grid_size))
                elif token_id == 1:  # <unk>
                    pattern = np.ones((self.grid_size, self.grid_size))
                elif token_id == 2:  # <eos>
                    pattern = np.full((self.grid_size, self.grid_size), 2)

            patterns[token_id] = pattern

        return patterns

    def encode(self, token_ids: List[int]) -> np.ndarray:
        """
        Encode a sequence of token IDs into visual patterns.

        Args:
            token_ids: List of token IDs

        Returns:
            Array of shape [len(token_ids), grid_size, grid_size] containing the visual patterns
        """
        patterns = []

        for token_id in token_ids:
            if token_id in self.token_patterns:
                patterns.append(self.token_patterns[token_id])
            else:
                # Use unknown token pattern
                patterns.append(self.token_patterns[1])  # <unk>

        return np.array(patterns)

    def get_pattern(self, token: str) -> Optional[np.ndarray]:
        """
        Get the visual pattern for a specific token.

        Args:
            token: Token string

        Returns:
            Visual pattern for the token, or None if token is not in vocabulary
        """
        if token in self.token_to_id:
            token_id = self.token_to_id[token]
            return self.token_patterns[token_id]
        return None


class LearnableEncoder(nn.Module):
    """
    Neural network-based encoder that learns to map token embeddings to visual patterns.
    """

    def __init__(self, token_to_id: Dict[str, int], embedding_dim: int = 128,
                 hidden_dim: int = 256, num_colors: int = 3, grid_size: int = 5):
        """
        Initialize the learnable encoder.

        Args:
            token_to_id: Dictionary mapping tokens to their IDs
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of hidden layers
            num_colors: Number of colors to use in the visual patterns
            grid_size: Size of the square grid for visual patterns
        """
        super().__init__()
        self.token_to_id = token_to_id
        self.vocab_size = len(token_to_id)
        self.num_colors = num_colors
        self.grid_size = grid_size

        # Token embedding layer
        self.token_embedding = nn.Embedding(self.vocab_size, embedding_dim)

        # Dense layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, grid_size * grid_size * num_colors)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            token_ids: Tensor of token IDs, shape [batch_size]

        Returns:
            Tensor of visual patterns, shape [batch_size, grid_size, grid_size, num_colors]
        """
        # Get token embeddings
        embeddings = self.token_embedding(token_ids)  # [batch_size, embedding_dim]

        # Process through dense layers
        x = F.relu(self.fc1(embeddings))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape to [batch_size, grid_size, grid_size, num_colors]
        batch_size = token_ids.shape[0]
        x = x.view(batch_size, self.grid_size, self.grid_size, self.num_colors)

        # Apply softmax along the color dimension to get probabilities
        pattern_probs = F.softmax(x, dim=-1)

        return pattern_probs

    def sample_patterns(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Sample discrete patterns from the encoder outputs.

        Args:
            token_ids: Tensor of token IDs, shape [batch_size]

        Returns:
            Tensor of discrete patterns, shape [batch_size, grid_size, grid_size]
        """
        pattern_probs = self.forward(token_ids)  # [batch_size, grid_size, grid_size, num_colors]

        # Sample from the categorical distribution
        categorical = torch.distributions.Categorical(probs=pattern_probs)
        samples = categorical.sample()  # [batch_size, grid_size, grid_size]

        return samples

    def encode(self, token_ids: List[int], device: torch.device) -> np.ndarray:
        """
        Encode a sequence of token IDs into visual patterns.

        Args:
            token_ids: List of token IDs
            device: Device to run the encoding on

        Returns:
            Array of shape [len(token_ids), grid_size, grid_size] containing the visual patterns
        """
        # Convert to tensor
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

        # Get patterns (using hard sampling for discrete patterns)
        with torch.no_grad():
            patterns = self.sample_patterns(token_tensor)

        # Convert to numpy
        return patterns.cpu().numpy()
