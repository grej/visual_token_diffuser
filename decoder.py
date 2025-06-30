"""
Decoder module for the Visual Token Diffusion Language Model.
Maps visual patterns (5x5 grids with 3 colors) back to text.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class DeterministicDecoder:
    """
    Simple deterministic decoder that maps visual patterns back to tokens
    using the pre-defined patterns from the encoder.
    """
    
    def __init__(self, pattern_to_token_id: Dict[str, int], id_to_token: Dict[int, str]):
        """
        Initialize the deterministic decoder.
        
        Args:
            pattern_to_token_id: Dictionary mapping pattern strings to token IDs
            id_to_token: Dictionary mapping token IDs to tokens
        """
        self.pattern_to_token_id = pattern_to_token_id
        self.id_to_token = id_to_token
    
    def _pattern_to_string(self, pattern: np.ndarray) -> str:
        """
        Convert a pattern to a string for lookup.
        
        Args:
            pattern: Visual pattern array
        
        Returns:
            String representation of the pattern
        """
        # Flatten the pattern and convert to string
        return ','.join(map(str, pattern.flatten()))
    
    def decode_single(self, pattern: np.ndarray) -> str:
        """
        Decode a single pattern to a token.
        
        Args:
            pattern: Visual pattern to decode
        
        Returns:
            Decoded token, or <unk> if pattern is not recognized
        """
        pattern_str = self._pattern_to_string(pattern)
        
        if pattern_str in self.pattern_to_token_id:
            token_id = self.pattern_to_token_id[pattern_str]
            return self.id_to_token[token_id]
        
        # If exact pattern not found, find the closest match
        # (Simplified approach: for full implementation, would use a more efficient method)
        
        # Convert to arrays for easier comparison
        query_array = np.array(list(map(int, pattern_str.split(','))))
        
        best_match = None
        best_similarity = -1
        
        for p_str in self.pattern_to_token_id:
            p_array = np.array(list(map(int, p_str.split(','))))
            # Calculate similarity (number of matching positions)
            similarity = np.sum(query_array == p_array) / len(query_array)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = p_str
        
        if best_match and best_similarity > 0.8:  # Threshold for matching
            token_id = self.pattern_to_token_id[best_match]
            return self.id_to_token[token_id]
        
        return self.id_to_token.get(1, "<unk>")  # Default to <unk>
    
    def decode(self, patterns: np.ndarray) -> List[str]:
        """
        Decode a sequence of patterns to tokens.
        
        Args:
            patterns: Array of patterns, shape [sequence_length, grid_size, grid_size]
        
        Returns:
            List of decoded tokens
        """
        return [self.decode_single(pattern) for pattern in patterns]
    
    @classmethod
    def from_encoder(cls, encoder):
        """
        Create a decoder from an encoder instance.
        
        Args:
            encoder: DeterministicEncoder instance
        
        Returns:
            DeterministicDecoder instance
        """
        # Create mapping from pattern to token ID
        pattern_to_token_id = {}
        
        for token_id, pattern in encoder.token_patterns.items():
            pattern_str = ','.join(map(str, pattern.flatten()))
            pattern_to_token_id[pattern_str] = token_id
        
        return cls(pattern_to_token_id, encoder.id_to_token)


class LearnableDecoder(nn.Module):
    """
    Neural network-based decoder that learns to map visual patterns to token probabilities.
    Uses forced input dependence to prevent mode collapse.
    """
    
    def __init__(self, id_to_token: Dict[int, str], vocab_size: int, 
                 hidden_dim: int = 256, num_colors: int = 3, grid_size: int = 5):
        """
        Initialize the learnable decoder.
        
        Args:
            id_to_token: Dictionary mapping token IDs to tokens
            vocab_size: Size of the token vocabulary
            hidden_dim: Dimension of hidden layers
            num_colors: Number of colors in the visual patterns
            grid_size: Size of the grid for visual patterns
        """
        super().__init__()
        self.id_to_token = id_to_token
        self.vocab_size = vocab_size
        self.num_colors = num_colors
        self.grid_size = grid_size
        
        # Spatial processing path - preserves 2D structure
        self.conv1 = nn.Conv2d(num_colors, 64, kernel_size=1)  # 1x1 conv preserves spatial info
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP path
        self.fc1 = nn.Linear(256, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
        # Direct path - forces input dependence
        flattened_size = grid_size * grid_size * num_colors
        self.direct_path = nn.Linear(flattened_size, vocab_size)
        
        # Pattern hash embedding - creates discrete codes from continuous patterns
        self.pattern_hash = nn.Linear(flattened_size, hidden_dim)
        self.hash_to_vocab = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize with uniform prior (not zero!)
        import numpy as np
        self.fc2.bias.data.fill_(-np.log(vocab_size))
        self.direct_path.bias.data.fill_(-np.log(vocab_size))
        self.hash_to_vocab.bias.data.fill_(-np.log(vocab_size))
        
        print(f"NEW Decoder: Using forced input dependence architecture")
        print(f"Vocab size: {vocab_size}, Grid: {grid_size}x{grid_size}, Colors: {num_colors}")
        print(f"Initialized all output biases to uniform prior: {-np.log(vocab_size):.3f}")
    
    def forward(self, patterns: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with forced input dependence to prevent mode collapse.
        
        Args:
            patterns: Visual patterns, shape [batch_size, grid_size, grid_size]
                     or [batch_size, grid_size, grid_size, num_colors] if one-hot encoded
                     or continuous patterns [batch_size, grid_size, grid_size, num_colors] with sigmoid values
        
        Returns:
            Token log probabilities, shape [batch_size, vocab_size]
        """
        batch_size = patterns.shape[0]
        
        # Handle both continuous and discrete patterns
        if patterns.dim() == 3:
            # Discrete patterns - convert to one-hot encoding
            patterns = F.one_hot(patterns.long(), num_classes=self.num_colors).float()
        elif patterns.dim() == 4:
            # Already in the right format (continuous or one-hot)
            pass
        
        # Path 1: Spatial CNN processing
        # Reshape for conv: [batch, channels, height, width]
        conv_input = patterns.permute(0, 3, 1, 2)  # [batch, num_colors, grid_size, grid_size]
        
        x = F.relu(self.conv1(conv_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global pooling to get feature vector
        spatial_features = self.global_pool(x).squeeze(-1).squeeze(-1)  # [batch, 256]
        
        # MLP on spatial features
        mlp_out = F.relu(self.fc1(spatial_features))
        spatial_logits = self.fc2(mlp_out)
        
        # Path 2: Direct linear path (forces input dependence)
        flat_patterns = patterns.view(batch_size, -1)  # [batch, grid*grid*colors]
        direct_logits = self.direct_path(flat_patterns)
        
        # Path 3: Pattern hash embedding (discretizes continuous patterns)
        hash_features = torch.tanh(self.pattern_hash(flat_patterns))  # Bounded activation
        hash_logits = self.hash_to_vocab(hash_features)
        
        # Combine all paths - weighted ensemble
        # This FORCES the model to use input information from multiple perspectives
        combined_logits = (
            0.5 * spatial_logits +     # CNN spatial features
            0.3 * direct_logits +      # Direct input dependence
            0.2 * hash_logits          # Discrete pattern codes
        )
        
        # Return log probabilities for stable training
        return F.log_softmax(combined_logits, dim=-1)
    
    def decode(self, patterns: torch.Tensor, temperature: float = 1.0, 
               greedy: bool = False) -> List[str]:
        """
        Decode patterns to tokens.
        
        Args:
            patterns: Visual patterns, shape [batch_size, grid_size, grid_size]
                     or [batch_size, grid_size, grid_size, num_colors] if one-hot encoded
            temperature: Temperature parameter for sampling (higher values mean more random)
            greedy: If True, take the most likely token instead of sampling
        
        Returns:
            List of decoded tokens
        """
        # Get token probabilities
        token_probs = self.forward(patterns)
        
        if greedy:
            # Take the most likely token
            token_ids = torch.argmax(token_probs, dim=1)
        else:
            # Apply temperature scaling
            if temperature != 1.0:
                token_probs = token_probs.pow(1.0 / temperature)
                token_probs = token_probs / token_probs.sum(dim=1, keepdim=True)
            
            # Sample from the distribution
            token_ids = torch.multinomial(token_probs, num_samples=1).squeeze(-1)
        
        # Convert to list of tokens
        tokens = [self.id_to_token.get(id.item(), "<unk>") for id in token_ids]
        
        return tokens