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
        
        # CNN layers to process the visual patterns
        self.conv1 = nn.Conv2d(num_colors, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Calculate the flattened size (bypassing CNN)
        flattened_size = grid_size * grid_size * num_colors
        
        # Dense layers
        self.fc1 = nn.Linear(flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, patterns: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            patterns: Visual patterns, shape [batch_size, grid_size, grid_size]
                     or [batch_size, grid_size, grid_size, num_colors] if one-hot encoded
        
        Returns:
            Token probabilities, shape [batch_size, vocab_size]
        """
        batch_size = patterns.shape[0]
        
        # DEBUG: Check decoder input
        if hasattr(patterns, 'requires_grad') and patterns.requires_grad:
            print(f"DEBUG Decoder: input shape={patterns.shape}, requires_grad={patterns.requires_grad}")
            print(f"DEBUG Decoder: input sample (first 5 values)={patterns[0].flatten()[:5].detach()}")
        
        # Convert to one-hot encoding if not already
        if patterns.dim() == 3:
            patterns = F.one_hot(patterns.long(), num_classes=self.num_colors).float()
            # Shape: [batch_size, grid_size, grid_size, num_colors]
        
        # BYPASS CNN - directly flatten the one-hot patterns
        # CNN doesn't work well with sparse one-hot patterns  
        x = patterns.view(batch_size, -1)
        # Shape: [batch_size, grid_size*grid_size*num_colors]
        
        # Apply dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Apply softmax to get probabilities
        token_probs = F.softmax(x, dim=-1)
        
        return token_probs
    
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