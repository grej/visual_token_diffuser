"""
Utility functions for the Visual Token Diffusion Language Model.
Includes visualization, token handling, and data processing utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F


def visualize_pattern(pattern: np.ndarray, title: Optional[str] = None) -> None:
    """
    Visualize a 5x5 pattern with 3 possible colors.
    
    Args:
        pattern: 5x5 numpy array with values in [0, 1, 2]
        title: Optional title for the plot
    """
    # Map the values to colors: 0->white, 1->blue, 2->red
    cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'red'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(pattern, cmap=cmap, norm=norm)
    plt.grid(True, which='both', color='black', linewidth=1.5)
    
    # Add the numeric values in each cell
    for i in range(5):
        for j in range(5):
            plt.text(j, i, str(int(pattern[i, j])), 
                    ha="center", va="center", color="black", fontweight='bold')
    
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_pattern_batch(patterns: np.ndarray, tokens: List[str] = None, max_display: int = 10) -> None:
    """
    Visualize a batch of 5x5 patterns.
    
    Args:
        patterns: Batch of 5x5 patterns, shape [batch_size, 5, 5]
        tokens: Optional list of tokens corresponding to the patterns
        max_display: Maximum number of patterns to display
    """
    batch_size = min(len(patterns), max_display)
    rows = (batch_size + 4) // 5  # Ceiling division to determine number of rows
    
    cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'red'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(15, 3 * rows))
    for i in range(batch_size):
        plt.subplot(rows, 5, i + 1)
        plt.imshow(patterns[i], cmap=cmap, norm=norm)
        plt.grid(True, which='both', color='black', linewidth=1.5)
        
        if tokens:
            plt.title(tokens[i])
    
    plt.tight_layout()
    plt.show()


def encode_text(text: str, tokenizer: Dict[str, int]) -> List[int]:
    """
    Encode text into a list of token IDs.
    
    Args:
        text: Input text
        tokenizer: Dictionary mapping tokens to their IDs
    
    Returns:
        List of token IDs
    """
    # Simple word-level tokenization for prototype
    words = text.lower().split()
    token_ids = []
    
    for word in words:
        if word in tokenizer:
            token_ids.append(tokenizer[word])
        else:
            # Handle unknown tokens - for now we'll use a designated OOV token
            token_ids.append(tokenizer.get("<unk>", 0))
    
    return token_ids


def create_simple_tokenizer(vocab: List[str]) -> Dict[str, int]:
    """
    Create a simple word-level tokenizer.
    
    Args:
        vocab: List of words to include in the vocabulary
    
    Returns:
        Dictionary mapping tokens to their IDs
    """
    # Add special tokens
    special_tokens = ["<pad>", "<unk>", "<eos>"]
    all_tokens = special_tokens + vocab
    
    # Create token to ID mapping
    token_to_id = {token: idx for idx, token in enumerate(all_tokens)}
    
    return token_to_id


def hamming_distance(pattern1: np.ndarray, pattern2: np.ndarray) -> int:
    """
    Compute the Hamming distance between two patterns.
    
    Args:
        pattern1: First pattern
        pattern2: Second pattern
    
    Returns:
        Hamming distance (number of positions where patterns differ)
    """
    return np.sum(pattern1 != pattern2)


def pattern_similarity(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
    """
    Compute a similarity score between two patterns.
    
    Args:
        pattern1: First pattern
        pattern2: Second pattern
    
    Returns:
        Similarity score in [0, 1], where 1 means identical patterns
    """
    total_positions = pattern1.size
    matching_positions = np.sum(pattern1 == pattern2)
    return matching_positions / total_positions


def cosine_similarity_batch(patterns1: torch.Tensor, patterns2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between batches of patterns.
    
    Args:
        patterns1: First batch of patterns, shape [batch_size, 5, 5]
        patterns2: Second batch of patterns, shape [batch_size, 5, 5]
    
    Returns:
        Tensor of cosine similarities, shape [batch_size]
    """
    # Reshape to [batch_size, 25]
    batch_size = patterns1.shape[0]
    flat1 = patterns1.reshape(batch_size, -1)
    flat2 = patterns2.reshape(batch_size, -1)
    
    # Compute cosine similarity
    return F.cosine_similarity(flat1, flat2, dim=1)


def compute_entropy(pattern: np.ndarray) -> float:
    """
    Compute the entropy of a pattern.
    
    Args:
        pattern: 5x5 pattern with values in [0, 1, 2]
    
    Returns:
        Entropy value
    """
    # Count occurrences of each value
    values, counts = np.unique(pattern, return_counts=True)
    probabilities = counts / np.sum(counts)
    
    # Compute entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


# Sample function for creating pattern space statistics
def analyze_pattern_space(patterns: List[np.ndarray]) -> Dict:
    """
    Analyze various statistics about the pattern space.
    
    Args:
        patterns: List of 5x5 patterns
    
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Compute average entropy
    entropies = [compute_entropy(p) for p in patterns]
    stats['avg_entropy'] = np.mean(entropies)
    stats['min_entropy'] = np.min(entropies)
    stats['max_entropy'] = np.max(entropies)
    
    # Compute pairwise Hamming distances
    distances = []
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            distances.append(hamming_distance(patterns[i], patterns[j]))
    
    if distances:
        stats['avg_distance'] = np.mean(distances)
        stats['min_distance'] = np.min(distances)
        stats['max_distance'] = np.max(distances)
    
    # Analyze color distribution
    all_patterns = np.vstack([p.flatten() for p in patterns])
    color_counts = np.bincount(all_patterns.astype(int), minlength=3)
    stats['color_distribution'] = color_counts / np.sum(color_counts)
    
    return stats