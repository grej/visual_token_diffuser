#!/usr/bin/env python3
"""
Topological Visual Embeddings: Semantic-to-Visual Embedding Model

Core Innovation: Train an embedding model to create topologically meaningful 
images that correspond to concepts, where visual similarity = semantic similarity.

This is no longer "arbitrary visual patterns for tokens" but rather 
"semantically grounded visual representations that preserve meaning in visual space."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import json

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class LearnedTopologicalProjection(nn.Module):
    """Learn to project semantic embeddings to topology-preserving 2D space"""
    
    def __init__(self, semantic_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        # Neural network that learns UMAP-like projection
        self.projector = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()  # Bounded to [-1, 1]
        )
        
        # Temperature for neighborhood preservation
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, semantic_embeddings: torch.Tensor) -> torch.Tensor:
        """Project to 2D while preserving local topology"""
        positions_2d = self.projector(semantic_embeddings)
        return positions_2d
    
    def neighborhood_preservation_loss(self, semantic_embeddings: torch.Tensor, 
                                     positions_2d: torch.Tensor,
                                     k: int = 5) -> torch.Tensor:
        """Loss that encourages semantic neighbors to be spatial neighbors"""
        
        batch_size = semantic_embeddings.shape[0]
        
        # Compute semantic similarities
        semantic_sims = F.cosine_similarity(
            semantic_embeddings.unsqueeze(1), 
            semantic_embeddings.unsqueeze(0), 
            dim=2
        )
        
        # Compute spatial distances in 2D
        spatial_dists = torch.cdist(positions_2d, positions_2d, p=2)
        
        # Convert to similarities (higher = more similar)
        spatial_sims = torch.exp(-spatial_dists / self.temperature)
        
        # Focus on preserving top-k neighborhoods
        semantic_topk = torch.topk(semantic_sims, k + 1, dim=1).indices[:, 1:]  # Exclude self
        
        loss = 0.0
        for i in range(batch_size):
            # For each point, ensure its semantic neighbors are also spatial neighbors
            semantic_neighbors = semantic_topk[i]
            
            # Get spatial similarities to these neighbors
            spatial_to_neighbors = spatial_sims[i, semantic_neighbors]
            
            # We want high spatial similarity to semantic neighbors
            loss += -torch.log(spatial_to_neighbors + 1e-8).mean()
        
        return loss / batch_size


class LocalStructureEncoder(nn.Module):
    """Generate visual patterns that encode local semantic structure"""
    
    def __init__(self, pattern_size: int = 32, semantic_dim: int = 512):
        super().__init__()
        self.pattern_size = pattern_size
        
        # Generate base pattern from 2D position
        self.position_to_base = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, pattern_size * pattern_size // 4)
        )
        
        # Add semantic details to the pattern
        self.semantic_to_details = nn.Sequential(
            nn.Linear(semantic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, pattern_size * pattern_size // 4)
        )
        
        # Combine position and semantic information
        self.combiner = nn.Sequential(
            nn.Linear(pattern_size * pattern_size // 2, 256),
            nn.ReLU(),
            nn.Linear(256, pattern_size * pattern_size * 3),
            nn.Sigmoid()
        )
        
    def forward(self, positions_2d: torch.Tensor, 
                semantic_embeddings: torch.Tensor) -> torch.Tensor:
        """Generate visual patterns from 2D position + semantic features"""
        
        # Base pattern from topological position
        base_features = self.position_to_base(positions_2d)
        
        # Detailed features from semantic information
        detail_features = self.semantic_to_details(semantic_embeddings)
        
        # Combine both sources of information
        combined = torch.cat([base_features, detail_features], dim=1)
        
        # Generate final visual pattern
        patterns = self.combiner(combined)
        
        # Reshape to image format
        batch_size = patterns.shape[0]
        patterns = patterns.view(batch_size, 3, self.pattern_size, self.pattern_size)
        
        return patterns


class VisualPatternDecoder(nn.Module):
    """Decode visual patterns back to semantic concepts"""
    
    def __init__(self, vocab_size: int, pattern_size: int = 32):
        super().__init__()
        self.pattern_size = pattern_size
        
        # CNN to process visual patterns
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(32 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, vocab_size)
        )
        
    def forward(self, visual_patterns: torch.Tensor) -> torch.Tensor:
        """Decode visual patterns to concept logits"""
        logits = self.cnn(visual_patterns)
        return F.log_softmax(logits, dim=1)


class TopologicalVisualEmbedding(nn.Module):
    """Complete model: Semantic embeddings → Topological visual patterns"""
    
    def __init__(self, vocab_size: int, semantic_dim: int = 512, pattern_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.semantic_dim = semantic_dim
        self.pattern_size = pattern_size
        
        # Components
        self.topological_projector = LearnedTopologicalProjection(semantic_dim)
        self.pattern_encoder = LocalStructureEncoder(pattern_size, semantic_dim)
        self.pattern_decoder = VisualPatternDecoder(vocab_size, pattern_size)
        
        print(f"Initialized topological visual embedding model:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Semantic dim: {semantic_dim}")
        print(f"  Pattern size: {pattern_size}×{pattern_size}")
        
    def forward(self, semantic_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: semantics → 2D → visual patterns → decoded concepts"""
        
        # Project to topology-preserving 2D space
        positions_2d = self.topological_projector(semantic_embeddings)
        
        # Generate visual patterns from position + semantics
        visual_patterns = self.pattern_encoder(positions_2d, semantic_embeddings)
        
        # Decode patterns back to concepts
        decoded_logits = self.pattern_decoder(visual_patterns)
        
        return visual_patterns, positions_2d, decoded_logits
    
    def encode_to_visual(self, semantic_embeddings: torch.Tensor) -> torch.Tensor:
        """Just the encoding: semantics → visual patterns"""
        positions_2d = self.topological_projector(semantic_embeddings)
        visual_patterns = self.pattern_encoder(positions_2d, semantic_embeddings)
        return visual_patterns
    
    def compose_visually(self, pattern1: torch.Tensor, pattern2: torch.Tensor, 
                        method: str = 'blend') -> torch.Tensor:
        """Compose two visual patterns"""
        if method == 'blend':
            return (pattern1 + pattern2) / 2
        elif method == 'overlay':
            return torch.max(pattern1, pattern2)
        elif method == 'multiply':
            return pattern1 * pattern2
        else:
            return (pattern1 + pattern2) / 2


def create_mock_semantic_embeddings(vocab: List[str]) -> torch.Tensor:
    """Create mock semantic embeddings with clear structure"""
    
    # Define semantic categories
    categories = {
        'animals': ['cat', 'dog', 'horse', 'bird', 'fish', 'lion'],
        'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white'],
        'sizes': ['big', 'small', 'large', 'tiny', 'huge', 'little'],
        'emotions': ['happy', 'sad', 'angry', 'excited', 'calm', 'worried'],
        'actions': ['run', 'walk', 'jump', 'fly', 'swim', 'dance'],
        'objects': ['car', 'house', 'tree', 'book', 'phone', 'chair']
    }
    
    # Category centers in semantic space
    torch.manual_seed(42)
    category_centers = {
        cat: torch.randn(512) for cat in categories.keys()
    }
    
    embeddings = []
    
    for word in vocab:
        # Find category
        category = None
        for cat, words in categories.items():
            if word in words:
                category = cat
                break
        
        if category:
            # Word embedding = category center + small variation
            center = category_centers[category]
            variation = torch.randn(512) * 0.15
            embedding = F.normalize(center + variation, dim=0)
        else:
            # Random embedding for unknown words
            embedding = F.normalize(torch.randn(512), dim=0)
        
        embeddings.append(embedding)
    
    return torch.stack(embeddings)


def train_topological_visual_embeddings(vocab: List[str], 
                                       semantic_embeddings: torch.Tensor,
                                       epochs: int = 300,
                                       lr: float = 0.001) -> Dict:
    """Train the topological visual embedding model"""
    
    vocab_size = len(vocab)
    model = TopologicalVisualEmbedding(vocab_size, semantic_embeddings.shape[1])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.8)
    
    # Training targets
    token_ids = torch.arange(vocab_size, dtype=torch.long)
    
    losses = {'total': [], 'reconstruction': [], 'topology': [], 'pattern_quality': []}
    accuracies = []
    
    print(f"\nTraining topological visual embeddings for {epochs} epochs...")
    print("Objectives:")
    print("  1. Perfect reconstruction (concept → visual → concept)")
    print("  2. Topology preservation (semantic neighbors = spatial neighbors)")
    print("  3. Visual pattern quality (meaningful, not noise)")
    print()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        visual_patterns, positions_2d, decoded_logits = model(semantic_embeddings)
        
        # Loss 1: Reconstruction - can we recover the original concepts?
        reconstruction_loss = F.nll_loss(decoded_logits, token_ids)
        
        # Loss 2: Topology preservation - are semantic neighbors spatial neighbors?
        topology_loss = model.topological_projector.neighborhood_preservation_loss(
            semantic_embeddings, positions_2d, k=5
        )
        
        # Loss 3: Pattern quality - patterns should be structured, not random
        pattern_quality_loss = compute_pattern_quality_loss(visual_patterns)
        
        # Combined loss
        total_loss = reconstruction_loss + 0.5 * topology_loss + 0.1 * pattern_quality_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.argmax(decoded_logits, dim=1)
            accuracy = (predictions == token_ids).float().mean().item()
        
        # Track metrics
        losses['total'].append(total_loss.item())
        losses['reconstruction'].append(reconstruction_loss.item())
        losses['topology'].append(topology_loss.item())
        losses['pattern_quality'].append(pattern_quality_loss.item())
        accuracies.append(accuracy)
        
        scheduler.step(total_loss)
        
        # Log progress
        if epoch % 25 == 0 or accuracy >= 0.95:
            print(f"Epoch {epoch:3d} | Loss: {total_loss:.4f} | "
                  f"Recon: {reconstruction_loss:.4f} | Topo: {topology_loss:.4f} | "
                  f"Quality: {pattern_quality_loss:.4f} | Acc: {accuracy:.1%}")
        
        # Early stopping
        if accuracy >= 0.95:
            print(f"\n✅ Excellent reconstruction achieved at epoch {epoch}!")
            break
        
        if epoch > 50 and accuracy < 0.3:
            print(f"\n⚠️ Poor convergence - may need architecture adjustments")
    
    return {
        'model': model,
        'losses': losses,
        'accuracies': accuracies,
        'final_accuracy': accuracy
    }


def compute_pattern_quality_loss(visual_patterns: torch.Tensor) -> torch.Tensor:
    """Encourage patterns to be structured rather than random noise"""
    
    # Patterns should have some structure (not uniform noise)
    # Measure local correlations - structured patterns have higher correlation
    
    batch_size, channels, height, width = visual_patterns.shape
    
    # Compute local spatial correlations
    spatial_loss = 0.0
    
    for c in range(channels):
        channel = visual_patterns[:, c, :, :]
        
        # Horizontal correlations
        h_diff = channel[:, :, 1:] - channel[:, :, :-1]
        h_smoothness = (h_diff ** 2).mean()
        
        # Vertical correlations  
        v_diff = channel[:, 1:, :] - channel[:, :-1, :]
        v_smoothness = (v_diff ** 2).mean()
        
        spatial_loss += h_smoothness + v_smoothness
    
    # Encourage some structure (penalize too much smoothness OR too much noise)
    target_smoothness = 0.1  # Sweet spot
    quality_loss = (spatial_loss - target_smoothness) ** 2
    
    return quality_loss


def test_topological_visual_embeddings(model: TopologicalVisualEmbedding,
                                     vocab: List[str],
                                     semantic_embeddings: torch.Tensor) -> Dict:
    """Test the trained model's capabilities"""
    
    model.eval()
    
    with torch.no_grad():
        visual_patterns, positions_2d, decoded_logits = model(semantic_embeddings)
        predictions = torch.argmax(decoded_logits, dim=1)
        
        # Test 1: Reconstruction accuracy
        accuracy = (predictions == torch.arange(len(vocab))).float().mean().item()
        
        # Test 2: Neighborhood preservation
        neighborhood_quality = test_neighborhood_preservation(
            vocab, semantic_embeddings, positions_2d
        )
        
        # Test 3: Visual composition
        composition_results = test_visual_composition(
            model, vocab, semantic_embeddings
        )
        
        # Test 4: Pattern interpretability
        interpretability_score = analyze_pattern_interpretability(
            vocab, visual_patterns
        )
        
        print("\n" + "="*60)
        print("TOPOLOGICAL VISUAL EMBEDDING RESULTS")
        print("="*60)
        
        print(f"Reconstruction Accuracy: {accuracy:.1%}")
        print(f"Neighborhood Preservation: {neighborhood_quality:.1%}")
        print(f"Composition Success Rate: {composition_results['success_rate']:.1%}")
        print(f"Pattern Interpretability: {interpretability_score:.1%}")
        
        overall_score = (accuracy + neighborhood_quality + 
                        composition_results['success_rate'] + interpretability_score) / 4
        
        print(f"Overall Score: {overall_score:.1%}")
        
        if overall_score >= 0.8:
            print("\n✅ EXCELLENT: Topological visual embeddings work very well!")
        elif overall_score >= 0.6:
            print("\n✅ SUCCESS: Promising results with room for improvement")
        elif overall_score >= 0.4:
            print("\n⚠️  PARTIAL: Some capabilities but needs work")
        else:
            print("\n❌ POOR: Fundamental issues remain")
        
        return {
            'reconstruction_accuracy': accuracy,
            'neighborhood_quality': neighborhood_quality,
            'composition_results': composition_results,
            'interpretability_score': interpretability_score,
            'overall_score': overall_score,
            'visual_patterns': visual_patterns,
            'positions_2d': positions_2d
        }


def test_neighborhood_preservation(vocab: List[str], 
                                 semantic_embeddings: torch.Tensor,
                                 positions_2d: torch.Tensor,
                                 k: int = 5) -> float:
    """Test if semantic neighbors are spatial neighbors"""
    
    semantic_sims = F.cosine_similarity(
        semantic_embeddings.unsqueeze(1), 
        semantic_embeddings.unsqueeze(0), 
        dim=2
    )
    
    spatial_dists = torch.cdist(positions_2d, positions_2d, p=2)
    
    overlaps = []
    
    for i, word in enumerate(vocab):
        # Semantic neighbors
        sem_neighbors = torch.topk(semantic_sims[i], k + 1).indices[1:]  # Exclude self
        
        # Spatial neighbors
        spatial_neighbors = torch.topk(-spatial_dists[i], k + 1).indices[1:]  # Exclude self
        
        # Calculate overlap
        overlap = len(set(sem_neighbors.tolist()) & set(spatial_neighbors.tolist()))
        overlaps.append(overlap / k)
    
    return np.mean(overlaps)


def test_visual_composition(model: TopologicalVisualEmbedding,
                          vocab: List[str],
                          semantic_embeddings: torch.Tensor) -> Dict:
    """Test if visual composition preserves semantic meaning"""
    
    # Test cases
    test_pairs = [
        ('big', 'cat'), ('red', 'car'), ('happy', 'dog'),
        ('small', 'bird'), ('blue', 'house'), ('fast', 'car')
    ]
    
    results = []
    
    word_to_id = {word: i for i, word in enumerate(vocab)}
    
    for word1, word2 in test_pairs:
        if word1 not in word_to_id or word2 not in word_to_id:
            continue
            
        id1, id2 = word_to_id[word1], word_to_id[word2]
        
        # Get individual visual patterns
        pattern1 = model.encode_to_visual(semantic_embeddings[id1:id1+1])
        pattern2 = model.encode_to_visual(semantic_embeddings[id2:id2+1])
        
        # Compose visually
        composed_pattern = model.compose_visually(pattern1, pattern2)
        
        # Decode composed pattern
        decoded_logits = model.pattern_decoder(composed_pattern)
        top_predictions = torch.topk(torch.exp(decoded_logits), 5).indices.squeeze()
        
        # Check if components appear in top predictions
        component_found = 0
        if id1 in top_predictions:
            component_found += 1
        if id2 in top_predictions:
            component_found += 1
            
        results.append({
            'pair': (word1, word2),
            'components_found': component_found,
            'top_predictions': [vocab[i] for i in top_predictions[:3]]
        })
    
    success_rate = np.mean([r['components_found'] / 2 for r in results])
    
    return {
        'results': results,
        'success_rate': success_rate
    }


def analyze_pattern_interpretability(vocab: List[str], 
                                   visual_patterns: torch.Tensor) -> float:
    """Analyze if visual patterns are interpretable (not random noise)"""
    
    # Simple interpretability metrics
    scores = []
    
    for i, pattern in enumerate(visual_patterns):
        # Metric 1: Spatial structure (not uniform)
        spatial_var = pattern.var(dim=(1, 2)).mean().item()
        
        # Metric 2: Not too noisy (some smoothness)
        h_grad = (pattern[:, :, 1:] - pattern[:, :, :-1]).abs().mean().item()
        v_grad = (pattern[:, 1:, :] - pattern[:, :-1, :]).abs().mean().item()
        smoothness = 1.0 / (1.0 + h_grad + v_grad)
        
        # Metric 3: Color coherence (channels relate to each other)
        channel_corr = 0
        for c1 in range(3):
            for c2 in range(c1 + 1, 3):
                corr = torch.corrcoef(torch.stack([
                    pattern[c1].flatten(),
                    pattern[c2].flatten()
                ]))[0, 1].abs().item()
                channel_corr += corr
        channel_corr /= 3  # Average correlation
        
        # Combined interpretability score
        score = (spatial_var + smoothness + channel_corr) / 3
        scores.append(score)
    
    return np.mean(scores)


def visualize_topological_results(vocab: List[str],
                                positions_2d: torch.Tensor,
                                visual_patterns: torch.Tensor):
    """Visualize the learned topological embeddings"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: 2D semantic landscape
    ax1.scatter(positions_2d[:, 0], positions_2d[:, 1], s=100, alpha=0.7)
    
    for i, word in enumerate(vocab):
        ax1.annotate(word, (positions_2d[i, 0], positions_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, alpha=0.8)
    
    ax1.set_title('Learned 2D Semantic Topology')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Visual patterns sample
    n_show = min(12, len(vocab))
    for i in range(n_show):
        ax = plt.subplot(3, 4, i + 1)
        
        # Show visual pattern
        pattern = visual_patterns[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(pattern, interpolation='nearest')
        ax.set_title(f"'{vocab[i]}'", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/greg/Documents/dev/visual_token_diffuser/topological_visual_embeddings.png', dpi=300)
    plt.show()


def run_topological_visual_experiment():
    """Run the complete topological visual embedding experiment"""
    
    print("="*60)
    print("TOPOLOGICAL VISUAL EMBEDDINGS EXPERIMENT")
    print("="*60)
    print("Training embeddings to create topologically meaningful visual patterns")
    print("Key hypothesis: Visual similarity = Semantic similarity (locally)")
    print()
    
    # Create vocabulary with semantic structure
    vocab = [
        # Animals
        'cat', 'dog', 'horse', 'bird', 'fish', 'lion',
        # Colors
        'red', 'blue', 'green', 'yellow', 'black', 'white',
        # Sizes
        'big', 'small', 'large', 'tiny', 'huge', 'little',
        # Emotions
        'happy', 'sad', 'angry', 'excited', 'calm', 'worried',
        # Actions
        'run', 'walk', 'jump', 'fly', 'swim', 'dance',
        # Objects
        'car', 'house', 'tree', 'book', 'phone', 'chair'
    ]
    
    print(f"Vocabulary: {len(vocab)} words")
    
    # Get semantic embeddings
    if CLIP_AVAILABLE:
        print("Using CLIP embeddings")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        with torch.no_grad():
            inputs = processor(text=vocab, return_tensors="pt", padding=True)
            semantic_embeddings = F.normalize(
                model.get_text_features(**inputs), dim=1
            )
    else:
        print("Using mock semantic embeddings")
        semantic_embeddings = create_mock_semantic_embeddings(vocab)
    
    print(f"Semantic embedding dimension: {semantic_embeddings.shape[1]}")
    print()
    
    # Train the model
    training_results = train_topological_visual_embeddings(
        vocab, semantic_embeddings, epochs=200, lr=0.001
    )
    
    # Test the trained model
    test_results = test_topological_visual_embeddings(
        training_results['model'], vocab, semantic_embeddings
    )
    
    # Visualize results
    visualize_topological_results(
        vocab, 
        test_results['positions_2d'],
        test_results['visual_patterns']
    )
    
    return {
        'training_results': training_results,
        'test_results': test_results,
        'vocab': vocab
    }


if __name__ == "__main__":
    results = run_topological_visual_experiment()