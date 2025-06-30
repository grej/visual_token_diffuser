#!/usr/bin/env python3
"""
Simple 10-Hour Experiment: Semantic Visual Token Mapping

This experiment tests the core hypothesis with minimal complexity:
- 5 word vocabulary
- CLIP features as semantic visual representations
- Simple encoder/decoder architecture
- Perfect reconstruction target

If this fails, the approach has fundamental issues.
If this succeeds, we can scale up with confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os

# Try to import transformers/CLIP, fall back to mock if unavailable
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
    print("CLIP available - using real semantic features")
except ImportError:
    print("CLIP not available - using mock semantic features")
    CLIP_AVAILABLE = False


class MockCLIPFeatures:
    """Mock CLIP features for when transformers is not available"""
    
    def __init__(self):
        # Create semantically meaningful mock features
        # These should be different but related for similar concepts
        torch.manual_seed(42)  # Reproducible mock features
        
        # Base semantic features (512-dim like CLIP)
        self.features = {
            "cat": torch.randn(512) * 0.1 + torch.tensor([1.0, 0.8, 0.2, -0.5] + [0.0] * 508),
            "dog": torch.randn(512) * 0.1 + torch.tensor([1.1, 0.9, 0.1, -0.4] + [0.0] * 508),  # Similar to cat
            "red": torch.randn(512) * 0.1 + torch.tensor([-0.5, -0.8, 1.2, 0.9] + [0.0] * 508),
            "blue": torch.randn(512) * 0.1 + torch.tensor([-0.6, -0.7, 1.0, 0.8] + [0.0] * 508),  # Similar to red
            "big": torch.randn(512) * 0.1 + torch.tensor([0.3, 1.5, -0.2, -0.1] + [0.0] * 508),
        }
        
        # Normalize features
        for word in self.features:
            self.features[word] = F.normalize(self.features[word], dim=0)
    
    def get_features(self, words: List[str]) -> torch.Tensor:
        """Get mock CLIP features for words"""
        return torch.stack([self.features[word] for word in words])


class RealCLIPFeatures:
    """Real CLIP features extractor"""
    
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
    
    def get_features(self, words: List[str]) -> torch.Tensor:
        """Get real CLIP text features for words"""
        with torch.no_grad():
            inputs = self.processor(text=words, return_tensors="pt", padding=True)
            text_features = self.model.get_text_features(**inputs)
            return F.normalize(text_features, dim=1)


class SimpleSemanticEncoder(nn.Module):
    """Maps from token IDs to semantic visual features"""
    
    def __init__(self, vocab_size: int, feature_dim: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        
        # Simple linear mapping from one-hot to features
        self.linear = nn.Linear(vocab_size, feature_dim)
        
        print(f"Initialized encoder: {vocab_size} tokens -> {feature_dim} features")
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Convert token IDs to one-hot
        one_hot = F.one_hot(token_ids, num_classes=self.vocab_size).float()
        
        # Map to feature space
        features = self.linear(one_hot)
        
        # Normalize (like CLIP features)
        return F.normalize(features, dim=-1)


class SimpleSemanticDecoder(nn.Module):
    """Maps from semantic visual features back to token IDs"""
    
    def __init__(self, vocab_size: int, feature_dim: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        
        # Simple linear mapping from features to logits
        self.linear = nn.Linear(feature_dim, vocab_size)
        
        print(f"Initialized decoder: {feature_dim} features -> {vocab_size} tokens")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Map features to token logits
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class SimpleSemanticModel(nn.Module):
    """Complete encoder-decoder model"""
    
    def __init__(self, vocab_size: int, feature_dim: int = 512):
        super().__init__()
        self.encoder = SimpleSemanticEncoder(vocab_size, feature_dim)
        self.decoder = SimpleSemanticDecoder(vocab_size, feature_dim)
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Encode tokens to features
        features = self.encoder(token_ids)
        
        # Decode features back to tokens
        log_probs = self.decoder(features)
        
        return log_probs
    
    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(token_ids)
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)


def create_experiment_data():
    """Create the 5-word vocabulary and target features"""
    
    # Our minimal vocabulary
    vocab = ["cat", "dog", "red", "blue", "big"]
    vocab_size = len(vocab)
    
    # Word to ID mapping
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for word, i in word_to_id.items()}
    
    # Get semantic features (target representations)
    if CLIP_AVAILABLE:
        clip_extractor = RealCLIPFeatures()
        target_features = clip_extractor.get_features(vocab)
        print("Using real CLIP features")
    else:
        mock_extractor = MockCLIPFeatures()
        target_features = mock_extractor.get_features(vocab)
        print("Using mock semantic features")
    
    # Convert to token IDs tensor
    token_ids = torch.arange(vocab_size, dtype=torch.long)
    
    print(f"Vocabulary: {vocab}")
    print(f"Target features shape: {target_features.shape}")
    print(f"Feature similarities:")
    
    # Show semantic relationships in target features
    similarities = torch.mm(target_features, target_features.t())
    for i, word1 in enumerate(vocab):
        for j, word2 in enumerate(vocab):
            if i < j:  # Only show upper triangle
                sim = similarities[i, j].item()
                print(f"  {word1} <-> {word2}: {sim:.3f}")
    
    return {
        'vocab': vocab,
        'vocab_size': vocab_size,
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'token_ids': token_ids,
        'target_features': target_features,
        'similarities': similarities
    }


def semantic_alignment_loss(predicted_features: torch.Tensor, 
                          target_features: torch.Tensor) -> torch.Tensor:
    """
    Loss that encourages predicted features to align with target semantic features
    Uses cosine similarity to match CLIP's normalized space
    """
    # Cosine similarity between predicted and target
    cosine_sim = F.cosine_similarity(predicted_features, target_features, dim=1)
    
    # We want similarity to be 1.0 (perfect alignment)
    alignment_loss = (1.0 - cosine_sim).mean()
    
    return alignment_loss


def semantic_structure_loss(predicted_features: torch.Tensor,
                          target_similarities: torch.Tensor) -> torch.Tensor:
    """
    Loss that preserves the semantic relationship structure
    Similar words should have similar feature representations
    """
    # Compute pairwise similarities in predicted space
    predicted_similarities = torch.mm(predicted_features, predicted_features.t())
    
    # MSE between similarity matrices
    structure_loss = F.mse_loss(predicted_similarities, target_similarities)
    
    return structure_loss


def train_simple_experiment(data: Dict, epochs: int = 1000, lr: float = 0.01):
    """Train the simple semantic mapping model"""
    
    vocab_size = data['vocab_size']
    target_features = data['target_features']
    target_similarities = data['similarities']
    token_ids = data['token_ids']
    vocab = data['vocab']
    
    # Create model
    model = SimpleSemanticModel(vocab_size, feature_dim=target_features.shape[1])
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    reconstruction_accuracies = []
    
    print(f"\nTraining for {epochs} epochs...")
    print("Target: Perfect reconstruction (100% accuracy)")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        predicted_log_probs = model(token_ids)
        predicted_features = model.encode(token_ids)
        
        # Reconstruction loss (can the model predict the original tokens?)
        reconstruction_loss = F.nll_loss(predicted_log_probs, token_ids)
        
        # Semantic alignment loss (do features match CLIP targets?)
        alignment_loss = semantic_alignment_loss(predicted_features, target_features)
        
        # Semantic structure loss (are relationships preserved?)
        structure_loss = semantic_structure_loss(predicted_features, target_similarities)
        
        # Combined loss
        total_loss = reconstruction_loss + alignment_loss + 0.1 * structure_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.argmax(predicted_log_probs, dim=1)
            accuracy = (predictions == token_ids).float().mean().item()
        
        train_losses.append(total_loss.item())
        reconstruction_accuracies.append(accuracy)
        
        # Log progress
        if epoch % 100 == 0 or accuracy >= 1.0:
            print(f"Epoch {epoch:4d} | Loss: {total_loss:.4f} | "
                  f"Recon: {reconstruction_loss:.4f} | Align: {alignment_loss:.4f} | "
                  f"Struct: {structure_loss:.4f} | Acc: {accuracy:.1%}")
            
            # Show predictions vs targets
            print("  Predictions:", [vocab[p] for p in predictions.tolist()])
            print("  Targets:    ", [vocab[t] for t in token_ids.tolist()])
        
        # Early stopping if perfect reconstruction achieved
        if accuracy >= 1.0:
            print(f"\n✅ Perfect reconstruction achieved at epoch {epoch}!")
            break
    
    return {
        'model': model,
        'train_losses': train_losses,
        'reconstruction_accuracies': reconstruction_accuracies,
        'final_accuracy': accuracy
    }


def analyze_results(data: Dict, results: Dict):
    """Analyze and visualize the experiment results"""
    
    model = results['model']
    vocab = data['vocab']
    token_ids = data['token_ids']
    target_features = data['target_features']
    
    model.eval()
    with torch.no_grad():
        # Get final predictions
        predicted_log_probs = model(token_ids)
        predicted_features = model.encode(token_ids)
        predictions = torch.argmax(predicted_log_probs, dim=1)
        
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS")
        print("="*60)
        
        print(f"Final Accuracy: {results['final_accuracy']:.1%}")
        print(f"Target: 100% (perfect reconstruction)")
        
        if results['final_accuracy'] >= 1.0:
            print("✅ SUCCESS: Perfect reconstruction achieved!")
            print("   The semantic visual token approach works with proper grounding.")
        elif results['final_accuracy'] >= 0.8:
            print("⚠️  PARTIAL SUCCESS: Good but not perfect reconstruction.")
            print("   May need architecture refinements.")
        else:
            print("❌ FAILURE: Poor reconstruction accuracy.")
            print("   Fundamental issues with the approach.")
        
        print("\nFinal Predictions vs Targets:")
        for i, (pred_id, target_id) in enumerate(zip(predictions, token_ids)):
            pred_word = vocab[pred_id]
            target_word = vocab[target_id]
            correct = "✅" if pred_id == target_id else "❌"
            print(f"  {target_word} -> {pred_word} {correct}")
        
        # Analyze feature space
        print(f"\nFeature Space Analysis:")
        target_similarities = torch.mm(target_features, target_features.t())
        predicted_similarities = torch.mm(predicted_features, predicted_features.t())
        
        print("Target Similarities (CLIP features):")
        for i, word1 in enumerate(vocab):
            for j, word2 in enumerate(vocab):
                if i < j:
                    sim = target_similarities[i, j].item()
                    print(f"  {word1} <-> {word2}: {sim:.3f}")
        
        print("Learned Similarities (Model features):")
        for i, word1 in enumerate(vocab):
            for j, word2 in enumerate(vocab):
                if i < j:
                    sim = predicted_similarities[i, j].item()
                    print(f"  {word1} <-> {word2}: {sim:.3f}")
        
        # Feature alignment
        alignments = F.cosine_similarity(predicted_features, target_features, dim=1)
        print("Feature Alignments (cosine similarity with CLIP):")
        for i, word in enumerate(vocab):
            alignment = alignments[i].item()
            print(f"  {word}: {alignment:.3f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['train_losses'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(results['reconstruction_accuracies'])
    plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Reconstruction')
    plt.title('Reconstruction Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('/Users/greg/Documents/dev/visual_token_diffuser/simple_experiment_results.png')
    plt.show()
    
    return results['final_accuracy'] >= 1.0


def main():
    """Run the simple experiment"""
    
    print("="*60)
    print("SIMPLE SEMANTIC VISUAL TOKEN EXPERIMENT")
    print("="*60)
    print("Testing core hypothesis with minimal complexity:")
    print("- 5 word vocabulary: cat, dog, red, blue, big")
    print("- CLIP features as semantic visual representations")
    print("- Simple encoder/decoder architecture")
    print("- Target: Perfect reconstruction (100% accuracy)")
    print()
    
    # Create experiment data
    data = create_experiment_data()
    
    # Train model
    results = train_simple_experiment(data, epochs=1000, lr=0.01)
    
    # Analyze results
    success = analyze_results(data, results)
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if success:
        print("✅ EXPERIMENT SUCCESSFUL!")
        print("The semantic visual token approach works when properly grounded.")
        print("Ready to proceed with full implementation:")
        print("  1. Factorized representations (texture/color/structure)")
        print("  2. Hierarchical decoding")
        print("  3. Semantic consistency losses")
        print("  4. Guided diffusion training")
        print("  5. Multi-scale patterns")
    else:
        print("❌ EXPERIMENT FAILED!")
        print("Even with perfect semantic grounding, reconstruction failed.")
        print("This suggests fundamental issues with the visual token approach.")
        print("Consider alternative architectures before proceeding.")
    
    return success


if __name__ == "__main__":
    success = main()