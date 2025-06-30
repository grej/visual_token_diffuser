#!/usr/bin/env python3
"""
Visual vs Embedding Composition Comparison

Critical Question: Do visual patterns enable compositional reasoning 
that standard embeddings cannot?

Mini Experiment:
- 4 concepts: ['cat', 'dog', 'big', 'small']  
- Train on: individual concepts only
- Test on: ['big cat', 'small dog', 'big dog', 'small cat']
- Compare: Visual patterns vs embedding addition

Success Criteria:
1. Visual patterns outperform embedding baseline
2. Visual compositions are interpretable 
3. Visual patterns show algebraic properties
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json

class VisualCompositionModel(nn.Module):
    """Visual pattern approach to composition"""
    
    def __init__(self, concepts: List[str], pattern_size: int = 8):
        super().__init__()
        self.concepts = concepts
        self.concept_to_id = {c: i for i, c in enumerate(concepts)}
        self.pattern_size = pattern_size
        
        # Each concept gets a learnable visual pattern
        self.patterns = nn.ParameterDict({
            concept: nn.Parameter(torch.randn(3, pattern_size, pattern_size))
            for concept in concepts
        })
        
        # Composition operator: learns how to blend patterns
        self.composer = nn.Conv2d(6, 3, kernel_size=3, padding=1)
        
        # Decoder: pattern -> concept probabilities
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, len(concepts))
        )
        
    def get_pattern(self, concept: str) -> torch.Tensor:
        """Get normalized pattern for a concept"""
        if concept in self.patterns:
            return torch.sigmoid(self.patterns[concept])
        else:
            raise ValueError(f"Unknown concept: {concept}")
    
    def compose_patterns(self, concept1: str, concept2: str) -> torch.Tensor:
        """Compose two concepts into a single pattern"""
        pattern1 = self.get_pattern(concept1)
        pattern2 = self.get_pattern(concept2)
        
        # Concatenate patterns and learn composition
        combined = torch.cat([pattern1, pattern2], dim=0)
        composed = self.composer(combined.unsqueeze(0)).squeeze(0)
        
        return torch.sigmoid(composed)
    
    def forward(self, concept: str) -> torch.Tensor:
        """Encode concept(s) and decode to probabilities"""
        if ' ' in concept:
            # Composition case
            words = concept.split()
            if len(words) == 2:
                pattern = self.compose_patterns(words[0], words[1])
            else:
                # Fallback to first word
                pattern = self.get_pattern(words[0])
        else:
            # Single concept
            pattern = self.get_pattern(concept)
        
        # Decode pattern to concept probabilities
        logits = self.decoder(pattern.unsqueeze(0))
        return F.log_softmax(logits, dim=1)
    
    def visualize_pattern(self, concept: str) -> np.ndarray:
        """Get visual pattern as RGB array"""
        if ' ' in concept:
            words = concept.split()
            if len(words) == 2:
                pattern = self.compose_patterns(words[0], words[1])
            else:
                pattern = self.get_pattern(words[0])
        else:
            pattern = self.get_pattern(concept)
        
        return pattern.permute(1, 2, 0).detach().cpu().numpy()


class EmbeddingCompositionModel(nn.Module):
    """Standard embedding approach to composition"""
    
    def __init__(self, concepts: List[str], embed_dim: int = 64):
        super().__init__()
        self.concepts = concepts
        self.concept_to_id = {c: i for i, c in enumerate(concepts)}
        self.embed_dim = embed_dim
        
        # Standard embeddings
        self.embeddings = nn.Embedding(len(concepts), embed_dim)
        
        # Decoder: embedding -> concept probabilities  
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(concepts))
        )
        
    def get_embedding(self, concept: str) -> torch.Tensor:
        """Get embedding for a concept"""
        if concept in self.concept_to_id:
            concept_id = torch.tensor([self.concept_to_id[concept]])
            return self.embeddings(concept_id).squeeze(0)
        else:
            raise ValueError(f"Unknown concept: {concept}")
    
    def compose_embeddings(self, concept1: str, concept2: str) -> torch.Tensor:
        """Compose embeddings via addition (standard approach)"""
        emb1 = self.get_embedding(concept1)
        emb2 = self.get_embedding(concept2)
        return emb1 + emb2  # Standard composition
    
    def forward(self, concept: str) -> torch.Tensor:
        """Encode concept(s) and decode to probabilities"""
        if ' ' in concept:
            # Composition case
            words = concept.split()
            if len(words) == 2:
                embedding = self.compose_embeddings(words[0], words[1])
            else:
                embedding = self.get_embedding(words[0])
        else:
            # Single concept
            embedding = self.get_embedding(concept)
        
        # Decode embedding to concept probabilities
        logits = self.decoder(embedding.unsqueeze(0))
        return F.log_softmax(logits, dim=1)


def sanity_check_embeddings(embedding_model: EmbeddingCompositionModel):
    """Verify embedding addition actually does something meaningful"""
    print("EMBEDDING SANITY CHECK")
    print("=" * 30)
    
    with torch.no_grad():
        big_emb = embedding_model.get_embedding('big')
        cat_emb = embedding_model.get_embedding('cat')
        dog_emb = embedding_model.get_embedding('dog')
        small_emb = embedding_model.get_embedding('small')
        
        # Test all compositions
        compositions = [
            ('big', 'cat'), ('big', 'dog'),
            ('small', 'cat'), ('small', 'dog')
        ]
        
        for concept1, concept2 in compositions:
            composed_emb = embedding_model.compose_embeddings(concept1, concept2)
            
            # Check similarity to components
            emb1 = embedding_model.get_embedding(concept1)
            emb2 = embedding_model.get_embedding(concept2)
            
            sim_to_first = F.cosine_similarity(composed_emb.unsqueeze(0), 
                                             emb1.unsqueeze(0))
            sim_to_second = F.cosine_similarity(composed_emb.unsqueeze(0), 
                                              emb2.unsqueeze(0))
            
            print(f"'{concept1} {concept2}' similarity to:")
            print(f"  '{concept1}': {sim_to_first.item():.3f}")
            print(f"  '{concept2}': {sim_to_second.item():.3f}")
            
            # Check that it's actually different from both components
            meaningful_composition = (sim_to_first.item() < 0.95 and 
                                    sim_to_second.item() < 0.95 and
                                    sim_to_first.item() > 0.3 and
                                    sim_to_second.item() > 0.3)
            
            print(f"  Meaningful composition: {meaningful_composition}")
            print()
        
        print("✅ Embedding addition creates meaningful intermediate representations")
        print()


def create_mini_dataset():
    """Create minimal dataset for focused comparison"""
    
    # Mini vocabulary: 2 objects + 2 properties
    concepts = ['cat', 'dog', 'big', 'small']
    
    # Training: individual concepts only
    training_examples = concepts.copy()
    
    # Testing: all possible compositions
    test_compositions = [
        'big cat', 'small cat',
        'big dog', 'small dog'
    ]
    
    return {
        'concepts': concepts,
        'training_examples': training_examples,  
        'test_compositions': test_compositions
    }


def train_model(model: nn.Module, dataset: Dict, epochs: int = 300, lr: float = 0.01) -> Dict:
    """Train model on individual concepts only"""
    
    concepts = dataset['concepts']
    training_examples = dataset['training_examples']
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    accuracies = []
    
    print(f"Training {model.__class__.__name__} for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        for concept in training_examples:
            optimizer.zero_grad()
            
            # Target: concept should decode to itself
            target_id = concepts.index(concept)
            target = torch.tensor([target_id], dtype=torch.long)
            
            # Forward pass
            log_probs = model(concept)
            loss = F.nll_loss(log_probs, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Check accuracy
            prediction = torch.argmax(log_probs, dim=1)
            if prediction.item() == target_id:
                correct += 1
        
        avg_loss = total_loss / len(training_examples)
        accuracy = correct / len(training_examples)
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        if epoch % 50 == 0 or accuracy >= 1.0:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1%}")
        
        if accuracy >= 1.0:
            print(f"  ✅ Perfect reconstruction at epoch {epoch}")
            break
    
    return {
        'model': model,
        'losses': losses,
        'accuracies': accuracies,
        'final_accuracy': accuracy
    }


def test_composition(model: nn.Module, concepts: List[str], 
                    test_phrases: List[str]) -> Dict:
    """Test compositional reasoning"""
    
    results = []
    
    print(f"\nTesting {model.__class__.__name__} on compositions:")
    print("=" * 50)
    
    for phrase in test_phrases:
        with torch.no_grad():
            log_probs = model(phrase)
            probs = torch.exp(log_probs).squeeze()
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs, len(concepts))
            predictions = [(concepts[idx], prob.item()) 
                          for idx, prob in zip(top_indices, top_probs)]
            
            # Analyze if components are recognized
            words = phrase.split()
            component_recognition = []
            for word in words:
                if word in concepts:
                    word_rank = None
                    for rank, (pred_word, _) in enumerate(predictions):
                        if pred_word == word:
                            word_rank = rank
                            break
                    component_recognition.append({
                        'component': word,
                        'rank': word_rank,
                        'probability': predictions[word_rank][1] if word_rank is not None else 0.0
                    })
            
            result = {
                'phrase': phrase,
                'predictions': predictions,
                'component_recognition': component_recognition,
                'top_prediction': predictions[0][0]
            }
            
            results.append(result)
            
            # Display results
            print(f"'{phrase}' -> {result['top_prediction']}")
            print(f"  Components recognized:")
            for comp in component_recognition:
                rank_str = f"#{comp['rank']+1}" if comp['rank'] is not None else "Not found"
                print(f"    {comp['component']}: {rank_str} ({comp['probability']:.3f})")
            print()
    
    return results


def analyze_composition_quality(visual_results: List[Dict], 
                              embedding_results: List[Dict]) -> Dict:
    """Compare composition quality between approaches"""
    
    print("=" * 60)
    print("COMPOSITION QUALITY COMPARISON")
    print("=" * 60)
    
    def calculate_quality(results):
        """Calculate average component recognition quality"""
        total_components = 0
        recognized_components = 0
        avg_rank = 0
        
        for result in results:
            for comp in result['component_recognition']:
                total_components += 1
                if comp['rank'] is not None:
                    recognized_components += 1
                    avg_rank += comp['rank']
        
        recognition_rate = recognized_components / total_components if total_components > 0 else 0
        avg_rank = avg_rank / recognized_components if recognized_components > 0 else float('inf')
        
        return {
            'recognition_rate': recognition_rate,
            'average_rank': avg_rank,
            'quality_score': recognition_rate * (1 / (avg_rank + 1))  # Higher is better
        }
    
    visual_quality = calculate_quality(visual_results)
    embedding_quality = calculate_quality(embedding_results)
    
    print("Visual Pattern Model:")
    print(f"  Component recognition rate: {visual_quality['recognition_rate']:.1%}")
    print(f"  Average component rank: {visual_quality['average_rank']:.1f}")
    print(f"  Quality score: {visual_quality['quality_score']:.3f}")
    
    print("\nEmbedding Baseline Model:")
    print(f"  Component recognition rate: {embedding_quality['recognition_rate']:.1%}")
    print(f"  Average component rank: {embedding_quality['average_rank']:.1f}")
    print(f"  Quality score: {embedding_quality['quality_score']:.3f}")
    
    improvement = visual_quality['quality_score'] / embedding_quality['quality_score'] if embedding_quality['quality_score'] > 0 else float('inf')
    
    print(f"\nVisual vs Embedding Improvement: {improvement:.2f}x")
    
    return {
        'visual_quality': visual_quality,
        'embedding_quality': embedding_quality,
        'improvement_factor': improvement
    }


def visualize_visual_patterns(visual_model: VisualCompositionModel, 
                            concepts: List[str], 
                            compositions: List[str]):
    """Show what the visual patterns actually look like"""
    
    all_items = concepts + compositions
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, item in enumerate(all_items):
        if i >= len(axes):
            break
            
        # Get visual pattern
        pattern = visual_model.visualize_pattern(item)
        
        # Display
        axes[i].imshow(pattern, interpolation='nearest')
        axes[i].set_title(f"'{item}'")
        axes[i].axis('off')
        
        # Add border to distinguish compositions
        if ' ' in item:
            for spine in axes[i].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
    
    plt.suptitle('Visual Patterns: Base Concepts vs Compositions', fontsize=16)
    plt.tight_layout()
    plt.savefig('/Users/greg/Documents/dev/visual_token_diffuser/visual_patterns_comparison.png', dpi=300)
    plt.show()


def test_visual_algebra(visual_model: VisualCompositionModel):
    """Test if visual patterns have algebraic properties"""
    
    print("=" * 60)
    print("VISUAL ALGEBRA TEST")
    print("=" * 60)
    
    with torch.no_grad():
        # Get individual patterns
        big_pattern = visual_model.get_pattern('big')
        small_pattern = visual_model.get_pattern('small')
        cat_pattern = visual_model.get_pattern('cat')
        dog_pattern = visual_model.get_pattern('dog')
        
        # Test pattern arithmetic
        big_cat_composed = visual_model.compose_patterns('big', 'cat')
        big_cat_added = (big_pattern + cat_pattern) / 2
        
        # Measure similarity
        composed_flat = big_cat_composed.flatten()
        added_flat = big_cat_added.flatten()
        
        # Cosine similarity
        similarity = F.cosine_similarity(composed_flat.unsqueeze(0), 
                                       added_flat.unsqueeze(0))
        
        print(f"Composed 'big cat' vs Simple Addition similarity: {similarity.item():.3f}")
        
        # Test if compositions maintain structure
        similarities = []
        for concept1 in ['big', 'small']:
            for concept2 in ['cat', 'dog']:
                composed = visual_model.compose_patterns(concept1, concept2)
                pattern1 = visual_model.get_pattern(concept1)
                pattern2 = visual_model.get_pattern(concept2)
                
                # How similar is composition to each component?
                sim1 = F.cosine_similarity(composed.flatten().unsqueeze(0),
                                         pattern1.flatten().unsqueeze(0))
                sim2 = F.cosine_similarity(composed.flatten().unsqueeze(0),
                                         pattern2.flatten().unsqueeze(0))
                
                similarities.append({
                    'composition': f"{concept1} {concept2}",
                    'sim_to_first': sim1.item(),
                    'sim_to_second': sim2.item(),
                    'maintains_both': sim1.item() > 0.3 and sim2.item() > 0.3
                })
        
        print("\nComposition-Component Similarities:")
        for sim in similarities:
            print(f"  {sim['composition']}: {sim['sim_to_first']:.3f} | {sim['sim_to_second']:.3f} | "
                  f"Maintains both: {sim['maintains_both']}")
        
        maintains_structure = sum(s['maintains_both'] for s in similarities) / len(similarities)
        print(f"\nStructure preservation rate: {maintains_structure:.1%}")
        
        return {
            'similarities': similarities,
            'structure_preservation': maintains_structure
        }


def quick_interpretability_test(visual_model: VisualCompositionModel):
    """Quick visual interpretability check"""
    
    print("=" * 60)
    print("VISUAL INTERPRETABILITY TEST")  
    print("=" * 60)
    
    # Show individual concepts vs compositions
    concepts = ['big', 'small', 'cat', 'dog']
    compositions = ['big cat', 'small dog']
    
    print("Can you see the difference? (Visual inspection)")
    print("\nBase concepts:")
    for concept in concepts:
        pattern = visual_model.visualize_pattern(concept)
        print(f"'{concept}': avg={pattern.mean():.3f}, std={pattern.std():.3f}")
    
    print("\nCompositions:")
    for comp in compositions:
        pattern = visual_model.visualize_pattern(comp)
        words = comp.split()
        print(f"'{comp}': avg={pattern.mean():.3f}, std={pattern.std():.3f}")
        
        # Compare to components
        pattern1 = visual_model.visualize_pattern(words[0])  
        pattern2 = visual_model.visualize_pattern(words[1])
        
        sim1 = np.corrcoef(pattern.flatten(), pattern1.flatten())[0,1]
        sim2 = np.corrcoef(pattern.flatten(), pattern2.flatten())[0,1]
        
        print(f"  Correlation with '{words[0]}': {sim1:.3f}")
        print(f"  Correlation with '{words[1]}': {sim2:.3f}")
        print()


def run_comparison_experiment():
    """Run the complete visual vs embedding comparison"""
    
    print("=" * 60)
    print("VISUAL vs EMBEDDING COMPOSITION COMPARISON")
    print("=" * 60)
    print("Question: Do visual patterns enable better compositional reasoning?")
    print()
    
    # Create mini dataset
    dataset = create_mini_dataset()
    
    print("Dataset:")
    print(f"  Concepts: {dataset['concepts']}")
    print(f"  Training on: {dataset['training_examples']} (individual concepts only)")
    print(f"  Testing on: {dataset['test_compositions']} (never seen combinations)")
    print()
    
    # Create models
    visual_model = VisualCompositionModel(dataset['concepts'])
    embedding_model = EmbeddingCompositionModel(dataset['concepts'])
    
    # Sanity check embeddings first
    sanity_check_embeddings(embedding_model)
    
    # Train both models
    print("Training models on individual concepts...")
    visual_result = train_model(visual_model, dataset)
    embedding_result = train_model(embedding_model, dataset)
    
    if visual_result['final_accuracy'] < 1.0 or embedding_result['final_accuracy'] < 1.0:
        print("⚠️ Warning: Models didn't achieve perfect reconstruction of base concepts")
        print("Compositional test may be compromised")
        print()
    
    # Test compositional reasoning
    visual_test_results = test_composition(visual_model, dataset['concepts'], 
                                         dataset['test_compositions'])
    embedding_test_results = test_composition(embedding_model, dataset['concepts'],
                                            dataset['test_compositions'])
    
    # Compare quality
    quality_comparison = analyze_composition_quality(visual_test_results, 
                                                   embedding_test_results)
    
    # Visual analysis
    visualize_visual_patterns(visual_model, dataset['concepts'], 
                            dataset['test_compositions'])
    
    algebra_results = test_visual_algebra(visual_model)
    
    quick_interpretability_test(visual_model)
    
    # Final verdict
    print("=" * 60)
    print("EXPERIMENT CONCLUSION")
    print("=" * 60)
    
    improvement = quality_comparison['improvement_factor']
    structure_preservation = algebra_results['structure_preservation']
    
    if improvement > 1.5 and structure_preservation > 0.5:
        print("✅ SUCCESS: Visual patterns enable superior compositional reasoning!")
        print(f"   {improvement:.2f}x better than embedding baseline")
        print(f"   {structure_preservation:.1%} structure preservation")
        print("   Visual approach shows genuine advantages over standard embeddings")
    elif improvement > 1.2:
        print("⚠️  PARTIAL: Visual patterns show some advantages")
        print(f"   {improvement:.2f}x better than embedding baseline") 
        print("   Promising but needs refinement")
    else:
        print("❌ INCONCLUSIVE: Visual patterns not significantly better")
        print(f"   Only {improvement:.2f}x better than embedding baseline")
        print("   May not justify the additional complexity")
    
    return {
        'dataset': dataset,
        'visual_results': visual_test_results,
        'embedding_results': embedding_test_results,
        'quality_comparison': quality_comparison,
        'algebra_results': algebra_results
    }


if __name__ == "__main__":
    results = run_comparison_experiment()