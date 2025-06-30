#!/usr/bin/env python3
"""
Semantic UMAP Topology Experiment

Key Hypothesis: UMAP can create 2D visual representations that preserve 
semantic topology, enabling meaningful composition in visual space.

Critical Questions:
1. Do semantic embeddings preserve topology when projected to 2D?
2. Can we predict semantic relationships from 2D spatial distances?
3. Does composition work better in topological 2D space?

If yes -> Visual representations with semantic structure work!
If no -> Visual approach is fundamentally flawed
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

# Try to import UMAP and CLIP
try:
    import umap
    UMAP_AVAILABLE = True
    print("UMAP available")
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available - using t-SNE as fallback")

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
    print("CLIP available")  
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available - using mock embeddings")


class MockSemanticEmbeddings:
    """High-quality mock embeddings with clear semantic structure"""
    
    def __init__(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Define semantic categories with clear relationships
        self.category_centers = {
            'animals': np.array([1.0, 0.8, 0.2, -0.5]),
            'colors': np.array([-0.5, -0.8, 1.2, 0.9]),
            'sizes': np.array([0.3, 1.5, -0.2, -0.1]),
            'emotions': np.array([0.8, -0.3, 0.6, 1.2]),
            'actions': np.array([-0.2, 0.9, -0.8, 0.4]),
            'food': np.array([0.6, 0.4, 0.1, 0.8]),
            'places': np.array([0.5, 0.1, -0.4, 0.6]),
            'objects': np.array([0.1, 0.1, 0.8, -0.6])
        }
        
        # Expand to full 384-dimensional space
        self.category_centers = {
            cat: np.concatenate([center, np.zeros(380)]) 
            for cat, center in self.category_centers.items()
        }
        
        # Define vocabulary with semantic relationships
        self.vocab_structure = {
            'animals': ['cat', 'dog', 'horse', 'bird', 'fish', 'lion', 'mouse', 'bear'],
            'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange'],
            'sizes': ['big', 'small', 'large', 'tiny', 'huge', 'little', 'massive', 'mini'],
            'emotions': ['happy', 'sad', 'angry', 'excited', 'calm', 'worried', 'joyful', 'nervous'],
            'actions': ['run', 'walk', 'jump', 'fly', 'swim', 'dance', 'sing', 'sleep'],
            'food': ['apple', 'bread', 'water', 'coffee', 'cake', 'pizza', 'rice', 'meat'],
            'places': ['home', 'school', 'park', 'city', 'beach', 'mountain', 'forest', 'office'],
            'objects': ['car', 'book', 'phone', 'chair', 'table', 'door', 'window', 'computer']
        }
        
        # Create embeddings
        self.embeddings = {}
        self.vocab = []
        
        for category, words in self.vocab_structure.items():
            center = self.category_centers[category]
            
            for word in words:
                # Add word-specific variation around category center
                variation = np.random.randn(384) * 0.1
                embedding = center + variation
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                
                self.embeddings[word] = embedding
                self.vocab.append(word)
        
        print(f"Created mock embeddings for {len(self.vocab)} words across {len(self.vocab_structure)} categories")
    
    def get_embeddings(self, words: List[str]) -> np.ndarray:
        """Get embeddings for list of words"""
        return np.array([self.embeddings[word] for word in words])


class RealCLIPEmbeddings:
    """Real CLIP embeddings for comparison"""
    
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
    
    def get_embeddings(self, words: List[str]) -> np.ndarray:
        """Get CLIP embeddings for words"""
        with torch.no_grad():
            inputs = self.processor(text=words, return_tensors="pt", padding=True)
            embeddings = self.model.get_text_features(**inputs)
            return F.normalize(embeddings, dim=1).cpu().numpy()


class SemanticTopologyTester:
    """Test if 2D projections preserve semantic relationships"""
    
    def __init__(self, vocab: List[str], embeddings: np.ndarray):
        self.vocab = vocab
        self.embeddings = embeddings
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        
        # Compute semantic similarity matrix
        self.semantic_similarities = cosine_similarity(embeddings)
        
    def project_to_2d(self, method='umap', **kwargs) -> np.ndarray:
        """Project embeddings to 2D using UMAP or t-SNE"""
        
        if method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1),
                metric='cosine',
                random_state=42
            )
            print("Using UMAP for 2D projection")
        else:
            reducer = TSNE(
                n_components=2,
                perplexity=kwargs.get('perplexity', 30),
                random_state=42,
                metric='cosine'
            )
            print("Using t-SNE for 2D projection")
        
        coords_2d = reducer.fit_transform(self.embeddings)
        
        # Normalize to [0, 1] range
        coords_2d = (coords_2d - coords_2d.min(axis=0)) / (coords_2d.max(axis=0) - coords_2d.min(axis=0))
        
        return coords_2d
    
    def test_topology_preservation(self, coords_2d: np.ndarray) -> Dict:
        """Test if 2D distances correlate with semantic similarities"""
        
        # Compute 2D distances
        spatial_distances = euclidean_distances(coords_2d)
        
        # Convert similarities to distances (1 - similarity)
        semantic_distances = 1 - self.semantic_similarities
        
        # Get upper triangular matrices (avoid diagonal and duplicates)
        n = len(self.vocab)
        upper_indices = np.triu_indices(n, k=1)
        
        semantic_dist_flat = semantic_distances[upper_indices]
        spatial_dist_flat = spatial_distances[upper_indices]
        
        # Correlations
        pearson_r, pearson_p = pearsonr(semantic_dist_flat, spatial_dist_flat)
        spearman_r, spearman_p = spearmanr(semantic_dist_flat, spatial_dist_flat)
        
        print(f"Topology Preservation Analysis:")
        print(f"  Pearson correlation: {pearson_r:.3f} (p={pearson_p:.4f})")
        print(f"  Spearman correlation: {spearman_r:.3f} (p={spearman_p:.4f})")
        
        # Strong correlation = topology preserved
        topology_preserved = abs(pearson_r) > 0.5 and pearson_p < 0.01
        
        if topology_preserved:
            print("  ✅ Topology well preserved - semantic relationships maintained in 2D")
        else:
            print("  ❌ Topology poorly preserved - semantic structure lost in projection")
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'topology_preserved': topology_preserved,
            'semantic_distances': semantic_distances,
            'spatial_distances': spatial_distances
        }
    
    def analyze_semantic_neighborhoods(self, coords_2d: np.ndarray, k=5) -> Dict:
        """Check if semantically similar words are spatial neighbors"""
        
        results = []
        
        for i, word in enumerate(self.vocab):
            # Get semantic neighbors (most similar words)
            semantic_sims = self.semantic_similarities[i]
            semantic_neighbors = np.argsort(semantic_sims)[-k-1:-1][::-1]  # Exclude self
            
            # Get spatial neighbors (closest in 2D)
            point = coords_2d[i]
            distances = np.sum((coords_2d - point)**2, axis=1)
            spatial_neighbors = np.argsort(distances)[1:k+1]  # Exclude self
            
            # Calculate overlap
            overlap = len(set(semantic_neighbors) & set(spatial_neighbors))
            overlap_ratio = overlap / k
            
            results.append({
                'word': word,
                'semantic_neighbors': [self.vocab[j] for j in semantic_neighbors],
                'spatial_neighbors': [self.vocab[j] for j in spatial_neighbors],
                'overlap': overlap,
                'overlap_ratio': overlap_ratio
            })
        
        avg_overlap = np.mean([r['overlap_ratio'] for r in results])
        
        print(f"\nSemantic Neighborhood Analysis (k={k}):")
        print(f"  Average overlap ratio: {avg_overlap:.2%}")
        
        # Show some examples
        print("  Examples:")
        for result in results[:5]:
            print(f"    {result['word']}:")
            print(f"      Semantic: {result['semantic_neighbors']}")
            print(f"      Spatial: {result['spatial_neighbors']} (overlap: {result['overlap_ratio']:.1%})")
        
        return {
            'results': results,
            'average_overlap': avg_overlap,
            'good_neighborhoods': avg_overlap > 0.4
        }
    
    def test_composition_in_2d(self, coords_2d: np.ndarray) -> Dict:
        """Test if composition works better in 2D manifold space"""
        
        # Test cases: property + object combinations
        test_cases = [
            ('big', 'cat'), ('small', 'dog'), ('red', 'car'),
            ('happy', 'cat'), ('fast', 'car'), ('green', 'tree')
        ]
        
        results = []
        
        for prop, obj in test_cases:
            if prop not in self.word_to_id or obj not in self.word_to_id:
                continue
                
            prop_id = self.word_to_id[prop]
            obj_id = self.word_to_id[obj]
            
            # 2D composition: simple average
            prop_2d = coords_2d[prop_id]
            obj_2d = coords_2d[obj_id]
            composed_2d = (prop_2d + obj_2d) / 2
            
            # Find nearest neighbors to composed point
            distances = np.sum((coords_2d - composed_2d)**2, axis=1)
            nearest_ids = np.argsort(distances)[:5]
            nearest_words = [self.vocab[i] for i in nearest_ids]
            
            # Check if components are among nearest neighbors
            component_ranks = []
            for component in [prop, obj]:
                try:
                    rank = nearest_words.index(component)
                    component_ranks.append(rank)
                except ValueError:
                    component_ranks.append(None)
            
            results.append({
                'composition': f"{prop} {obj}",
                'nearest_neighbors': nearest_words,
                'component_ranks': component_ranks,
                'components_in_top5': sum(1 for r in component_ranks if r is not None and r < 5)
            })
        
        avg_components_preserved = np.mean([r['components_in_top5'] for r in results])
        
        print(f"\n2D Composition Analysis:")
        print(f"  Average components preserved in top-5: {avg_components_preserved:.1f}/2")
        
        for result in results:
            print(f"  '{result['composition']}' -> {result['nearest_neighbors'][:3]}")
            print(f"    Component ranks: {result['component_ranks']}")
        
        return {
            'results': results,
            'avg_components_preserved': avg_components_preserved,
            'composition_works': avg_components_preserved >= 1.0
        }


def visualize_semantic_landscape(vocab: List[str], coords_2d: np.ndarray, 
                               categories: Dict[str, List[str]] = None):
    """Visualize the 2D semantic landscape"""
    
    plt.figure(figsize=(12, 10))
    
    if categories:
        # Color by category
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        for i, (category, words) in enumerate(categories.items()):
            category_indices = [j for j, word in enumerate(vocab) if word in words]
            if category_indices:
                category_coords = coords_2d[category_indices]
                plt.scatter(category_coords[:, 0], category_coords[:, 1], 
                          c=[colors[i]], label=category, s=50, alpha=0.7)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Single color
        plt.scatter(coords_2d[:, 0], coords_2d[:, 1], s=50, alpha=0.7)
    
    # Add word labels
    for i, word in enumerate(vocab):
        plt.annotate(word, (coords_2d[i, 0], coords_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    plt.title('Semantic Landscape in 2D Space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/greg/Documents/dev/visual_token_diffuser/semantic_landscape_2d.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_umap_topology_experiment():
    """Run the complete UMAP topology experiment"""
    
    print("=" * 60)
    print("SEMANTIC TOPOLOGY PRESERVATION EXPERIMENT")
    print("=" * 60)
    print("Testing: Can 2D projections preserve semantic relationships?")
    print()
    
    # Get semantic embeddings
    if CLIP_AVAILABLE:
        print("Using real CLIP embeddings")
        embedder = RealCLIPEmbeddings()
        # Use a curated vocabulary for cleaner results
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
        embeddings = embedder.get_embeddings(vocab)
        categories = {
            'animals': ['cat', 'dog', 'horse', 'bird', 'fish', 'lion'],
            'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white'],
            'sizes': ['big', 'small', 'large', 'tiny', 'huge', 'little'],
            'emotions': ['happy', 'sad', 'angry', 'excited', 'calm', 'worried'],
            'actions': ['run', 'walk', 'jump', 'fly', 'swim', 'dance'],
            'objects': ['car', 'house', 'tree', 'book', 'phone', 'chair']
        }
    else:
        print("Using high-quality mock embeddings")
        embedder = MockSemanticEmbeddings()
        vocab = embedder.vocab
        embeddings = embedder.get_embeddings(vocab)
        categories = embedder.vocab_structure
    
    print(f"Vocabulary: {len(vocab)} words")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print()
    
    # Create topology tester
    tester = SemanticTopologyTester(vocab, embeddings)
    
    # Project to 2D
    coords_2d = tester.project_to_2d(method='umap' if UMAP_AVAILABLE else 'tsne')
    
    # Test topology preservation
    topology_results = tester.test_topology_preservation(coords_2d)
    
    # Test semantic neighborhoods
    neighborhood_results = tester.analyze_semantic_neighborhoods(coords_2d)
    
    # Test 2D composition
    composition_results = tester.test_composition_in_2d(coords_2d)
    
    # Visualize results
    visualize_semantic_landscape(vocab, coords_2d, categories)
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    
    topology_score = abs(topology_results['pearson_r'])
    neighborhood_score = neighborhood_results['average_overlap']
    composition_score = composition_results['avg_components_preserved'] / 2.0
    
    overall_score = (topology_score + neighborhood_score + composition_score) / 3
    
    print(f"Topology Preservation: {topology_score:.2f} ({'✅' if topology_results['topology_preserved'] else '❌'})")
    print(f"Semantic Neighborhoods: {neighborhood_score:.2f} ({'✅' if neighborhood_results['good_neighborhoods'] else '❌'})")
    print(f"2D Composition: {composition_score:.2f} ({'✅' if composition_results['composition_works'] else '❌'})")
    print(f"Overall Score: {overall_score:.2f}")
    
    if overall_score >= 0.6:
        print("\n✅ SUCCESS: Semantic topology well preserved in 2D!")
        print("   UMAP/visual approach shows genuine promise")
        print("   Semantic relationships maintained in visual space")
        print("   Composition works naturally in 2D manifold")
    elif overall_score >= 0.4:
        print("\n⚠️  PARTIAL: Some topology preserved but imperfect")
        print("   Promising direction but needs refinement")
    else:
        print("\n❌ FAILURE: Semantic topology not preserved")
        print("   2D projection loses too much semantic structure")
        print("   Visual approach unlikely to work")
    
    return {
        'topology_results': topology_results,
        'neighborhood_results': neighborhood_results,
        'composition_results': composition_results,
        'overall_score': overall_score,
        'coords_2d': coords_2d,
        'vocab': vocab
    }


if __name__ == "__main__":
    # Install UMAP if needed
    if not UMAP_AVAILABLE:
        print("To get the best results, install UMAP: pip install umap-learn")
        print("Falling back to t-SNE for now...")
        print()
    
    results = run_umap_topology_experiment()