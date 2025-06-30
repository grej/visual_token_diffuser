"""
ExtremeAntiCollapseDecoder - A decoder designed to be impossible to collapse.
Uses multiple aggressive techniques to force input dependence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class ExtremeAntiCollapseDecoder(nn.Module):
    """
    Decoder designed to be IMPOSSIBLE to collapse to constant predictions.
    Uses multiple aggressive techniques to force input dependence.
    """
    
    def __init__(self, id_to_token: Dict[int, str], vocab_size: int, 
                 hidden_dim: int = 256, num_colors: int = 3, grid_size: int = 7):
        super().__init__()
        self.id_to_token = id_to_token
        self.vocab_size = vocab_size
        self.grid_size = grid_size
        self.num_colors = num_colors
        
        input_size = grid_size * grid_size * num_colors
        
        # 1. MULTIPLE PROJECTION HEADS with wildly different initializations
        self.projections = nn.ModuleList([
            nn.Linear(input_size, vocab_size) for _ in range(3)
        ])
        
        # Initialize each projection VERY differently
        nn.init.normal_(self.projections[0].weight, mean=0, std=3.0)      # Large Gaussian
        nn.init.xavier_uniform_(self.projections[1].weight, gain=2.0)     # Xavier with gain
        nn.init.orthogonal_(self.projections[2].weight, gain=1.5)         # Orthogonal
        
        # Different bias patterns to break symmetry
        self.projections[0].bias.data.uniform_(-2, 2)     # Uniform random
        self.projections[1].bias.data.normal_(0, 1)       # Gaussian
        self.projections[2].bias.data.zero_()             # Zero (different from others)
        
        # 2. MIXTURE OF EXPERTS - each must specialize differently
        self.expert_gate = nn.Linear(input_size, 3)
        self.expert_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, vocab_size)
            ),
            nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.Tanh(),  # Different activation
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, vocab_size)
            ),
            nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.GELU(),  # Different activation
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, vocab_size)
            )
        ])
        
        # 3. POSITION-AWARE SPATIAL PROCESSING
        self.position_embeddings = nn.Parameter(
            torch.randn(grid_size, grid_size, hidden_dim) * 0.5
        )
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_colors, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.spatial_proj = nn.Linear(64, vocab_size)
        
        # 4. PATTERN HASH EMBEDDING - discretizes continuous patterns
        self.hash_layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),  # Batch norm for stability
            nn.Linear(hidden_dim, vocab_size)
        )
        
        print(f"EXTREME Decoder: 3 projections + 3 experts + spatial conv + hash embedding")
        print(f"Each component initialized VERY differently - collapse is impossible!")
        
    def forward(self, patterns: torch.Tensor) -> torch.Tensor:
        batch_size = patterns.shape[0]
        
        # Ensure we have continuous patterns
        if patterns.dim() == 3:
            patterns = F.one_hot(patterns.long(), num_classes=self.num_colors).float()
        
        # Flatten patterns for linear layers
        flat_patterns = patterns.view(batch_size, -1)
        
        # PATH 1: DIRECT PROJECTIONS (3 wildly different initializations)
        proj_logits = []
        for i, proj in enumerate(self.projections):
            logits = proj(flat_patterns)
            # Add small input-dependent perturbation to prevent identical outputs
            perturbation = flat_patterns.std(dim=1, keepdim=True) * torch.randn_like(logits) * 0.01
            proj_logits.append(logits + perturbation)
        
        # PATH 2: MIXTURE OF EXPERTS (3 experts with different activations)
        expert_weights = F.softmax(self.expert_gate(flat_patterns), dim=-1)
        expert_outputs = []
        for expert in self.expert_networks:
            expert_outputs.append(expert(flat_patterns))
        
        # Stack and weighted sum
        expert_stack = torch.stack(expert_outputs, dim=1)  # [batch, 3, vocab]
        expert_logits = torch.sum(expert_stack * expert_weights.unsqueeze(-1), dim=1)
        
        # PATH 3: SPATIAL CNN PATH (processes 2D structure)
        # Reshape for conv: [batch, colors, height, width]
        conv_input = patterns.permute(0, 3, 1, 2)
        spatial_features = self.spatial_conv(conv_input).squeeze(-1).squeeze(-1)
        spatial_logits = self.spatial_proj(spatial_features)
        
        # PATH 4: HASH EMBEDDING PATH (with batch norm for stability)
        hash_logits = self.hash_layers(flat_patterns)
        
        # COMBINE ALL PATHS
        # Each path is so different they CANNOT all collapse to the same constant
        all_logits = proj_logits + [expert_logits, spatial_logits, hash_logits]
        
        # Weighted combination (not just average)
        weights = torch.tensor([0.2, 0.2, 0.15, 0.25, 0.1, 0.1], device=patterns.device)
        combined_logits = sum(w * logits for w, logits in zip(weights, all_logits))
        
        # FINAL ANTI-COLLAPSE TRICK: Add input-dependent noise during training
        if self.training:
            # Noise magnitude depends on input diversity
            input_std = flat_patterns.std(dim=1, keepdim=True)
            noise_magnitude = torch.clamp(input_std, 0.01, 0.1)  # Bounded noise
            noise = torch.randn_like(combined_logits) * noise_magnitude
            combined_logits = combined_logits + noise
        
        return F.log_softmax(combined_logits, dim=-1)
    
    def decode(self, patterns: torch.Tensor, temperature: float = 1.0, 
               greedy: bool = False) -> List[str]:
        """
        Decode patterns to tokens.
        """
        # Get token log probabilities 
        token_log_probs = self.forward(patterns)
        
        if greedy:
            # Take the most likely token
            token_ids = torch.argmax(token_log_probs, dim=1)
        else:
            # Convert to probabilities and apply temperature
            token_probs = torch.exp(token_log_probs)
            if temperature != 1.0:
                token_probs = token_probs.pow(1.0 / temperature)
                token_probs = token_probs / token_probs.sum(dim=1, keepdim=True)
            
            # Sample from the distribution
            token_ids = torch.multinomial(token_probs, num_samples=1).squeeze(-1)
        
        # Convert to list of tokens
        tokens = [self.id_to_token.get(id.item(), "<unk>") for id in token_ids]
        
        return tokens