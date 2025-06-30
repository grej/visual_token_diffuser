#!/usr/bin/env python3
"""
Test script to verify gradient flow is working with a minimal setup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import LearnableEncoder
from decoder import LearnableDecoder
from utils import create_simple_tokenizer

def test_gradient_flow():
    """Test that gradients flow from decoder loss back to encoder."""
    
    # Create tiny vocabulary for testing
    tiny_vocab = ["hello", "world", "test"]
    token_to_id = create_simple_tokenizer(tiny_vocab)
    vocab_size = len(token_to_id)
    id_to_token = {v: k for k, v in token_to_id.items()}
    
    print(f"Testing with vocabulary size: {vocab_size}")
    print(f"Token mapping: {token_to_id}")
    
    # Create encoder and decoder
    encoder = LearnableEncoder(token_to_id, embedding_dim=64, hidden_dim=128)
    decoder = LearnableDecoder(id_to_token, vocab_size, hidden_dim=128)
    
    # Create test data
    test_tokens = [token_to_id["hello"], token_to_id["world"], token_to_id["test"]]
    token_ids = torch.tensor(test_tokens, dtype=torch.long)
    
    print(f"Test token IDs: {token_ids}")
    
    # Test forward pass with Gumbel-Softmax
    print("\n--- Testing Gumbel-Softmax Forward Pass ---")
    
    # Get initial encoder embedding weights
    initial_weights = encoder.token_embedding.weight.clone()
    
    # Forward pass
    pattern_gumbel = encoder.forward_gumbel(token_ids, temperature=1.0)
    print(f"Gumbel patterns shape: {pattern_gumbel.shape}")
    print(f"Gumbel patterns require grad: {pattern_gumbel.requires_grad}")
    
    # Decode
    predicted_probs = decoder(pattern_gumbel)
    print(f"Predicted probs shape: {predicted_probs.shape}")
    
    # Calculate loss
    loss = F.cross_entropy(predicted_probs, token_ids)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check if gradients are non-zero
    encoder_grad_norm = encoder.token_embedding.weight.grad.norm().item()
    decoder_grad_norm = decoder.fc1.weight.grad.norm().item()
    
    print(f"Encoder embedding gradient norm: {encoder_grad_norm:.6f}")
    print(f"Decoder fc1 gradient norm: {decoder_grad_norm:.6f}")
    
    if encoder_grad_norm > 1e-6:
        print("✅ SUCCESS: Gradients are flowing to encoder!")
    else:
        print("❌ FAILURE: No gradients flowing to encoder!")
    
    # Test training step
    print("\n--- Testing Training Step ---")
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
    
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward pass
        pattern_gumbel = encoder.forward_gumbel(token_ids, temperature=1.0)
        predicted_probs = decoder(pattern_gumbel)
        loss = F.cross_entropy(predicted_probs, token_ids)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predicted_ids = torch.argmax(predicted_probs, dim=-1)
        accuracy = (predicted_ids == token_ids).float().mean().item()
        
        print(f"Step {step+1}: Loss={loss.item():.4f}, Accuracy={accuracy:.4f}")
        
        if accuracy > 0.9:
            print("✅ SUCCESS: High accuracy achieved!")
            break
    
    return accuracy > 0.5

if __name__ == "__main__":
    success = test_gradient_flow()
    if success:
        print("\n🎉 Gradient flow test PASSED!")
    else:
        print("\n😞 Gradient flow test FAILED!")