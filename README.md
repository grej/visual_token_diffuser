# NOTE: THIS REPO IS UNDER ACTIVE DEVELOPMENT AND IS NOT YET READY FOR USE. IT DOES NOT YET WORK! YMMV :)

# A Picture Is Worth 1000 Words: Visual Token Diffusion Language Models

> *"The atoms of language, made visible through pixels, dancing to the tune of diffusion."*

## Introduction: Reimagining Language Representation

This repository introduces a novel approach to language modeling that bridges the gap between textual and visual domains. Rather than processing language through the conventional lens of token IDs and embeddings, we explore an alternative paradigm: representing language tokens as visual patterns in a compact pixel space, and leveraging diffusion models to generate new text.

**Core Idea**: Transform text tokens into visual patterns (5×5 pixel grids with 3 possible colors per pixel), apply diffusion models within this visual space, and decode the generated patterns back into text. This creates a fundamentally different architecture for language generation than traditional autoregressive models.

By Greg Jennings

## Why Visual Token Representations?

Language models typically transform words into numeric token IDs, which are then mapped to high-dimensional vector embeddings. The core innovation here is to replace this abstraction with a visual representation:

```
"hello" → [3, 4, 5, 1, 2, 0, 2, 1, ...]  (Traditional embeddings)

"hello" →  ⬜⬜🟦⬜⬜     (Visual token approach (notional only)
           ⬜🟦⬜🟦⬜
           🟦⬜⬜⬜🟦
           ⬜🟦⬜🟦⬜
           ⬜⬜🟦⬜⬜
```

### The Mathematics of Possibility

The combinatorial space of a 5×5 grid with 3 possible colors per pixel is **3^25 = 847,288,609,443** - offering a vast representation space that dwarfs typical vocabulary sizes (50K-100K tokens). This richness creates the potential for:

1. **Efficient Token Compression**: Complex semantic concepts could be encoded in single visual patterns
2. **Emergent Semantic Structure**: Similar meanings could naturally map to visually similar patterns
3. **Evolutionary Vocabulary**: As the model learns, it can colonize unused regions of the visual space for new concepts

## Theoretical Foundations

This approach draws inspiration from diverse fields:

### From Cognitive Science

The human brain excels at processing and remembering visual information. By leveraging a visual representation space, we may tap into the neural machinery that has evolved for spatial and visual processing, potentially offering advantages for certain types of linguistic structures.

### From Diffusion Models

Recent advances in diffusion models have shown remarkable results in generating high-quality images, audio, and even discrete data like text. By operating in a visual token space, we can apply these powerful generative techniques to language in a novel way.

### From Information Theory

The 5×5 grid with 3 colors provides an information-dense representation space. Each pattern can theoretically encode log₂(3^25) ≈ 39.6 bits of information, significantly more than typical word embeddings require for unique identification.

## Implementation

This repository contains a minimal, Karpathy-inspired implementation of Visual Token Diffusion Language Models. The code is designed to be clear, educational, and modular, rather than optimized for production performance.

### Components

1. **Encoder**: Maps text tokens to visual patterns (5×5 grids)
   - `DeterministicEncoder`: Simple mapping from tokens to fixed patterns
   - `LearnableEncoder`: Neural network that learns to map tokens to visual patterns

2. **Diffusion Model**: Operates in the visual token space
   - `SimpleDiffusionModel`: Basic discrete diffusion for visual patterns
   - `AdvancedDiffusionModel`: Transformer-based diffusion with attention

3. **Decoder**: Maps visual patterns back to text
   - `DeterministicDecoder`: Maps patterns to tokens based on similarity
   - `LearnableDecoder`: Neural network that learns to decode patterns

4. **Utilities**: Visualization and data processing tools

### Training Approach

The model can be trained in three progressive stages:

1. **Reconstruction**: Train the encoder-decoder to accurately reconstruct text through the visual space
2. **Generation**: Train the diffusion model to generate plausible visual patterns
3. **RL Fine-tuning**: Use reinforcement learning to ensure generated patterns remain decodable

## Potential Applications and Advantages

### 1. Novel Representation Learning

This approach offers a fundamentally different way to represent and generate language, potentially capturing relationships that vector-based methods might miss.

### 2. Efficiency in Token Usage

The visual pattern space enables representing complex linguistic structures in compact forms, potentially leading to more efficient utilization of context windows.

### 3. Non-Autoregressive Generation

Unlike traditional language models that generate text one token at a time, diffusion models can generate all tokens in parallel, offering potential speed advantages.

### 4. Evolutionary Token Learning

The vast combinatorial space enables a model that can dynamically assign new visual patterns to represent novel concepts or word combinations, creating an adaptable, evolving vocabulary.

### 5. Theoretical Insights

This approach may provide new perspectives on the nature of language representation and generation, bridging insights from vision and language research.

## Current Limitations and Future Directions

### Limitations

- The visual encoding introduces computational overhead compared to simple embedding lookups
- Ensuring that generated visual patterns decode to valid language presents unique challenges
- The approach may struggle with very long-range dependencies that autoregressive models handle well

### Future Directions

1. **Semantic Visual Encoding**: Develop encoding schemes where visual similarity corresponds to semantic similarity
2. **Hierarchical Representations**: Create multi-scale visual patterns that capture both word and phrase-level meanings
3. **Cross-modal Transfer**: Explore whether pre-training on actual images helps the model learn better visual token representations
4. **Hybrid Approaches**: Combine visual token representations with traditional embeddings to leverage the strengths of both

## Getting Started

### Installation

We use conda to manage dependencies. Clone the repository and create the conda environment:

```bash
# Clone the repository
git clone https://github.com/gregjennings/visual-token-diffusion-lm.git
cd visual-token-diffusion-lm

# Create and activate conda environment
conda env create -f environment.yml
conda activate visual-token-diffusion
```

If you prefer pip, you can use:

```bash
pip install -r requirements.txt
```

### Training a Simple Model

```bash
python train.py --data text_data.txt --vocab_size 500 --encoder_type deterministic --diffusion_type simple
```

### Training an Advanced Model

```bash
python train.py --data text_data.txt --vocab_size 1000 --encoder_type learnable --diffusion_type advanced --decoder_type learnable
```

### Generating Text

```bash
python generate.py --checkpoint checkpoints/model_epoch_10.pt --prompt "The quick brown" --max_length 20
```


## Conclusion: Why This Matters

This project represents a fundamental rethinking of how language can be represented and generated. By bringing together insights from computer vision, diffusion models, and language processing, it creates an architecture that challenges our conventional understanding of language modeling.

In the spirit of Richard Feynman's approach to physics, we've stripped language modeling down to a visual essence and rebuilt it with diffusion mathematics. The potential payoff is significant: models that can represent language more efficiently, learn organically expanding vocabularies, and generate text through a fundamentally different process than today's autoregressive giants.

Is this the future of language modeling? Perhaps not in its current form. But by exploring these alternative perspectives, we enhance our understanding of what's possible and potentially discover principles that advance the entire field.

## References and Acknowledgments

This work draws inspiration from:

- Andrej Karpathy's nanoGPT and educational approaches to deep learning
- Ho et al.'s seminal work on Denoising Diffusion Probabilistic Models
- Austin et al.'s research on discrete diffusion in state-spaces
- The growing body of research on multimodal representation learning

## Citation

If you find this work useful, please consider citing:

```
@misc{jennings2025picture,
  title={A Picture Is Worth 1000 Words: Visual Token Diffusion Language Models},
  author={Jennings, Greg},
  year={2025},
  howpublished={\url{https://github.com/grej/visual-token-diffuser}}
}
```

## License

This project is released under the MIT License.
