#!/usr/bin/env python3
"""
Token-efficient context creator for Visual Token Diffusion LM.
Creates a curated file package for sharing with other models without token bloat.
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Set

# Define file categories by importance
CRITICAL_FILES = [
    'model.py',      # Core orchestrator
    'encoder.py',    # Token-to-visual encoding
    'decoder.py',    # Visual-to-token decoding  
    'diffusion.py',  # Diffusion model
    'extreme_decoder.py',  # Anti-collapse decoder
]

IMPORTANT_FILES = [
    'train.py',      # Training pipeline
    'utils.py',      # Utilities and helpers
    'generate.py',   # Text generation
]

CONTEXT_FILES = [
    'README.md',           # Project overview and theory
    'findings.md',         # Current research findings
    'next_steps.md',       # Future directions
    'requirements.txt',    # Dependencies
    'environment-cpu.yml', # Conda environment
    'updates28apr.md',     # Recent updates
]

# Files to always exclude (too large/not helpful for context)
EXCLUDE_FILES = [
    '__pycache__',
    '.git',
    'checkpoints',
    '.png',
    '.jpg', 
    '.jpeg',
    '.pkl',
    '.pt',
    '.pth',
    'environment-gpu.yml',  # Keep only CPU version
    'environment.yml',      # Keep only CPU version
    '.pyc',
    'sample_data.txt',
    'sample_val_data.txt',
    'small_train.txt',
    'small_val.txt',
    'tiny_test.txt',
    'tiny_val.txt',
    'digest.txt',
]

def should_exclude_file(filepath: Path) -> bool:
    """Check if file should be excluded based on name or extension."""
    name = filepath.name.lower()
    
    # Check exact matches
    if name in EXCLUDE_FILES:
        return True
        
    # Check extensions
    for exclude in EXCLUDE_FILES:
        if exclude.startswith('.') and name.endswith(exclude):
            return True
            
    # Check if it's in excluded directories
    for part in filepath.parts:
        if part in EXCLUDE_FILES:
            return True
            
    return False

def get_file_size_mb(filepath: Path) -> float:
    """Get file size in MB."""
    try:
        return filepath.stat().st_size / (1024 * 1024)
    except:
        return 0

def create_context_package(
    root_dir: str,
    include_files: List[str] = None,
    exclude_files: List[str] = None,
    max_file_size_mb: float = 2.0,  # Increased for larger files
    output_file: str = "context_package.md"
) -> None:
    """
    Create a token-efficient context package.
    
    Args:
        root_dir: Project root directory
        include_files: Specific files to include (overrides defaults)
        exclude_files: Additional files to exclude
        max_file_size_mb: Maximum file size in MB to include
        output_file: Output file name
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Error: Directory {root_dir} does not exist")
        return
        
    # Determine which files to include
    if include_files:
        target_files = include_files
    else:
        target_files = CRITICAL_FILES + IMPORTANT_FILES + CONTEXT_FILES
        
    # Add any additional exclusions
    exclude_set = set(EXCLUDE_FILES)
    if exclude_files:
        exclude_set.update(exclude_files)
    
    # Find and validate files
    found_files = []
    missing_files = []
    skipped_files = []
    
    for filename in target_files:
        filepath = root_path / filename
        
        if not filepath.exists():
            missing_files.append(filename)
            continue
            
        if should_exclude_file(filepath):
            skipped_files.append(filename)
            continue
            
        file_size = get_file_size_mb(filepath)
        if file_size > max_file_size_mb:
            skipped_files.append(f"{filename} (size: {file_size:.1f}MB)")
            continue
            
        found_files.append((filepath, file_size))
    
    # Sort by importance (critical first, then alphabetical)
    def sort_key(item):
        filepath, _ = item
        filename = filepath.name
        if filename in CRITICAL_FILES:
            return (0, CRITICAL_FILES.index(filename))
        elif filename in IMPORTANT_FILES:
            return (1, IMPORTANT_FILES.index(filename))
        else:
            return (2, filename)
    
    found_files.sort(key=sort_key)
    
    # Create the context package
    total_size = sum(size for _, size in found_files)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Visual Token Diffusion LM - Context Package\n\n")
        f.write(f"**Generated from:** `{root_path.resolve()}`\n")
        f.write(f"**Total files:** {len(found_files)}\n")
        f.write(f"**Total size:** {total_size:.2f}MB\n\n")
        
        f.write("## 🎯 Project Overview\n")
        f.write("This is a novel approach to language modeling that represents tokens as visual patterns ")
        f.write("in a 5x5 grid with 3 colors, then uses diffusion models to generate new text. ")
        f.write("The project has breakthrough findings in semantic grounding and anti-collapse techniques.\n\n")
        
        if missing_files:
            f.write("## ⚠️ Missing Files\n")
            for filename in missing_files:
                f.write(f"- `{filename}`\n")
            f.write("\n")
            
        if skipped_files:
            f.write("## ⏭️ Skipped Files\n")
            for filename in skipped_files:
                f.write(f"- `{filename}`\n")
            f.write("\n")
        
        f.write("## 📁 File Contents\n\n")
        
        for filepath, file_size in found_files:
            relative_path = filepath.relative_to(root_path)
            f.write(f"### `{relative_path}` ({file_size:.2f}MB)\n\n")
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    
                # Add syntax highlighting based on extension
                ext = filepath.suffix.lower()
                if ext == '.py':
                    f.write("```python\n")
                elif ext == '.md':
                    f.write("```markdown\n")
                elif ext in ['.txt', '.yml', '.yaml']:
                    f.write("```yaml\n" if ext in ['.yml', '.yaml'] else "```text\n")
                else:
                    f.write("```\n")
                    
                f.write(content)
                f.write("\n```\n\n")
                
            except Exception as e:
                f.write(f"*Error reading file: {e}*\n\n")
    
    # Print summary
    print(f"✅ Context package created: {output_file}")
    print(f"📊 Included {len(found_files)} files ({total_size:.2f}MB total)")
    
    if missing_files:
        print(f"⚠️  Missing {len(missing_files)} files: {', '.join(missing_files)}")
    if skipped_files:
        print(f"⏭️  Skipped {len(skipped_files)} files")
    
    # Estimate token count (rough approximation: 1 token ≈ 4 characters)
    with open(output_file, 'r', encoding='utf-8') as f:
        char_count = len(f.read())
    estimated_tokens = char_count // 4
    print(f"📝 Estimated tokens: ~{estimated_tokens:,} (rough approximation)")

def main():
    parser = argparse.ArgumentParser(description="Create token-efficient context package")
    parser.add_argument("--root", "-r", default=".", help="Project root directory")
    parser.add_argument("--output", "-o", default="context_package.md", help="Output file name")
    parser.add_argument("--include", nargs="+", help="Specific files to include (overrides defaults)")
    parser.add_argument("--exclude", nargs="+", help="Additional files to exclude")
    parser.add_argument("--max-size", type=float, default=2.0, help="Max file size in MB")
    parser.add_argument("--list-files", action="store_true", help="List available files and exit")
    
    args = parser.parse_args()
    
    if args.list_files:
        root_path = Path(args.root)
        print(f"Available files in {root_path.resolve()}:")
        print("\n🔥 CRITICAL FILES:")
        for f in CRITICAL_FILES:
            exists = "✅" if (root_path / f).exists() else "❌"
            print(f"  {exists} {f}")
        print("\n📖 IMPORTANT FILES:")
        for f in IMPORTANT_FILES:
            exists = "✅" if (root_path / f).exists() else "❌"
            print(f"  {exists} {f}")
        print("\n📄 CONTEXT FILES:")
        for f in CONTEXT_FILES:
            exists = "✅" if (root_path / f).exists() else "❌"
            print(f"  {exists} {f}")
        return
    
    create_context_package(
        root_dir=args.root,
        include_files=args.include,
        exclude_files=args.exclude,
        max_file_size_mb=args.max_size,
        output_file=args.output
    )

if __name__ == "__main__":
    main()