#!/usr/bin/env python3
"""
Test training compatibility with formatted datasets.

This script validates that our formatted datasets are compatible
with the training pipeline without actually running training.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def test_dataset_loading(dataset_path: Path) -> Dict[str, Any]:
    """Test loading and basic validation of a dataset."""
    print(f"🧪 Testing dataset: {dataset_path}")
    
    if not dataset_path.exists():
        return {"error": f"Dataset file not found: {dataset_path}"}
    
    # Test basic file properties
    file_size = dataset_path.stat().st_size
    print(f"   📊 File size: {file_size / (1024*1024):.1f} MB")
    
    # Test loading examples
    examples = []
    errors = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                examples.append(example)
                
                # Validate required fields
                if 'text' not in example:
                    errors.append(f"Line {line_num}: Missing 'text' field")
                
                # Check text length (training will truncate long texts)
                text_length = len(example.get('text', ''))
                if text_length < 10:
                    errors.append(f"Line {line_num}: Text too short ({text_length} chars)")
                elif text_length > 5000:  # Only flag extremely long texts
                    errors.append(f"Line {line_num}: Text extremely long ({text_length} chars)")
                
                # Check for chat template format
                text = example.get('text', '')
                if '<start_of_turn>' not in text:
                    errors.append(f"Line {line_num}: Missing chat template format")
                
                if line_num >= 100:  # Only test first 100 examples
                    break
                    
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON decode error - {e}")
            except Exception as e:
                errors.append(f"Line {line_num}: Unexpected error - {e}")
    
    # Calculate statistics
    if examples:
        text_lengths = [len(ex.get('text', '')) for ex in examples]
        avg_length = sum(text_lengths) / len(text_lengths)
        min_length = min(text_lengths)
        max_length = max(text_lengths)
        
        # Check sources
        sources = {}
        for ex in examples:
            source = ex.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        # Check categories
        categories = {}
        for ex in examples:
            category = ex.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
    else:
        avg_length = min_length = max_length = 0
        sources = categories = {}
    
    result = {
        "file_path": str(dataset_path),
        "file_size_mb": file_size / (1024*1024),
        "examples_tested": len(examples),
        "errors": errors,
        "text_length_stats": {
            "average": avg_length,
            "min": min_length,
            "max": max_length
        },
        "sources": sources,
        "categories": categories,
        "compatible": len(errors) == 0
    }
    
    print(f"   ✅ Tested {len(examples)} examples")
    print(f"   📊 Text length: {avg_length:.0f} avg, {min_length}-{max_length} range")
    print(f"   📋 Sources: {list(sources.keys())}")
    print(f"   📋 Categories: {list(categories.keys())}")
    
    if errors:
        print(f"   ❌ {len(errors)} errors found:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"      - {error}")
        if len(errors) > 5:
            print(f"      ... and {len(errors) - 5} more errors")
    else:
        print(f"   ✅ No errors found - dataset is compatible!")
    
    return result


def main():
    """Test training compatibility for formatted datasets."""
    print("🧪 Testing Training Compatibility")
    print("=" * 50)
    
    # Test the main formatted dataset
    dataset_path = Path("data/formatted/formatted_combined_dataset.jsonl")
    result = test_dataset_loading(dataset_path)
    
    print("\n" + "=" * 50)
    print("📋 Compatibility Summary")
    print("=" * 50)
    
    if result.get("compatible", False):
        print("✅ Dataset is compatible with training pipeline!")
        print(f"📊 Ready for training with {result['examples_tested']} examples")
        print(f"📁 File: {result['file_path']}")
        print(f"💾 Size: {result['file_size_mb']:.1f} MB")
    else:
        print("❌ Dataset has compatibility issues:")
        for error in result.get("errors", []):
            print(f"   - {error}")
        sys.exit(1)
    
    # Test a few individual formatted datasets
    print("\n🔍 Testing Individual Datasets")
    print("-" * 30)
    
    formatted_dir = Path("data/formatted")
    test_files = [
        "formatted_cot_reasoning_examples.jsonl",
        "formatted_chess_finetune_expanded.jsonl",
        "formatted_chess_finetune_focused.jsonl"
    ]
    
    for filename in test_files:
        file_path = formatted_dir / filename
        if file_path.exists():
            test_dataset_loading(file_path)
            print()
    
    print("🎉 All compatibility tests completed!")


if __name__ == "__main__":
    main()
