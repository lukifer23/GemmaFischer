#!/usr/bin/env python3
"""
Format chess datasets for training compatibility.

This script converts datasets to the proper format for training:
1. Converts conversations to chat template format
2. Standardizes all datasets to use consistent formatting
3. Ensures compatibility with the training pipeline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class TrainingDataFormatter:
    """Format chess datasets for training compatibility."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.datasets_dir = data_dir / "datasets"
        self.formatted_dir = data_dir / "formatted"
        self.formatted_dir.mkdir(parents=True, exist_ok=True)
    
    def format_conversations_to_chat_template(self, conversations: List[Dict[str, str]]) -> str:
        """Convert conversations to chat template format."""
        # Gemma-3 chat template format
        formatted_parts = []
        
        for message in conversations:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'system':
                formatted_parts.append(f"<start_of_turn>system\n{content}<end_of_turn>")
            elif role == 'user':
                formatted_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == 'assistant':
                formatted_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        
        return '\n'.join(formatted_parts)
    
    def format_simple_text(self, text: str) -> str:
        """Format simple text to chat template format."""
        # Try to extract question and answer from text
        if "Question:" in text and "Answer:" in text:
            parts = text.split("Answer:", 1)
            if len(parts) == 2:
                question = parts[0].replace("Question:", "").strip()
                answer = parts[1].strip()
                
                return self.format_conversations_to_chat_template([
                    {"role": "system", "content": "You are a chess tutor. Provide clear, educational answers about chess."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ])
        
        # If no clear Q&A format, treat as general text
        return self.format_conversations_to_chat_template([
            {"role": "system", "content": "You are a chess tutor."},
            {"role": "user", "content": "Please help with this chess question."},
            {"role": "assistant", "content": text}
        ])
    
    def format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single example for training."""
        formatted_example = {
            "text": "",
            "source": example.get("source", "unknown"),
            "category": example.get("category", "general"),
            "difficulty": example.get("difficulty", "intermediate")
        }
        
        # Handle different input formats
        if "conversations" in example and example["conversations"]:
            # Use structured conversations
            formatted_example["text"] = self.format_conversations_to_chat_template(example["conversations"])
        elif "text" in example:
            # Use existing text field
            formatted_example["text"] = self.format_simple_text(example["text"])
        else:
            # Skip malformed examples
            return None
        
        # Preserve additional metadata
        for key in ["fen", "tactical_motif", "reasoning_steps", "rating"]:
            if key in example:
                formatted_example[key] = example[key]
        
        return formatted_example
    
    def format_dataset(self, input_file: Path, output_file: Path) -> Dict[str, Any]:
        """Format a dataset file for training."""
        print(f"🎯 Formatting dataset: {input_file.name}")
        
        formatted_examples = []
        skipped_examples = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    formatted_example = self.format_example(example)
                    
                    if formatted_example:
                        formatted_examples.append(formatted_example)
                    else:
                        skipped_examples += 1
                        
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  Skipping malformed JSON on line {line_num}: {e}")
                    skipped_examples += 1
                except Exception as e:
                    print(f"   ⚠️  Error processing line {line_num}: {e}")
                    skipped_examples += 1
        
        # Write formatted dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in formatted_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Create metadata
        metadata = {
            "source_file": str(input_file),
            "formatted_file": str(output_file),
            "total_examples": len(formatted_examples),
            "skipped_examples": skipped_examples,
            "format_date": datetime.now().isoformat(),
            "format_version": "1.0"
        }
        
        metadata_file = output_file.with_suffix('.metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Formatted {len(formatted_examples)} examples")
        print(f"   ⚠️  Skipped {skipped_examples} examples")
        print(f"   📁 Output: {output_file}")
        
        return metadata
    
    def format_all_datasets(self) -> Dict[str, Any]:
        """Format all datasets in the datasets directory."""
        print("🚀 Starting dataset formatting for training...")
        
        dataset_files = list(self.datasets_dir.glob("*.jsonl"))
        if not dataset_files:
            print("❌ No JSONL datasets found in datasets directory")
            return {}
        
        results = {}
        total_examples = 0
        
        for dataset_file in dataset_files:
            if dataset_file.name.startswith('formatted_'):
                continue  # Skip already formatted files
            
            output_file = self.formatted_dir / f"formatted_{dataset_file.name}"
            metadata = self.format_dataset(dataset_file, output_file)
            results[dataset_file.name] = metadata
            total_examples += metadata.get('total_examples', 0)
        
        # Create combined formatted dataset
        self.create_combined_formatted_dataset()
        
        # Create summary
        summary = {
            "format_date": datetime.now().isoformat(),
            "total_datasets": len(results),
            "total_examples": total_examples,
            "formatted_datasets": results
        }
        
        summary_file = self.formatted_dir / "formatting_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Formatting complete!")
        print(f"📊 Total examples formatted: {total_examples:,}")
        print(f"📁 Formatted datasets: {self.formatted_dir}")
        print(f"📋 Summary: {summary_file}")
        
        return summary
    
    def create_combined_formatted_dataset(self):
        """Create a combined formatted dataset for training."""
        print("🎯 Creating combined formatted dataset...")
        
        formatted_files = list(self.formatted_dir.glob("formatted_*.jsonl"))
        if not formatted_files:
            print("   ⚠️  No formatted datasets found")
            return
        
        combined_examples = []
        source_counts = {}
        
        for file_path in formatted_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line.strip())
                    combined_examples.append(example)
                    
                    source = example.get('source', 'unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
        
        # Write combined dataset
        combined_file = self.formatted_dir / "formatted_combined_dataset.jsonl"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for example in combined_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Create metadata
        combined_metadata = {
            "total_examples": len(combined_examples),
            "source_counts": source_counts,
            "creation_date": datetime.now().isoformat(),
            "source_files": [str(f) for f in formatted_files]
        }
        
        metadata_file = combined_file.with_suffix('.metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(combined_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Combined dataset created: {combined_file}")
        print(f"   📊 Total examples: {len(combined_examples):,}")
        print(f"   📋 Source breakdown: {source_counts}")


def main():
    parser = argparse.ArgumentParser(description="Format chess datasets for training")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), 
                       help="Data directory path")
    parser.add_argument("--input", type=Path, 
                       help="Specific dataset file to format")
    parser.add_argument("--output", type=Path,
                       help="Output file path (for single file formatting)")
    
    args = parser.parse_args()
    
    formatter = TrainingDataFormatter(args.data_dir)
    
    if args.input:
        # Format single file
        if not args.output:
            args.output = formatter.formatted_dir / f"formatted_{args.input.name}"
        
        metadata = formatter.format_dataset(args.input, args.output)
        print(f"✅ Single file formatting complete!")
    else:
        # Format all datasets
        summary = formatter.format_all_datasets()
        print(f"✅ All datasets formatted successfully!")


if __name__ == "__main__":
    main()
