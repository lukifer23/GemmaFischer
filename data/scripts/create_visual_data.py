#!/usr/bin/env python3
"""
Visual Training Data Generator

Creates visual training data for the chess vision module:
- Renders chess positions from FEN strings
- Generates board images with various styles
- Creates piece recognition training data
- Produces board detection examples

Supports multiple board styles, piece sets, and orientations.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import chess
import chess.svg
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from datetime import datetime


class VisualDataGenerator:
    """Generate visual training data for chess board recognition."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.datasets_dir = data_dir / "datasets"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.datasets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Visual data directories
        self.visual_dir = self.raw_dir / "visual"
        self.visual_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different visual data types
        self.board_positions_dir = self.visual_dir / "board_positions"
        self.piece_recognition_dir = self.visual_dir / "piece_recognition"
        self.board_detection_dir = self.visual_dir / "board_detection"
        self.synthetic_boards_dir = self.visual_dir / "synthetic_boards"
        
        for dir_path in [self.board_positions_dir, self.piece_recognition_dir, 
                        self.board_detection_dir, self.synthetic_boards_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def generate_board_positions(self, num_positions: int = 1000) -> List[Dict[str, Any]]:
        """Generate board position images from FEN strings."""
        print(f"🎯 Generating {num_positions:,} board position images...")
        
        examples = []
        
        # Common chess positions to render
        positions = [
            # Starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            
            # Common opening positions
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",  # e4 e5
            "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 1",  # e4 e5 d4
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",  # e4 e5 Nf3
            
            # Tactical positions
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            
            # Endgame positions
            "8/8/8/8/8/8/4P3/4K3 w - - 0 1",  # King and pawn
            "8/8/8/8/8/8/4P3/4K2R w - - 0 1",  # Rook and pawn
            "8/8/8/8/8/8/4K3/4k3 w - - 0 1",  # King vs King
            
            # Complex positions
            "r3k2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R3K2R w KQkq - 0 1",
            "r3k2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R3K2R w KQkq - 0 1"
        ]
        
        for i in range(num_positions):
            # Select a position (cycle through the list)
            fen = positions[i % len(positions)]
            
            # Generate variations by adding random moves
            if i > len(positions):
                fen = self._generate_random_position()
            
            # Create board image
            image_path = self.board_positions_dir / f"position_{i:06d}.png"
            success = self._render_board_image(fen, image_path)
            
            if success:
                # Create training example
                question = "What is the position on this chess board?"
                answer = f"This is the position: {fen}"
                
                example = {
                    "text": f"Question: {question}\nAnswer: {answer}",
                    "conversations": [
                        {"role": "system", "content": "You are a chess vision expert. Analyze board positions from images."},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ],
                    "category": "visual_recognition",
                    "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                    "image_path": str(image_path),
                    "fen": fen,
                    "board_type": "standard",
                    "source": "generated_visual"
                }
                
                examples.append(example)
        
        print(f"✅ Generated {len(examples):,} board position images")
        return examples
    
    def generate_piece_recognition_data(self, num_examples: int = 500) -> List[Dict[str, Any]]:
        """Generate piece recognition training data."""
        print(f"🎯 Generating {num_examples:,} piece recognition examples...")
        
        examples = []
        
        # Piece types and their representations
        pieces = {
            'K': {'name': 'White King', 'symbol': '♔'},
            'Q': {'name': 'White Queen', 'symbol': '♕'},
            'R': {'name': 'White Rook', 'symbol': '♖'},
            'B': {'name': 'White Bishop', 'symbol': '♗'},
            'N': {'name': 'White Knight', 'symbol': '♘'},
            'P': {'name': 'White Pawn', 'symbol': '♙'},
            'k': {'name': 'Black King', 'symbol': '♚'},
            'q': {'name': 'Black Queen', 'symbol': '♛'},
            'r': {'name': 'Black Rook', 'symbol': '♜'},
            'b': {'name': 'Black Bishop', 'symbol': '♝'},
            'n': {'name': 'Black Knight', 'symbol': '♞'},
            'p': {'name': 'Black Pawn', 'symbol': '♟'}
        }
        
        for i in range(num_examples):
            piece_char = random.choice(list(pieces.keys()))
            piece_info = pieces[piece_char]
            
            # Create piece image
            image_path = self.piece_recognition_dir / f"piece_{piece_char}_{i:06d}.png"
            success = self._render_piece_image(piece_char, piece_info, image_path)
            
            if success:
                # Create training example
                question = "What piece is shown in this image?"
                answer = f"This is a {piece_info['name']} ({piece_info['symbol']})."
                
                example = {
                    "text": f"Question: {question}\nAnswer: {answer}",
                    "conversations": [
                        {"role": "system", "content": "You are a chess piece recognition expert."},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ],
                    "category": "piece_recognition",
                    "difficulty": "beginner",
                    "image_path": str(image_path),
                    "piece_type": piece_char,
                    "piece_name": piece_info['name'],
                    "piece_symbol": piece_info['symbol'],
                    "source": "generated_visual"
                }
                
                examples.append(example)
        
        print(f"✅ Generated {len(examples):,} piece recognition examples")
        return examples
    
    def generate_board_detection_data(self, num_examples: int = 300) -> List[Dict[str, Any]]:
        """Generate board detection training data with various orientations."""
        print(f"🎯 Generating {num_examples:,} board detection examples...")
        
        examples = []
        
        # Board orientations and styles
        orientations = ['normal', 'rotated_90', 'rotated_180', 'rotated_270']
        styles = ['wood', 'marble', 'digital', 'classic']
        
        for i in range(num_examples):
            orientation = random.choice(orientations)
            style = random.choice(styles)
            
            # Generate a position
            fen = self._generate_random_position()
            
            # Create board image with specific orientation and style
            image_path = self.board_detection_dir / f"board_{orientation}_{style}_{i:06d}.png"
            success = self._render_board_image(fen, image_path, orientation, style)
            
            if success:
                # Create training example
                question = "Detect and analyze the chess board in this image."
                answer = f"Board detected: {orientation} orientation, {style} style. Position: {fen}"
                
                example = {
                    "text": f"Question: {question}\nAnswer: {answer}",
                    "conversations": [
                        {"role": "system", "content": "You are a chess board detection expert."},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ],
                    "category": "board_detection",
                    "difficulty": random.choice(["intermediate", "advanced"]),
                    "image_path": str(image_path),
                    "fen": fen,
                    "orientation": orientation,
                    "style": style,
                    "source": "generated_visual"
                }
                
                examples.append(example)
        
        print(f"✅ Generated {len(examples):,} board detection examples")
        return examples
    
    def generate_synthetic_boards(self, num_examples: int = 200) -> List[Dict[str, Any]]:
        """Generate synthetic board positions for training."""
        print(f"🎯 Generating {num_examples:,} synthetic board examples...")
        
        examples = []
        
        for i in range(num_examples):
            # Generate a synthetic position
            fen = self._generate_synthetic_position()
            
            # Create board image
            image_path = self.synthetic_boards_dir / f"synthetic_{i:06d}.png"
            success = self._render_board_image(fen, image_path)
            
            if success:
                # Create training example
                question = "Analyze this synthetic chess position."
                answer = f"Synthetic position: {fen}. This is a computer-generated position for training purposes."
                
                example = {
                    "text": f"Question: {question}\nAnswer: {answer}",
                    "conversations": [
                        {"role": "system", "content": "You are a chess position analyst."},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ],
                    "category": "synthetic_analysis",
                    "difficulty": random.choice(["intermediate", "advanced"]),
                    "image_path": str(image_path),
                    "fen": fen,
                    "board_type": "synthetic",
                    "source": "generated_visual"
                }
                
                examples.append(example)
        
        print(f"✅ Generated {len(examples):,} synthetic board examples")
        return examples
    
    def _render_board_image(self, fen: str, output_path: Path, 
                          orientation: str = 'normal', style: str = 'classic') -> bool:
        """Render a chess board image from FEN string."""
        try:
            # Create a simple board representation
            board = chess.Board(fen)
            
            # Create image
            img_size = 512
            img = Image.new('RGB', (img_size, img_size), 'white')
            draw = ImageDraw.Draw(img)
            
            # Draw board squares
            square_size = img_size // 8
            
            for rank in range(8):
                for file in range(8):
                    x = file * square_size
                    y = rank * square_size
                    
                    # Alternate square colors
                    if (rank + file) % 2 == 0:
                        color = (240, 217, 181)  # Light square
                    else:
                        color = (181, 136, 99)   # Dark square
                    
                    draw.rectangle([x, y, x + square_size, y + square_size], fill=color)
            
            # Draw pieces (simplified representation)
            piece_symbols = {
                'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
                'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
            }
            
            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            for rank in range(8):
                for file in range(8):
                    square = chess.square(file, 7 - rank)  # Convert to chess square
                    piece = board.piece_at(square)
                    
                    if piece:
                        symbol = piece_symbols.get(piece.symbol(), '?')
                        x = file * square_size + square_size // 2 - 12
                        y = rank * square_size + square_size // 2 - 12
                        
                        # Draw piece symbol
                        draw.text((x, y), symbol, fill='black', font=font)
            
            # Apply orientation if needed
            if orientation == 'rotated_90':
                img = img.rotate(90, expand=True)
            elif orientation == 'rotated_180':
                img = img.rotate(180, expand=True)
            elif orientation == 'rotated_270':
                img = img.rotate(270, expand=True)
            
            # Save image
            img.save(output_path, 'PNG')
            return True
            
        except Exception as e:
            print(f"   ❌ Error rendering board image: {e}")
            return False
    
    def _render_piece_image(self, piece_char: str, piece_info: Dict[str, str], 
                          output_path: Path) -> bool:
        """Render a piece image."""
        try:
            # Create image
            img_size = 128
            img = Image.new('RGB', (img_size, img_size), 'white')
            draw = ImageDraw.Draw(img)
            
            # Draw piece symbol
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 48)
            except:
                font = ImageFont.load_default()
            
            # Center the piece
            bbox = draw.textbbox((0, 0), piece_info['symbol'], font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (img_size - text_width) // 2
            y = (img_size - text_height) // 2
            
            draw.text((x, y), piece_info['symbol'], fill='black', font=font)
            
            # Save image
            img.save(output_path, 'PNG')
            return True
            
        except Exception as e:
            print(f"   ❌ Error rendering piece image: {e}")
            return False
    
    def _generate_random_position(self) -> str:
        """Generate a random chess position."""
        # In practice, use real positions from databases
        positions = [
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",
            "r3k2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R3K2R w KQkq - 0 1"
        ]
        return random.choice(positions)
    
    def _generate_synthetic_position(self) -> str:
        """Generate a synthetic chess position."""
        # In practice, use more sophisticated position generation
        return self._generate_random_position()
    
    def save_visual_dataset(self, examples: List[Dict[str, Any]], output_file: Path) -> Dict[str, Any]:
        """Save visual training examples to file."""
        print(f"💾 Saving {len(examples):,} visual examples to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Create metadata
        metadata = {
            'total_examples': len(examples),
            'categories': {},
            'difficulty_distribution': {},
            'creation_date': datetime.now().isoformat(),
            'description': 'Visual training data for chess board recognition'
        }
        
        for example in examples:
            category = example['category']
            difficulty = example['difficulty']
            
            metadata['categories'][category] = metadata['categories'].get(category, 0) + 1
            metadata['difficulty_distribution'][difficulty] = metadata['difficulty_distribution'].get(difficulty, 0) + 1
        
        metadata_file = output_file.with_suffix('.metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Visual dataset saved!")
        print(f"   📁 File: {output_file}")
        print(f"   📊 Metadata: {metadata_file}")
        
        return metadata


def main():
    """Main entry point for visual data generation."""
    parser = argparse.ArgumentParser(description="Generate visual training data")
    parser.add_argument("--data_dir", default="data", help="Data directory path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--type", choices=["board_positions", "piece_recognition", "board_detection", "synthetic", "all"], 
                       default="all", help="Type of visual data to generate")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to generate")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    generator = VisualDataGenerator(data_dir)
    
    all_examples = []
    
    if args.type in ["board_positions", "all"]:
        board_examples = generator.generate_board_positions(args.num_examples)
        all_examples.extend(board_examples)
    
    if args.type in ["piece_recognition", "all"]:
        piece_examples = generator.generate_piece_recognition_data(args.num_examples // 2)
        all_examples.extend(piece_examples)
    
    if args.type in ["board_detection", "all"]:
        detection_examples = generator.generate_board_detection_data(args.num_examples // 3)
        all_examples.extend(detection_examples)
    
    if args.type in ["synthetic", "all"]:
        synthetic_examples = generator.generate_synthetic_boards(args.num_examples // 5)
        all_examples.extend(synthetic_examples)
    
    # Save combined dataset
    output_file = Path(args.output) if args.output else data_dir / "datasets" / "visual_training_data.jsonl"
    metadata = generator.save_visual_dataset(all_examples, output_file)
    
    print(f"\n🎉 Visual data generation complete!")
    print(f"   📊 Total examples: {len(all_examples):,}")
    print(f"   📁 Output: {output_file}")


if __name__ == "__main__":
    main()
