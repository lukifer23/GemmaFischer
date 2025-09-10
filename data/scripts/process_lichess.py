#!/usr/bin/env python3
"""
Lichess Data Processor

Processes raw Lichess datasets into training-ready formats:
- Puzzles: Convert to Q&A format with tactical explanations
- Games: Extract positions and moves for training
- Studies: Convert to educational content

Supports filtering by rating, themes, and quality metrics.
"""

import argparse
import csv
import json
import gzip
import zstandard as zstd
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
import chess
import chess.pgn
from datetime import datetime
import random


class LichessProcessor:
    """Process Lichess datasets into training formats."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.datasets_dir = data_dir / "datasets"
        
        # Create directories
        for dir_path in [self.processed_dir, self.datasets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process_puzzles(self, input_file: Path, output_file: Path, 
                       min_rating: int = 1000, max_rating: int = 2000,
                       max_puzzles: int = 100000) -> Dict[str, Any]:
        """Process Lichess puzzle database into Q&A format."""
        print(f"🎯 Processing Lichess puzzles from {input_file}")
        print(f"📊 Filter: rating {min_rating}-{max_rating}, max {max_puzzles:,} puzzles")
        
        if not input_file.exists():
            print(f"❌ Input file not found: {input_file}")
            return {}
        
        # Decompress if needed
        if input_file.suffix == '.zst':
            print("   🔓 Decompressing .zst file...")
            temp_file = input_file.with_suffix('.csv')
            with open(input_file, 'rb') as f_in:
                dctx = zstd.ZstdDecompressor()
                with open(temp_file, 'wb') as f_out:
                    dctx.copy_stream(f_in, f_out)
            input_file = temp_file
        
        processed_count = 0
        total_count = 0
        quality_stats = {
            'total_processed': 0,
            'rating_distribution': {},
            'theme_distribution': {},
            'quality_scores': []
        }
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            reader = csv.DictReader(f_in)
            
            for row in reader:
                total_count += 1
                
                try:
                    # Parse puzzle data
                    puzzle_id = row.get('PuzzleId', '')
                    fen = row.get('FEN', '')
                    moves = row.get('Moves', '')
                    rating = int(row.get('Rating', 0))
                    themes = row.get('Themes', '')
                    popularity = int(row.get('Popularity', 0))
                    nb_plays = int(row.get('NbPlays', 0))
                    
                    # Filter by rating
                    if rating < min_rating or rating > max_rating:
                        continue
                    
                    # Validate FEN
                    try:
                        board = chess.Board(fen)
                    except:
                        continue
                    
                    # Parse moves
                    move_list = moves.split()
                    if not move_list:
                        continue
                    
                    best_move = move_list[0]
                    
                    # Validate move
                    try:
                        move = chess.Move.from_uci(best_move)
                        if move not in board.legal_moves:
                            continue
                    except:
                        continue
                    
                    # Generate question and answer
                    question, answer = self._create_puzzle_qa(fen, best_move, themes, rating)
                    
                    # Create training example
                    training_example = {
                        "text": f"Question: {question}\nAnswer: {answer}",
                        "conversations": [
                            {"role": "system", "content": "You are a chess tactics tutor."},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ],
                        "category": "tactical_puzzle",
                        "difficulty": self._get_difficulty_level(rating),
                        "rating": rating,
                        "fen": fen,
                        "best_move": best_move,
                        "themes": themes.split(',') if themes else [],
                        "puzzle_id": puzzle_id,
                        "popularity": popularity,
                        "nb_plays": nb_plays,
                        "source": "lichess_puzzles"
                    }
                    
                    # Write to output
                    f_out.write(json.dumps(training_example, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                    # Update stats
                    quality_stats['total_processed'] += 1
                    quality_stats['rating_distribution'][rating] = quality_stats['rating_distribution'].get(rating, 0) + 1
                    
                    for theme in themes.split(','):
                        if theme.strip():
                            quality_stats['theme_distribution'][theme.strip()] = quality_stats['theme_distribution'].get(theme.strip(), 0) + 1
                    
                    # Quality score based on popularity and plays
                    quality_score = min(10, (popularity / 100) + (nb_plays / 1000))
                    quality_stats['quality_scores'].append(quality_score)
                    
                    if processed_count >= max_puzzles:
                        break
                    
                    if processed_count % 10000 == 0:
                        print(f"   📊 Processed {processed_count:,} puzzles...")
                
                except Exception as e:
                    continue
        
        # Calculate final stats
        quality_stats['average_quality'] = sum(quality_stats['quality_scores']) / len(quality_stats['quality_scores']) if quality_stats['quality_scores'] else 0
        quality_stats['processing_date'] = datetime.now().isoformat()
        
        print(f"✅ Puzzle processing complete!")
        print(f"   📊 Processed: {processed_count:,} / {total_count:,} puzzles")
        print(f"   📁 Output: {output_file}")
        print(f"   🎯 Average quality: {quality_stats['average_quality']:.2f}/10")
        
        return quality_stats
    
    def _create_puzzle_qa(self, fen: str, best_move: str, themes: str, rating: int) -> tuple:
        """Create question and answer for a puzzle."""
        # Parse themes
        theme_list = [t.strip().lower() for t in themes.split(',') if t.strip()]
        
        # Create question
        question = f"FEN: {fen}\nFind the best tactical move in this position."
        
        # Create answer with explanation
        explanation = self._get_tactical_explanation(theme_list, rating)
        answer = f"{explanation}\n\nBest move: {best_move}"
        
        return question, answer
    
    def _get_tactical_explanation(self, themes: List[str], rating: int) -> str:
        """Generate tactical explanation based on themes and rating."""
        if not themes:
            return "Find the best tactical continuation."
        
        primary_theme = themes[0]
        
        explanations = {
            'fork': "This creates a fork, attacking two pieces simultaneously.",
            'pin': "This exploits a pin to win material or create threats.",
            'skewer': "This uses a skewer to expose a more valuable piece.",
            'discovered_attack': "This unleashes a discovered attack for material gain.",
            'double_attack': "This creates a double attack, overloading the defense.",
            'deflection': "This deflects a key defender from an important square.",
            'decoy': "This decoys a piece onto a vulnerable square.",
            'clearance': "This clears a line for a decisive tactical blow.",
            'zwischenzug': "This inserts an intermediate move that changes the outcome.",
            'sacrifice': "This temporary sacrifice opens lines for a winning combination.",
            'back_rank_mate': "This threatens back-rank mate, forcing material gain.",
            'mate_in_1': "This delivers checkmate in one move.",
            'mate_in_2': "This sets up a forced mate in two moves.",
            'mate_in_3': "This creates a forced mate in three moves.",
            'mate_in_4': "This leads to a forced mate in four moves.",
            'mate_in_5': "This creates a complex mating combination."
        }
        
        explanation = explanations.get(primary_theme, "This converts a tactical motif into material or positional advantage.")
        
        # Add difficulty context
        if rating >= 1800:
            explanation += " This is an advanced tactical pattern requiring precise calculation."
        elif rating >= 1400:
            explanation += " This is an intermediate tactical concept."
        else:
            explanation += " This is a fundamental tactical pattern."
        
        return explanation
    
    def _get_difficulty_level(self, rating: int) -> str:
        """Convert rating to difficulty level."""
        if rating >= 1800:
            return "advanced"
        elif rating >= 1400:
            return "intermediate"
        else:
            return "beginner"
    
    def process_games(self, input_file: Path, output_file: Path, 
                     min_rating: int = 1800, max_games: int = 10000) -> Dict[str, Any]:
        """Process Lichess game database into training format."""
        print(f"🎯 Processing Lichess games from {input_file}")
        print(f"📊 Filter: min rating {min_rating}, max {max_games:,} games")
        
        if not input_file.exists():
            print(f"❌ Input file not found: {input_file}")
            return {}
        
        # Decompress if needed
        if input_file.suffix == '.zst':
            print("   🔓 Decompressing .zst file...")
            temp_file = input_file.with_suffix('.pgn')
            with open(input_file, 'rb') as f_in:
                dctx = zstd.ZstdDecompressor()
                with open(temp_file, 'wb') as f_out:
                    dctx.copy_stream(f_in, f_out)
            input_file = temp_file
        
        processed_count = 0
        total_count = 0
        quality_stats = {
            'total_processed': 0,
            'rating_distribution': {},
            'opening_distribution': {},
            'game_length_distribution': {}
        }
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            while processed_count < max_games:
                try:
                    game = chess.pgn.read_game(f_in)
                    if game is None:
                        break
                    
                    total_count += 1
                    
                    # Extract game info
                    white_rating = int(game.headers.get('WhiteElo', 0))
                    black_rating = int(game.headers.get('BlackElo', 0))
                    result = game.headers.get('Result', '')
                    eco = game.headers.get('ECO', '')
                    
                    # Filter by rating
                    if white_rating < min_rating or black_rating < min_rating:
                        continue
                    
                    # Process game moves
                    board = chess.Board()
                    move_count = 0
                    
                    for node in game.mainline():
                        if node.move is None:
                            break
                        
                        move_count += 1
                        
                        # Create position-based training example
                        position_fen = board.fen()
                        move_uci = node.move.uci()
                        
                        # Generate question and answer
                        question = f"FEN: {position_fen}\nWhat is the best move in this position?"
                        answer = f"The best move is {move_uci}."
                        
                        training_example = {
                            "text": f"Question: {question}\nAnswer: {answer}",
                            "conversations": [
                                {"role": "system", "content": "You are a chess engine."},
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer}
                            ],
                            "category": "move_prediction",
                            "difficulty": "intermediate",
                            "fen": position_fen,
                            "best_move": move_uci,
                            "game_info": {
                                "white_rating": white_rating,
                                "black_rating": black_rating,
                                "result": result,
                                "eco": eco,
                                "move_number": move_count
                            },
                            "source": "lichess_games"
                        }
                        
                        f_out.write(json.dumps(training_example, ensure_ascii=False) + '\n')
                        
                        board.push(node.move)
                    
                    processed_count += 1
                    
                    # Update stats
                    quality_stats['total_processed'] += 1
                    avg_rating = (white_rating + black_rating) // 2
                    quality_stats['rating_distribution'][avg_rating] = quality_stats['rating_distribution'].get(avg_rating, 0) + 1
                    quality_stats['opening_distribution'][eco] = quality_stats['opening_distribution'].get(eco, 0) + 1
                    quality_stats['game_length_distribution'][move_count] = quality_stats['game_length_distribution'].get(move_count, 0) + 1
                    
                    if processed_count % 1000 == 0:
                        print(f"   📊 Processed {processed_count:,} games...")
                
                except Exception as e:
                    continue
        
        quality_stats['processing_date'] = datetime.now().isoformat()
        
        print(f"✅ Game processing complete!")
        print(f"   📊 Processed: {processed_count:,} / {total_count:,} games")
        print(f"   📁 Output: {output_file}")
        
        return quality_stats
    
    def create_combined_dataset(self, puzzle_file: Path, game_file: Path, 
                               output_file: Path, puzzle_ratio: float = 0.7) -> Dict[str, Any]:
        """Combine puzzle and game data into a balanced training dataset."""
        print(f"🎯 Creating combined dataset from puzzles and games")
        print(f"📊 Puzzle ratio: {puzzle_ratio:.1%}")
        
        combined_data = []
        puzzle_count = 0
        game_count = 0
        
        # Read puzzle data
        if puzzle_file.exists():
            with open(puzzle_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        puzzle_count += 1
                        combined_data.append(json.loads(line))
        
        # Read game data
        if game_file.exists():
            with open(game_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        game_count += 1
                        combined_data.append(json.loads(line))
        
        # Shuffle and balance
        random.shuffle(combined_data)
        
        # Write combined dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in combined_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        stats = {
            'total_examples': len(combined_data),
            'puzzle_examples': puzzle_count,
            'game_examples': game_count,
            'puzzle_ratio': puzzle_count / len(combined_data) if combined_data else 0,
            'creation_date': datetime.now().isoformat()
        }
        
        print(f"✅ Combined dataset created!")
        print(f"   📊 Total examples: {stats['total_examples']:,}")
        print(f"   🧩 Puzzles: {stats['puzzle_examples']:,}")
        print(f"   🎮 Games: {stats['game_examples']:,}")
        print(f"   📁 Output: {output_file}")
        
        return stats


def main():
    """Main entry point for Lichess data processing."""
    parser = argparse.ArgumentParser(description="Process Lichess datasets")
    parser.add_argument("--data_dir", default="data", help="Data directory path")
    parser.add_argument("--input", help="Input file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--type", choices=["puzzles", "games", "combined"], required=True, help="Processing type")
    parser.add_argument("--min_rating", type=int, default=1000, help="Minimum rating filter")
    parser.add_argument("--max_rating", type=int, default=2000, help="Maximum rating filter")
    parser.add_argument("--max_items", type=int, default=100000, help="Maximum items to process")
    parser.add_argument("--include_games", action="store_true", help="Include games in combined dataset")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    processor = LichessProcessor(data_dir)
    
    if args.type == "puzzles":
        input_file = Path(args.input) if args.input else data_dir / "raw" / "lichess" / "puzzles" / "lichess_puzzles.csv"
        output_file = Path(args.output) if args.output else data_dir / "datasets" / f"lichess_puzzles_{args.min_rating}_{args.max_rating}.jsonl"
        
        stats = processor.process_puzzles(input_file, output_file, args.min_rating, args.max_rating, args.max_items)
        
        # Save processing stats
        stats_file = output_file.with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    elif args.type == "games":
        input_file = Path(args.input) if args.input else data_dir / "raw" / "lichess" / "games" / "lichess_games_2024-01.pgn"
        output_file = Path(args.output) if args.output else data_dir / "datasets" / f"lichess_games_{args.min_rating}.jsonl"
        
        stats = processor.process_games(input_file, output_file, args.min_rating, args.max_items)
        
        # Save processing stats
        stats_file = output_file.with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    elif args.type == "combined":
        puzzle_file = data_dir / "datasets" / f"lichess_puzzles_{args.min_rating}_{args.max_rating}.jsonl"
        game_file = data_dir / "datasets" / f"lichess_games_{args.min_rating}.jsonl" if args.include_games else None
        output_file = Path(args.output) if args.output else data_dir / "datasets" / "lichess_combined.jsonl"
        
        stats = processor.create_combined_dataset(puzzle_file, game_file, output_file)
        
        # Save processing stats
        stats_file = output_file.with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
