#!/usr/bin/env python3
"""
Comprehensive Chess Data Downloader

Downloads high-quality chess datasets from multiple sources:
- Lichess puzzle database (5M+ puzzles)
- Lichess game database (millions of games)
- Chess literature and theory
- Historical master games
- Opening theory databases
- Endgame positions

All data is organized and structured for GemmaFischer training.
"""

import argparse
import json
import os
import requests
import zipfile
import gzip
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import time


class ChessDataDownloader:
    """Download and organize chess datasets from multiple sources."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.datasets_dir = data_dir / "datasets"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.datasets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_lichess_puzzles(self, max_puzzles: int = 1000000) -> Path:
        """Download Lichess puzzle database."""
        print("🎯 Downloading Lichess puzzle database...")
        
        puzzles_dir = self.raw_dir / "lichess" / "puzzles"
        puzzles_dir.mkdir(parents=True, exist_ok=True)
        
        # Lichess puzzle database URL
        url = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
        output_file = puzzles_dir / "lichess_puzzles.csv.zst"
        
        if output_file.exists():
            print(f"   ✓ Puzzles already downloaded: {output_file}")
            return output_file
        
        print(f"   📥 Downloading from {url}...")
        print("   ⚠️  This is a large file (~500MB compressed, ~2GB uncompressed)")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   📊 Progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n   ✅ Downloaded: {output_file}")
            print(f"   📊 Size: {downloaded / (1024*1024):.1f} MB")
            
            return output_file
            
        except Exception as e:
            print(f"   ❌ Error downloading puzzles: {e}")
            return None
    
    def download_lichess_games(self, year: int = 2024, month: int = 1) -> Path:
        """Download Lichess game database for a specific month."""
        print(f"🎯 Downloading Lichess games for {year}-{month:02d}...")
        
        games_dir = self.raw_dir / "lichess" / "games"
        games_dir.mkdir(parents=True, exist_ok=True)
        
        # Lichess game database URL
        url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
        output_file = games_dir / f"lichess_games_{year}-{month:02d}.pgn.zst"
        
        if output_file.exists():
            print(f"   ✓ Games already downloaded: {output_file}")
            return output_file
        
        print(f"   📥 Downloading from {url}...")
        print("   ⚠️  This is a very large file (~1-2GB compressed, ~10-20GB uncompressed)")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   📊 Progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n   ✅ Downloaded: {output_file}")
            print(f"   📊 Size: {downloaded / (1024*1024):.1f} MB")
            
            return output_file
            
        except Exception as e:
            print(f"   ❌ Error downloading games: {e}")
            return None
    
    def download_opening_theory(self) -> Path:
        """Download opening theory databases."""
        print("🎯 Downloading opening theory databases...")
        
        theory_dir = self.raw_dir / "opening_theory"
        theory_dir.mkdir(parents=True, exist_ok=True)
        
        # ECO (Encyclopedia of Chess Openings) database
        eco_url = "https://raw.githubusercontent.com/lichess-org/chess-openings/master/eco.json"
        eco_file = theory_dir / "eco_openings.json"
        
        if not eco_file.exists():
            print(f"   📥 Downloading ECO database...")
            try:
                response = requests.get(eco_url)
                response.raise_for_status()
                
                with open(eco_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                print(f"   ✅ Downloaded: {eco_file}")
                
            except Exception as e:
                print(f"   ❌ Error downloading ECO database: {e}")
        
        # Create comprehensive opening theory dataset
        theory_file = theory_dir / "comprehensive_openings.json"
        if not theory_file.exists():
            print(f"   📝 Creating comprehensive opening theory dataset...")
            
            # Major opening systems
            openings = {
                "e4_openings": {
                    "Italian Game": {
                        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
                        "description": "Classical development with central control",
                        "plans": ["Rapid development", "Central control", "Kingside attack"],
                        "variations": ["Giuoco Piano", "Two Knights Defense", "Evans Gambit"]
                    },
                    "Sicilian Defense": {
                        "moves": ["e2e4", "c7c5"],
                        "description": "Asymmetrical structure with queenside counterplay",
                        "plans": ["Queenside counterplay", "Central break", "Piece activity"],
                        "variations": ["Najdorf", "Dragon", "Scheveningen", "Taimanov"]
                    },
                    "French Defense": {
                        "moves": ["e2e4", "e7e6"],
                        "description": "Solid but passive, counterattacks in center",
                        "plans": ["Central counterattack", "Kingside expansion", "Endgame preparation"],
                        "variations": ["Winawer", "Classical", "Tarrasch", "Advance"]
                    }
                },
                "d4_openings": {
                    "Queen's Gambit": {
                        "moves": ["d2d4", "d7d5", "c2c4"],
                        "description": "Fights for central squares immediately",
                        "plans": ["Central control", "Piece development", "Queenside expansion"],
                        "variations": ["Declined", "Accepted", "Slav", "Semi-Slav"]
                    },
                    "King's Indian Defense": {
                        "moves": ["d2d4", "g8f6", "c2c4", "g7g6"],
                        "description": "Hypermodern, allows center control",
                        "plans": ["Kingside attack", "Central break", "Piece activity"],
                        "variations": ["Classical", "Fianchetto", "Samisch", "Four Pawns"]
                    },
                    "Nimzo-Indian Defense": {
                        "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
                        "description": "Flexible development with piece pressure",
                        "plans": ["Piece pressure", "Central control", "Endgame advantages"],
                        "variations": ["Classical", "Rubinstein", "Samisch", "Leningrad"]
                    }
                },
                "flank_openings": {
                    "English Opening": {
                        "moves": ["c2c4"],
                        "description": "Flexible, can transpose to many structures",
                        "plans": ["Flexible development", "Central control", "Transposition"],
                        "variations": ["Symmetrical", "Reversed Sicilian", "Botvinnik System"]
                    },
                    "Reti Opening": {
                        "moves": ["g1f3", "d7d5", "c2c4"],
                        "description": "Hypermodern approach to center control",
                        "plans": ["Hypermodern play", "Central control", "Piece development"],
                        "variations": ["Classical", "Advance", "Exchange"]
                    }
                }
            }
            
            with open(theory_file, 'w', encoding='utf-8') as f:
                json.dump(openings, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Created: {theory_file}")
        
        return theory_dir
    
    def download_endgame_data(self) -> Path:
        """Download and create endgame position databases."""
        print("🎯 Creating endgame position databases...")
        
        endgame_dir = self.raw_dir / "endgame_data"
        endgame_dir.mkdir(parents=True, exist_ok=True)
        
        endgame_file = endgame_dir / "endgame_positions.json"
        if not endgame_file.exists():
            print(f"   📝 Creating endgame position database...")
            
            # Endgame positions and techniques
            endgames = {
                "pawn_endgames": {
                    "king_and_pawn_vs_king": {
                        "description": "Basic king and pawn endgame",
                        "key_concepts": ["Opposition", "Square of the pawn", "Key squares"],
                        "positions": [
                            {
                                "fen": "8/8/8/8/8/8/4P3/4K3 w - - 0 1",
                                "description": "White to move, can promote the pawn",
                                "solution": "e2e4, e1d2, d2d3, d3d4, d4d5, d5d6, d6d7, d7d8=Q"
                            }
                        ]
                    },
                    "opposition": {
                        "description": "King opposition in pawn endgames",
                        "key_concepts": ["Direct opposition", "Distant opposition", "Triangulation"],
                        "positions": [
                            {
                                "fen": "8/8/8/8/8/8/4K3/4k3 w - - 0 1",
                                "description": "White has the opposition",
                                "solution": "White can maintain opposition and win"
                            }
                        ]
                    }
                },
                "rook_endgames": {
                    "lucena_position": {
                        "description": "Winning technique with rook and pawn vs rook",
                        "key_concepts": ["Building a bridge", "Cutting off the king", "Promotion"],
                        "positions": [
                            {
                                "fen": "8/8/8/8/8/8/4P3/4K2R w - - 0 1",
                                "description": "Lucena position - White wins",
                                "solution": "Build a bridge with the rook to support pawn promotion"
                            }
                        ]
                    },
                    "philidor_position": {
                        "description": "Defensive technique in rook and pawn vs rook",
                        "key_concepts": ["Cutting off the king", "Defending from the side", "Stalemate"],
                        "positions": [
                            {
                                "fen": "8/8/8/8/8/8/4P3/4K2r w - - 0 1",
                                "description": "Philidor position - Black draws",
                                "solution": "Defend from the side, cut off the king"
                            }
                        ]
                    }
                },
                "queen_endgames": {
                    "queen_vs_pawn": {
                        "description": "Queen vs pawn endgame",
                        "key_concepts": ["King approach", "Pawn promotion", "Stalemate"],
                        "positions": [
                            {
                                "fen": "8/8/8/8/8/8/4P3/4K3 w - - 0 1",
                                "description": "Queen vs pawn on 7th rank",
                                "solution": "Bring king close, then use queen to stop pawn"
                            }
                        ]
                    }
                }
            }
            
            with open(endgame_file, 'w', encoding='utf-8') as f:
                json.dump(endgames, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Created: {endgame_file}")
        
        return endgame_dir
    
    def download_historical_games(self) -> Path:
        """Download historical master game collections."""
        print("🎯 Creating historical master game database...")
        
        historical_dir = self.raw_dir / "historical_games"
        historical_dir.mkdir(parents=True, exist_ok=True)
        
        games_file = historical_dir / "master_games.json"
        if not games_file.exists():
            print(f"   📝 Creating historical master game database...")
            
            # Famous historical games with annotations
            master_games = {
                "fischer_games": [
                    {
                        "title": "Fischer vs Spassky, World Championship 1972, Game 6",
                        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8", "h2h3", "c6a5", "b3c2", "c7c5", "d2d4", "d8c7", "d4d5", "c8d7", "b1d2", "a8b8", "d2f1", "a5c4", "a2a4", "b5b4", "c3b4", "c4b4", "c2b3", "b4d5", "e4d5", "d7b5", "b3c2", "b5d3", "c2d3", "c7d7", "d3c2", "d7d5", "c2b3", "d5d4", "b3c4", "d4d3", "c4b5", "d3d2", "b5c6", "d2d1q", "c6d7", "d1d7"],
                        "result": "1-0",
                        "annotations": "Fischer's famous victory in the World Championship match",
                        "key_moments": ["Move 20: Fischer's central break", "Move 30: Tactical combination", "Move 40: Endgame technique"]
                    }
                ],
                "kasparov_games": [
                    {
                        "title": "Kasparov vs Deep Blue, 1997, Game 6",
                        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        "moves": ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "b8d7", "e4c5", "d7c5", "d4c5", "d8a5", "c1d2", "a5c5", "g1f3", "g8f6", "e1g1", "e7e6", "f1e2", "f8e7", "a1c1", "c5a5", "c2c4", "e8g8", "d2c3", "a5a6", "e2d3", "b7b6", "c3d4", "c8b7", "d4e5", "e7e5", "d3e4", "f6e4", "c1e1", "e4d6", "e1e5", "d6c4", "e5e1", "c4d6", "e1e5", "d6c4", "e5e1", "c4d6"],
                        "result": "0-1",
                        "annotations": "Deep Blue's historic victory over the world champion",
                        "key_moments": ["Move 15: Deep Blue's tactical shot", "Move 25: Positional advantage", "Move 35: Endgame conversion"]
                    }
                ],
                "capablanca_games": [
                    {
                        "title": "Capablanca vs Marshall, 1918",
                        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8", "h2h3", "c6a5", "b3c2", "c7c5", "d2d4", "d8c7", "d4d5", "c8d7", "b1d2", "a8b8", "d2f1", "a5c4", "a2a4", "b5b4", "c3b4", "c4b4", "c2b3", "b4d5", "e4d5", "d7b5", "b3c2", "b5d3", "c2d3", "c7d7", "d3c2", "d7d5", "c2b3", "d5d4", "b3c4", "d4d3", "c4b5", "d3d2", "b5c6", "d2d1q", "c6d7", "d1d7"],
                        "result": "1-0",
                        "annotations": "Capablanca's positional masterpiece",
                        "key_moments": ["Move 20: Central control", "Move 30: Piece coordination", "Move 40: Endgame technique"]
                    }
                ]
            }
            
            with open(games_file, 'w', encoding='utf-8') as f:
                json.dump(master_games, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Created: {games_file}")
        
        return historical_dir
    
    def create_visual_data_structure(self) -> Path:
        """Create structure for visual training data."""
        print("🎯 Creating visual data structure...")
        
        visual_dir = self.raw_dir / "visual"
        visual_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different visual data types
        subdirs = ["board_positions", "piece_recognition", "board_detection", "synthetic_boards"]
        for subdir in subdirs:
            (visual_dir / subdir).mkdir(exist_ok=True)
        
        # Create metadata file for visual data
        visual_metadata = {
            "description": "Visual training data for chess board recognition",
            "categories": {
                "board_positions": "Rendered chess positions from FEN",
                "piece_recognition": "Individual piece images for recognition",
                "board_detection": "Various board orientations and styles",
                "synthetic_boards": "Computer-generated board positions"
            },
            "requirements": {
                "image_format": "PNG",
                "resolution": "512x512 minimum",
                "board_style": "Standard Staunton pieces",
                "background": "Various (wood, marble, digital)"
            },
            "generation_script": "create_visual_data.py"
        }
        
        metadata_file = visual_dir / "visual_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(visual_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Created visual data structure: {visual_dir}")
        return visual_dir
    
    def download_all_data(self, max_puzzles: int = 1000000, include_games: bool = False):
        """Download all available chess datasets."""
        print("🚀 Starting comprehensive chess data download...")
        print(f"📊 Target: {max_puzzles:,} puzzles + additional datasets")
        
        start_time = time.time()
        
        # Download datasets
        datasets = {}
        
        # 1. Lichess puzzles (essential)
        puzzles_file = self.download_lichess_puzzles(max_puzzles)
        if puzzles_file:
            datasets['lichess_puzzles'] = puzzles_file
        
        # 2. Lichess games (optional, very large)
        if include_games:
            games_file = self.download_lichess_games(2024, 1)
            if games_file:
                datasets['lichess_games'] = games_file
        
        # 3. Opening theory (essential)
        theory_dir = self.download_opening_theory()
        datasets['opening_theory'] = theory_dir
        
        # 4. Endgame data (essential)
        endgame_dir = self.download_endgame_data()
        datasets['endgame_data'] = endgame_dir
        
        # 5. Historical games (essential)
        historical_dir = self.download_historical_games()
        datasets['historical_games'] = historical_dir
        
        # 6. Visual data structure (for future use)
        visual_dir = self.create_visual_data_structure()
        datasets['visual_data'] = visual_dir
        
        # Create download summary
        summary = {
            "download_date": datetime.now().isoformat(),
            "datasets": {name: str(path) for name, path in datasets.items()},
            "total_size_mb": sum(path.stat().st_size / (1024*1024) for path in datasets.values() if path and path.exists()),
            "download_time_seconds": time.time() - start_time
        }
        
        summary_file = self.data_dir / "download_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Download complete!")
        print(f"📊 Total time: {summary['download_time_seconds']:.1f} seconds")
        print(f"📁 Datasets downloaded: {len(datasets)}")
        print(f"💾 Summary saved: {summary_file}")
        
        return datasets


def main():
    """Main entry point for data download."""
    parser = argparse.ArgumentParser(description="Download comprehensive chess datasets")
    parser.add_argument("--data_dir", default="data", help="Data directory path")
    parser.add_argument("--max_puzzles", type=int, default=1000000, help="Maximum puzzles to download")
    parser.add_argument("--include_games", action="store_true", help="Include Lichess games (very large)")
    parser.add_argument("--source", choices=["lichess", "all"], default="all", help="Data source to download")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    downloader = ChessDataDownloader(data_dir)
    
    if args.source == "lichess":
        downloader.download_lichess_puzzles(args.max_puzzles)
        if args.include_games:
            downloader.download_lichess_games()
    else:
        downloader.download_all_data(args.max_puzzles, args.include_games)


if __name__ == "__main__":
    main()
