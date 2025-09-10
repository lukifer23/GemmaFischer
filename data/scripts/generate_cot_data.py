#!/usr/bin/env python3
"""
Chain-of-Thought Training Data Generator

Generates structured reasoning examples for chess analysis:
- Step-by-step position evaluation
- Tactical calculation sequences
- Strategic planning processes
- Endgame technique explanations

Creates training data that teaches the model to think through chess problems systematically.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import chess
import chess.engine
from datetime import datetime


class ChainOfThoughtGenerator:
    """Generate chain-of-thought training examples for chess reasoning."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.processed_dir = data_dir / "processed"
        self.datasets_dir = data_dir / "datasets"
        
        # Create directories
        for dir_path in [self.processed_dir, self.datasets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_tactical_cot(self, num_examples: int = 1000) -> List[Dict[str, Any]]:
        """Generate chain-of-thought examples for tactical analysis."""
        print(f"🎯 Generating {num_examples:,} tactical CoT examples...")
        
        examples = []
        
        # Tactical motifs and their reasoning patterns
        tactical_patterns = {
            'fork': {
                'description': 'Knight fork attacking king and queen',
                'reasoning_steps': [
                    'First, I need to identify the target pieces',
                    'The knight can attack both the king and queen simultaneously',
                    'This creates a fork that wins material',
                    'The opponent must move the king, losing the queen'
                ]
            },
            'pin': {
                'description': 'Bishop pinning a piece to the king',
                'reasoning_steps': [
                    'I can see a piece is blocking the king from moving',
                    'The bishop can pin this piece to the king',
                    'The pinned piece cannot move without exposing the king',
                    'This allows me to win the pinned piece'
                ]
            },
            'skewer': {
                'description': 'Rook skewering queen and king',
                'reasoning_steps': [
                    'The queen and king are aligned on the same rank',
                    'I can attack the queen with my rook',
                    'The queen must move, exposing the king',
                    'This wins material through the skewer'
                ]
            },
            'discovered_attack': {
                'description': 'Moving a piece to reveal an attack',
                'reasoning_steps': [
                    'I need to move a piece to clear a line',
                    'This will reveal an attack by another piece',
                    'The discovered attack will win material',
                    'I must calculate the sequence carefully'
                ]
            },
            'double_attack': {
                'description': 'Attacking two targets simultaneously',
                'reasoning_steps': [
                    'I can attack two different targets at once',
                    'The opponent cannot defend both threats',
                    'One of the targets will be lost',
                    'This creates a winning advantage'
                ]
            }
        }
        
        for i in range(num_examples):
            pattern_name = random.choice(list(tactical_patterns.keys()))
            pattern = tactical_patterns[pattern_name]
            
            # Generate a position (simplified - in practice, use real positions)
            fen = self._generate_tactical_position(pattern_name)
            
            # Create reasoning steps
            reasoning_steps = pattern['reasoning_steps'].copy()
            random.shuffle(reasoning_steps)  # Vary the order sometimes
            
            # Create the CoT example
            question = f"FEN: {fen}\nAnalyze this position step by step and find the best tactical move."
            
            # Build the answer with reasoning
            answer_parts = ["Let me analyze this position step by step:\n"]
            
            for j, step in enumerate(reasoning_steps, 1):
                answer_parts.append(f"{j}. {step}")
            
            answer_parts.append(f"\nBased on this analysis, the best move is [tactical_move].")
            answer_parts.append(f"This {pattern_name} wins material and gives me a significant advantage.")
            
            answer = "\n".join(answer_parts)
            
            example = {
                "text": f"Question: {question}\nAnswer: {answer}",
                "conversations": [
                    {"role": "system", "content": "You are a chess tactics tutor. Analyze positions step by step."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "category": "tactical_cot",
                "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                "reasoning_steps": reasoning_steps,
                "tactical_motif": pattern_name,
                "fen": fen,
                "source": "generated_cot"
            }
            
            examples.append(example)
        
        print(f"✅ Generated {len(examples):,} tactical CoT examples")
        return examples
    
    def generate_positional_cot(self, num_examples: int = 500) -> List[Dict[str, Any]]:
        """Generate chain-of-thought examples for positional analysis."""
        print(f"🎯 Generating {num_examples:,} positional CoT examples...")
        
        examples = []
        
        # Positional concepts and their analysis patterns
        positional_concepts = {
            'pawn_structure': {
                'description': 'Analyzing pawn structure weaknesses',
                'reasoning_steps': [
                    'First, I examine the pawn structure',
                    'I look for isolated, doubled, or backward pawns',
                    'These weaknesses can be exploited in the endgame',
                    'I should target these weaknesses with my pieces'
                ]
            },
            'piece_activity': {
                'description': 'Evaluating piece activity and coordination',
                'reasoning_steps': [
                    'I assess the activity of each piece',
                    'Active pieces control more squares and create threats',
                    'I look for ways to improve my piece placement',
                    'Better piece activity often leads to tactical opportunities'
                ]
            },
            'king_safety': {
                'description': 'Evaluating king safety and potential attacks',
                'reasoning_steps': [
                    'I check the safety of both kings',
                    'An exposed king is vulnerable to tactical shots',
                    'I look for ways to attack the opponent\'s king',
                    'I also ensure my own king is well-protected'
                ]
            },
            'space_advantage': {
                'description': 'Analyzing space control and territorial advantage',
                'reasoning_steps': [
                    'I evaluate who controls more space',
                    'Space advantage allows for better piece mobility',
                    'I look for ways to expand my territorial control',
                    'Space advantage can be converted into tactical opportunities'
                ]
            },
            'weak_squares': {
                'description': 'Identifying and exploiting weak squares',
                'reasoning_steps': [
                    'I look for squares that cannot be defended by pawns',
                    'These weak squares are ideal outposts for my pieces',
                    'I should place my pieces on these weak squares',
                    'Weak squares in the opponent\'s camp are particularly valuable'
                ]
            }
        }
        
        for i in range(num_examples):
            concept_name = random.choice(list(positional_concepts.keys()))
            concept = positional_concepts[concept_name]
            
            # Generate a position
            fen = self._generate_positional_position(concept_name)
            
            # Create reasoning steps
            reasoning_steps = concept['reasoning_steps'].copy()
            
            # Create the CoT example
            question = f"FEN: {fen}\nAnalyze this position from a positional perspective. What are the key factors to consider?"
            
            # Build the answer with reasoning
            answer_parts = ["Let me analyze this position step by step:\n"]
            
            for j, step in enumerate(reasoning_steps, 1):
                answer_parts.append(f"{j}. {step}")
            
            answer_parts.append(f"\nBased on this {concept_name} analysis, the key factors are:")
            answer_parts.append(f"- [specific_factor_1]")
            answer_parts.append(f"- [specific_factor_2]")
            answer_parts.append(f"- [specific_factor_3]")
            answer_parts.append(f"\nThe best plan is to [positional_plan].")
            
            answer = "\n".join(answer_parts)
            
            example = {
                "text": f"Question: {question}\nAnswer: {answer}",
                "conversations": [
                    {"role": "system", "content": "You are a chess positional analyst. Think through positions systematically."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "category": "positional_cot",
                "difficulty": random.choice(["intermediate", "advanced"]),
                "reasoning_steps": reasoning_steps,
                "positional_concept": concept_name,
                "fen": fen,
                "source": "generated_cot"
            }
            
            examples.append(example)
        
        print(f"✅ Generated {len(examples):,} positional CoT examples")
        return examples
    
    def generate_endgame_cot(self, num_examples: int = 300) -> List[Dict[str, Any]]:
        """Generate chain-of-thought examples for endgame analysis."""
        print(f"🎯 Generating {num_examples:,} endgame CoT examples...")
        
        examples = []
        
        # Endgame concepts and their analysis patterns
        endgame_concepts = {
            'king_and_pawn': {
                'description': 'King and pawn endgame technique',
                'reasoning_steps': [
                    'First, I count the material on both sides',
                    'I check if the pawn can be promoted',
                    'I calculate the opposition and key squares',
                    'I determine the correct technique to win or draw'
                ]
            },
            'rook_endgame': {
                'description': 'Rook endgame principles',
                'reasoning_steps': [
                    'I evaluate the activity of both rooks',
                    'I look for ways to cut off the opponent\'s king',
                    'I check if I can create a passed pawn',
                    'I apply the correct technique for this type of endgame'
                ]
            },
            'queen_endgame': {
                'description': 'Queen endgame technique',
                'reasoning_steps': [
                    'I assess the relative king positions',
                    'I look for ways to check the opponent\'s king',
                    'I calculate if I can win the opponent\'s pawn',
                    'I must be careful not to allow stalemate'
                ]
            },
            'opposition': {
                'description': 'Understanding and using opposition',
                'reasoning_steps': [
                    'I identify if this is a direct or distant opposition',
                    'I calculate who has the opposition',
                    'I determine the correct moves to maintain or gain opposition',
                    'I use the opposition to achieve my goal'
                ]
            }
        }
        
        for i in range(num_examples):
            concept_name = random.choice(list(endgame_concepts.keys()))
            concept = endgame_concepts[concept_name]
            
            # Generate an endgame position
            fen = self._generate_endgame_position(concept_name)
            
            # Create reasoning steps
            reasoning_steps = concept['reasoning_steps'].copy()
            
            # Create the CoT example
            question = f"FEN: {fen}\nAnalyze this endgame position. What is the correct technique?"
            
            # Build the answer with reasoning
            answer_parts = ["Let me analyze this endgame step by step:\n"]
            
            for j, step in enumerate(reasoning_steps, 1):
                answer_parts.append(f"{j}. {step}")
            
            answer_parts.append(f"\nBased on this {concept_name} analysis, the correct technique is:")
            answer_parts.append(f"- [technique_step_1]")
            answer_parts.append(f"- [technique_step_2]")
            answer_parts.append(f"- [technique_step_3]")
            answer_parts.append(f"\nThis leads to [result].")
            
            answer = "\n".join(answer_parts)
            
            example = {
                "text": f"Question: {question}\nAnswer: {answer}",
                "conversations": [
                    {"role": "system", "content": "You are a chess endgame expert. Analyze endgames systematically."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "category": "endgame_cot",
                "difficulty": random.choice(["intermediate", "advanced"]),
                "reasoning_steps": reasoning_steps,
                "endgame_concept": concept_name,
                "fen": fen,
                "source": "generated_cot"
            }
            
            examples.append(example)
        
        print(f"✅ Generated {len(examples):,} endgame CoT examples")
        return examples
    
    def generate_opening_cot(self, num_examples: int = 400) -> List[Dict[str, Any]]:
        """Generate chain-of-thought examples for opening analysis."""
        print(f"🎯 Generating {num_examples:,} opening CoT examples...")
        
        examples = []
        
        # Opening concepts and their analysis patterns
        opening_concepts = {
            'development': {
                'description': 'Piece development principles',
                'reasoning_steps': [
                    'I prioritize developing my pieces quickly',
                    'I avoid moving the same piece twice in the opening',
                    'I develop knights before bishops when possible',
                    'I aim to control the center with my pieces'
                ]
            },
            'center_control': {
                'description': 'Controlling the center squares',
                'reasoning_steps': [
                    'I evaluate who controls the central squares',
                    'Central control gives my pieces more mobility',
                    'I look for ways to challenge the opponent\'s center',
                    'I aim to establish a strong central presence'
                ]
            },
            'king_safety': {
                'description': 'Ensuring king safety in the opening',
                'reasoning_steps': [
                    'I need to castle to get my king to safety',
                    'I avoid moving pawns in front of my castled king',
                    'I look for ways to attack the opponent\'s uncastled king',
                    'King safety is crucial before launching attacks'
                ]
            },
            'pawn_structure': {
                'description': 'Building a good pawn structure',
                'reasoning_steps': [
                    'I avoid creating pawn weaknesses early',
                    'I look for ways to create pawn majorities',
                    'I consider the long-term pawn structure',
                    'Good pawn structure supports piece activity'
                ]
            }
        }
        
        for i in range(num_examples):
            concept_name = random.choice(list(opening_concepts.keys()))
            concept = opening_concepts[concept_name]
            
            # Generate an opening position
            fen = self._generate_opening_position(concept_name)
            
            # Create reasoning steps
            reasoning_steps = concept['reasoning_steps'].copy()
            
            # Create the CoT example
            question = f"FEN: {fen}\nWhat are the key principles to follow in this opening position?"
            
            # Build the answer with reasoning
            answer_parts = ["Let me analyze this opening position step by step:\n"]
            
            for j, step in enumerate(reasoning_steps, 1):
                answer_parts.append(f"{j}. {step}")
            
            answer_parts.append(f"\nBased on this {concept_name} analysis, the key principles are:")
            answer_parts.append(f"- [principle_1]")
            answer_parts.append(f"- [principle_2]")
            answer_parts.append(f"- [principle_3]")
            answer_parts.append(f"\nThe best move is [opening_move].")
            
            answer = "\n".join(answer_parts)
            
            example = {
                "text": f"Question: {question}\nAnswer: {answer}",
                "conversations": [
                    {"role": "system", "content": "You are a chess opening expert. Think through opening principles systematically."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "category": "opening_cot",
                "difficulty": random.choice(["beginner", "intermediate"]),
                "reasoning_steps": reasoning_steps,
                "opening_concept": concept_name,
                "fen": fen,
                "source": "generated_cot"
            }
            
            examples.append(example)
        
        print(f"✅ Generated {len(examples):,} opening CoT examples")
        return examples
    
    def _generate_tactical_position(self, motif: str) -> str:
        """Generate a tactical position (simplified - use real positions in practice)."""
        # In practice, use real tactical positions from databases
        tactical_positions = {
            'fork': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1',
            'pin': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1',
            'skewer': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1',
            'discovered_attack': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1',
            'double_attack': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1'
        }
        return tactical_positions.get(motif, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    
    def _generate_positional_position(self, concept: str) -> str:
        """Generate a positional position (simplified - use real positions in practice)."""
        # In practice, use real positional positions from databases
        positional_positions = {
            'pawn_structure': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1',
            'piece_activity': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1',
            'king_safety': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1',
            'space_advantage': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1',
            'weak_squares': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1'
        }
        return positional_positions.get(concept, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    
    def _generate_endgame_position(self, concept: str) -> str:
        """Generate an endgame position (simplified - use real positions in practice)."""
        # In practice, use real endgame positions from databases
        endgame_positions = {
            'king_and_pawn': '8/8/8/8/8/8/4P3/4K3 w - - 0 1',
            'rook_endgame': '8/8/8/8/8/8/4P3/4K2R w - - 0 1',
            'queen_endgame': '8/8/8/8/8/8/4P3/4K3 w - - 0 1',
            'opposition': '8/8/8/8/8/8/4K3/4k3 w - - 0 1'
        }
        return endgame_positions.get(concept, '8/8/8/8/8/8/4K3/4k3 w - - 0 1')
    
    def _generate_opening_position(self, concept: str) -> str:
        """Generate an opening position (simplified - use real positions in practice)."""
        # In practice, use real opening positions from databases
        opening_positions = {
            'development': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1',
            'center_control': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1',
            'king_safety': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1',
            'pawn_structure': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1'
        }
        return opening_positions.get(concept, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    
    def save_cot_dataset(self, examples: List[Dict[str, Any]], output_file: Path) -> Dict[str, Any]:
        """Save chain-of-thought examples to file."""
        print(f"💾 Saving {len(examples):,} CoT examples to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Create metadata
        metadata = {
            'total_examples': len(examples),
            'categories': {},
            'difficulty_distribution': {},
            'creation_date': datetime.now().isoformat(),
            'description': 'Chain-of-thought reasoning examples for chess analysis'
        }
        
        for example in examples:
            category = example['category']
            difficulty = example['difficulty']
            
            metadata['categories'][category] = metadata['categories'].get(category, 0) + 1
            metadata['difficulty_distribution'][difficulty] = metadata['difficulty_distribution'].get(difficulty, 0) + 1
        
        metadata_file = output_file.with_suffix('.metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ CoT dataset saved!")
        print(f"   📁 File: {output_file}")
        print(f"   📊 Metadata: {metadata_file}")
        
        return metadata


def main():
    """Main entry point for CoT data generation."""
    parser = argparse.ArgumentParser(description="Generate chain-of-thought training data")
    parser.add_argument("--data_dir", default="data", help="Data directory path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--type", choices=["tactical", "positional", "endgame", "opening", "all"], 
                       default="all", help="Type of CoT examples to generate")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to generate")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    generator = ChainOfThoughtGenerator(data_dir)
    
    all_examples = []
    
    if args.type in ["tactical", "all"]:
        tactical_examples = generator.generate_tactical_cot(args.num_examples)
        all_examples.extend(tactical_examples)
    
    if args.type in ["positional", "all"]:
        positional_examples = generator.generate_positional_cot(args.num_examples // 2)
        all_examples.extend(positional_examples)
    
    if args.type in ["endgame", "all"]:
        endgame_examples = generator.generate_endgame_cot(args.num_examples // 3)
        all_examples.extend(endgame_examples)
    
    if args.type in ["opening", "all"]:
        opening_examples = generator.generate_opening_cot(args.num_examples // 2)
        all_examples.extend(opening_examples)
    
    # Save combined dataset
    output_file = Path(args.output) if args.output else data_dir / "datasets" / "cot_reasoning_examples.jsonl"
    metadata = generator.save_cot_dataset(all_examples, output_file)
    
    print(f"\n🎉 CoT data generation complete!")
    print(f"   📊 Total examples: {len(all_examples):,}")
    print(f"   📁 Output: {output_file}")


if __name__ == "__main__":
    main()
