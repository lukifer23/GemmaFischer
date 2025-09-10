#!/usr/bin/env python3
"""Generate expanded chess Q&A dataset with diverse examples.

Creates 500+ high-quality chess questions covering:
- Opening principles
- Tactical motifs (forks, pins, skewers, discovered attacks)
- Strategic concepts (king safety, piece coordination, space)
- Endgame patterns (pawn promotion, opposition, zugzwang)
- Positional evaluation
- Move explanations
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class ChessDatasetGenerator:
    """Generate diverse chess Q&A examples."""

    def __init__(self):
        # Chess openings database
        self.openings = [
            ("Italian Game", "e2e4 e7e5 g1f3 b8c6 f1c4", "Develops pieces and controls center"),
            ("Sicilian Defense", "e2e4 c7c5", "Controls d4 square, creates asymmetry"),
            ("French Defense", "e2e4 e7e6", "Solid but passive, prepares d5 push"),
            ("Caro-Kann", "e2e4 c6c6", "Very solid, controls d5"),
            ("Queen's Gambit", "d2d4 d7d5 c2c4", "Fights for center control"),
            ("King's Indian", "d2d4 g8f6 c2c4 g7g6", "Hypermodern, allows center control"),
            ("English Opening", "c2c4", "Flexible, can transpose to many openings"),
            ("Reti Opening", "g1f3 d7d5 c2c4", "Hypermodern, develops knight first")
        ]

        # Tactical motifs
        self.tactics = [
            ("Fork", "A piece attacks two enemy pieces simultaneously"),
            ("Pin", "A piece cannot move because it would expose a more valuable piece"),
            ("Skewer", "A valuable piece is attacked, forcing it to move and exposing a less valuable piece"),
            ("Discovered Attack", "Moving one piece reveals an attack by another"),
            ("Double Attack", "Two pieces attack the same target"),
            ("Overloading", "A piece is doing too many defensive tasks"),
            ("Deflection", "Forcing a piece away from its defensive duties"),
            ("Decoy", "Luring a piece to a square where it can be captured")
        ]

        # Strategic concepts
        self.strategies = [
            ("King Safety", "Castle early, avoid unnecessary king moves"),
            ("Center Control", "Control e4, e5, d4, d5 squares"),
            ("Piece Development", "Develop knights before bishops, don't move same piece twice"),
            ("Pawn Structure", "Avoid doubled pawns, create pawn chains wisely"),
            ("Piece Coordination", "Pieces should work together, not individually"),
            ("Space Advantage", "Control more squares than opponent"),
            ("Initiative", "Keep the opponent on defensive, don't allow counterplay"),
            ("Material Balance", "Know relative piece values: P=1, N=B=3, R=5, Q=9")
        ]

        # Endgame principles
        self.endgames = [
            ("Pawn Promotion", "Advance pawns to 8th rank for promotion"),
            ("Opposition", "King in direct opposition prevents enemy king advance"),
            ("Zugzwang", "Position where any move worsens your situation"),
            ("King Activity", "Centralize king in endgame for attack/defense"),
            ("Passed Pawn", "Advance passed pawns, block with king if possible"),
            ("Piece vs Pawns", "Queen and rook usually win vs pawns, knights are tricky"),
            ("Fortress", "Create impregnable defensive position"),
            ("Triangulation", "Waste tempo to gain opposition")
        ]

    def generate_opening_questions(self, count: int = 50) -> List[Dict]:
        """Generate opening principle questions."""
        questions = []

        for _ in range(count):
            opening = random.choice(self.openings)
            name, moves, explanation = opening

            question_types = [
                f"What are the main ideas behind the {name}?",
                f"Why is the {name} considered a solid opening?",
                f"What are the typical pawn structures in the {name}?",
                f"How should White develop pieces in the {name}?",
                f"What are Black's typical responses to the {name}?"
            ]

            question = random.choice(question_types)
            answer = f"The {name} ({moves}) {explanation.lower()}. {self._expand_opening_explanation(name)}"

            questions.append({
                "question": question,
                "answer": answer,
                "category": "opening",
                "difficulty": random.choice(["beginner", "intermediate"])
            })

        return questions

    def _expand_opening_explanation(self, opening_name: str) -> str:
        """Add more detailed explanation for openings."""
        expansions = {
            "Italian Game": "White develops the kingside knight and bishop to active squares, controlling key central squares and preparing for kingside castling.",
            "Sicilian Defense": "This leads to complex, tactical positions where Black has more space on the queenside but White has a lead in development.",
            "French Defense": "Black creates a solid pawn chain but may experience difficulties developing the light-squared bishop.",
            "Queen's Gambit": "White offers a pawn to gain central control and faster development.",
            "King's Indian": "Black allows White to build a strong center, planning to attack it later with ...e5 or ...c5.",
            "English Opening": "This opening is very flexible and can lead to many different types of positions."
        }
        return expansions.get(opening_name, "This is a solid and principled opening choice.")

    def generate_tactical_questions(self, count: int = 80) -> List[Dict]:
        """Generate tactical motif questions."""
        questions = []

        for _ in range(count):
            tactic = random.choice(self.tactics)
            name, description = tactic

            question_types = [
                f"What is a {name.lower()} in chess?",
                f"How can you create a {name.lower()} in your games?",
                f"Why are {name.lower()}s dangerous for opponents?",
                f"What should you look for to avoid {name.lower()}s?",
                f"Can you give an example of a {name.lower()}?"
            ]

            question = random.choice(question_types)

            # Create detailed answer
            answer = f"A {name.lower()} is {description.lower()}. "
            if name == "Fork":
                answer += "For example, a knight on e5 attacking both a rook on a1 and a queen on h8 creates a fork. Knights are particularly good at forking due to their unique L-shaped movement."
            elif name == "Pin":
                answer += "For example, a bishop on g2 pinning a knight on f3 to the king on g1. The knight cannot move because it would expose the king to check."
            elif name == "Skewer":
                answer += "For example, a rook on the 7th rank attacking both the king and a queen behind it. The king must move, exposing the queen to capture."
            else:
                answer += f"Understanding {name.lower()}s is crucial for tactical awareness in chess."

            questions.append({
                "question": question,
                "answer": answer,
                "category": "tactics",
                "difficulty": random.choice(["beginner", "intermediate", "advanced"])
            })

        return questions

    def generate_strategic_questions(self, count: int = 60) -> List[Dict]:
        """Generate strategic concept questions."""
        questions = []

        for _ in range(count):
            strategy = random.choice(self.strategies)
            name, description = strategy

            question_types = [
                f"Why is {name.lower()} important in chess?",
                f"How can you achieve {name.lower()} in your games?",
                f"What are common mistakes related to {name.lower()}?",
                f"Can you give examples of good {name.lower()}?",
                f"How does {name.lower()} affect the game outcome?"
            ]

            question = random.choice(question_types)
            answer = f"{description}. "

            # Add specific advice based on strategy
            if name == "Center Control":
                answer += "The four central squares (e4, e5, d4, d5) are the most important in chess. Controlling them gives your pieces more mobility and restricts your opponent's options."
            elif name == "King Safety":
                answer += "Castling should usually happen before move 10. Avoid moving your king unnecessarily and keep it protected behind a pawn shield."
            elif name == "Piece Development":
                answer += "The general rule is: knights before bishops, don't move the same piece twice in the opening, and bring all pieces into the game."
            elif name == "Material Balance":
                answer += "Remember: Pawn = 1 point, Knight = 3, Bishop = 3, Rook = 5, Queen = 9. Trading equal value pieces is usually fine, but winning material creates a lasting advantage."

            questions.append({
                "question": question,
                "answer": answer,
                "category": "strategy",
                "difficulty": random.choice(["beginner", "intermediate"])
            })

        return questions

    def generate_endgame_questions(self, count: int = 40) -> List[Dict]:
        """Generate endgame principle questions."""
        questions = []

        for _ in range(count):
            endgame = random.choice(self.endgames)
            name, description = endgame

            question_types = [
                f"What is {name.lower()} in chess endgames?",
                f"Why is {name.lower()} important?",
                f"How can you use {name.lower()} to win?",
                f"What are common mistakes in {name.lower()} situations?",
                f"Can you give an example of {name.lower()}?"
            ]

            question = random.choice(question_types)
            answer = f"{description}. "

            # Add specific examples
            if name == "Opposition":
                answer += "For example, if kings are on the same file with one empty square between them, the player who does NOT have to move has the opposition and can prevent the enemy king from advancing."
            elif name == "Zugzwang":
                answer += "This often happens in endgames where all possible moves worsen your position. The opponent can then force you to make a bad move."
            elif name == "Pawn Promotion":
                answer += "Always advance pawns towards promotion, especially when the enemy king is far away. Use your own king to support pawn advances."
            elif name == "King Activity":
                answer += "In the endgame, the king becomes a strong piece. Centralize it to attack enemy pawns and support your own."

            questions.append({
                "question": question,
                "answer": answer,
                "category": "endgame",
                "difficulty": random.choice(["intermediate", "advanced"])
            })

        return questions

    def generate_positional_questions(self, count: int = 30) -> List[Dict]:
        """Generate positional evaluation questions."""
        questions = []

        positions = [
            ("Better development", "When you have more pieces developed and active"),
            ("Space advantage", "When you control more squares and have more room for your pieces"),
            ("Weak squares", "Squares that cannot be defended by pawns"),
            ("Outposts", "Strong squares for your pieces that cannot be attacked by enemy pawns"),
            ("Pawn breaks", "Pawn moves that open the position or create counterplay"),
            ("Piece activity", "When your pieces are on active squares with maximum mobility"),
            ("Structural weaknesses", "Pawn weaknesses like isolated or doubled pawns")
        ]

        for _ in range(count):
            position = random.choice(positions)
            concept, description = position

            question_types = [
                f"What does '{concept.lower()}' mean in chess?",
                f"Why is {concept.lower()} important?",
                f"How can you create {concept.lower()}?",
                f"What are the consequences of {concept.lower()}?",
                f"Can you give examples of {concept.lower()}?"
            ]

            question = random.choice(question_types)
            answer = f"{description}. This positional concept is crucial for understanding chess strategy and often determines the outcome of games."

            questions.append({
                "question": question,
                "answer": answer,
                "category": "positional",
                "difficulty": random.choice(["intermediate", "advanced"])
            })

        return questions

    def generate_dataset(self, total_examples: int = 500) -> List[Dict]:
        """Generate complete dataset with balanced categories."""
        # Calculate distribution
        opening_count = int(total_examples * 0.2)  # 20%
        tactical_count = int(total_examples * 0.32)  # 32%
        strategic_count = int(total_examples * 0.24)  # 24%
        endgame_count = int(total_examples * 0.16)  # 16%
        positional_count = int(total_examples * 0.08)  # 8%

        print(f"Generating dataset with {total_examples} examples:")
        print(f"  Openings: {opening_count}")
        print(f"  Tactics: {tactical_count}")
        print(f"  Strategy: {strategic_count}")
        print(f"  Endgames: {endgame_count}")
        print(f"  Positional: {positional_count}")

        # Generate all categories
        dataset = []
        dataset.extend(self.generate_opening_questions(opening_count))
        dataset.extend(self.generate_tactical_questions(tactical_count))
        dataset.extend(self.generate_strategic_questions(strategic_count))
        dataset.extend(self.generate_endgame_questions(endgame_count))
        dataset.extend(self.generate_positional_questions(positional_count))

        # Shuffle for randomness
        random.shuffle(dataset)

        return dataset

    def save_dataset(self, dataset: List[Dict], output_file: str):
        """Save dataset in JSONL format for training."""
        print(f"Saving {len(dataset)} examples to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                # Convert to training format
                training_example = {
                    "text": f"Question: {example['question']}\nAnswer: {example['answer']}"
                }
                json.dump(training_example, f, ensure_ascii=False)
                f.write('\n')

        # Save metadata
        metadata_file = output_file.replace('.jsonl', '_metadata.json')
        metadata = {
            "total_examples": len(dataset),
            "categories": {},
            "difficulty_distribution": {},
            "generation_date": str(datetime.now())
        }

        for example in dataset:
            category = example['category']
            difficulty = example['difficulty']

            metadata['categories'][category] = metadata['categories'].get(category, 0) + 1
            metadata['difficulty_distribution'][difficulty] = metadata['difficulty_distribution'].get(difficulty, 0) + 1

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {metadata_file}")
        print("Dataset generation complete!")


def main():
    """Generate expanded chess dataset."""
    generator = ChessDatasetGenerator()

    # Generate dataset
    dataset = generator.generate_dataset(500)

    # Save to file
    output_dir = Path(__file__).parent.parent / 'data' / 'finetune'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'chess_finetune_expanded.jsonl'

    generator.save_dataset(dataset, str(output_file))


if __name__ == '__main__':
    main()
