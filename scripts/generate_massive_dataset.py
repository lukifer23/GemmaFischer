#!/usr/bin/env python3
"""
Generate a massive chess dataset by running multiple iterations.

This script creates a large, diverse dataset by repeatedly calling
the existing generation functions with different parameters.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class MassiveDatasetGenerator:
    """Generate a large dataset by running multiple iterations."""

    def __init__(self):
        self.questions_seen = set()
        self.dataset = []

    def generate_opening_examples(self, count: int) -> List[Dict]:
        """Generate opening-related examples."""
        examples = []
        openings = [
            ('Italian Game', 'e2e4 e7e5 g1f3 b8c6 f1c4'),
            ('Sicilian Defense', 'e2e4 c7c5'),
            ('French Defense', 'e2e4 e6e6'),
            ('Caro-Kann', 'e2e4 c6c6'),
            ('Queen\'s Gambit', 'd2d4 d7d5 c2c4'),
            ('King\'s Indian', 'd2d4 g8f6 c2c4 g7g6'),
            ('English Opening', 'c2c4'),
            ('Reti Opening', 'g1f3 d7d5 c2c4'),
            ('Scotch Game', 'e2e4 e7e5 g1f3 b8c6 d2d4'),
            ('Ruy Lopez', 'e2e4 e7e5 g1f3 b8c6 f1b5'),
            ('Queen\'s Indian', 'd2d4 g8f6 c2c4 e7e6 g1f3 b7b6'),
            ('Nimzo-Indian', 'd2d4 g8f6 c2c4 e7e6 b1c3 f8b4')
        ]

        question_templates = [
            "What are the main ideas behind the {}?",
            "Why is the {} considered a solid opening?",
            "What are the typical pawn structures in the {}?",
            "How should I develop my pieces in the {}?",
            "What are Black's typical responses to the {}?",
            "When should I play the {}?",
            "What are the strategic goals in the {}?",
            "How does the {} control the center?",
            "What piece development should I prioritize in the {}?",
            "What are the key tactical ideas in the {}?",
            "How does the {} handle king safety?",
            "What are the most common mistakes in the {}?",
            "How should I play against the {}?",
            "What are the main lines in the {}?",
            "How does the {} transition to the middlegame?"
        ]

        for _ in range(count):
            opening_name, moves = random.choice(openings)
            template = random.choice(question_templates)
            question = template.format(opening_name)

            if question not in self.questions_seen:
                self.questions_seen.add(question)

                emphasis = random.choice(['rapid development', 'central control', 'positional understanding', 'tactical opportunities', 'king safety', 'piece activity'])
                leads_to = random.choice(['complex positions', 'sharp battles', 'strategic struggles', 'tactical complications'])
                suitable_for = random.choice(['beginners', 'intermediate players', 'advanced players', 'competitive play'])
                answer = f"The {opening_name} ({moves}) is a solid opening that emphasizes {emphasis}. It leads to {leads_to} and is suitable for {suitable_for}."

                examples.append({
                    'question': question,
                    'answer': answer,
                    'source': 'massive_openings',
                    'category': 'opening',
                    'difficulty': random.choice(['beginner', 'intermediate'])
                })

        return examples

    def generate_tactical_examples(self, count: int) -> List[Dict]:
        """Generate tactical examples."""
        examples = []
        tactics = [
            ('fork', 'attacking two pieces simultaneously'),
            ('pin', 'trapping a piece behind another'),
            ('skewer', 'attacking a valuable piece forcing it to move'),
            ('discovered attack', 'moving one piece reveals an attack'),
            ('double attack', 'attacking two different targets'),
            ('overloading', 'a piece defending too many squares'),
            ('deflection', 'forcing a piece away from defense'),
            ('decoy', 'luring a piece to a bad square'),
            ('zwischenzug', 'an intermediate move'),
            ('sacrifice', 'giving up material for advantage')
        ]

        question_templates = [
            "What is a {} in chess?",
            "How can I create a {} in my games?",
            "Why are {}s dangerous for opponents?",
            "What should I look for to avoid {}s?",
            "Can you give an example of a {}?",
            "When should I look for {} opportunities?",
            "How do I calculate {} combinations?",
            "What pieces are best for creating {}s?",
            "How do I defend against a {}?",
            "What are the consequences of missing a {}?"
        ]

        for _ in range(count):
            tactic_name, description = random.choice(tactics)
            template = random.choice(question_templates)
            question = template.format(tactic_name + "s" if template.endswith("s?") else tactic_name)

            if question not in self.questions_seen:
                self.questions_seen.add(question)

                answer = f"A {tactic_name} is when {description}. {random.choice(['Knights are particularly good at forking', 'Bishops excel at pins', 'Rooks are strong on open files', 'Queens combine power with flexibility', 'Pawns can create powerful levers'])}. Always be alert for tactical opportunities and calculate carefully!"

                examples.append({
                    'question': question,
                    'answer': answer,
                    'source': 'massive_tactics',
                    'category': 'tactics',
                    'difficulty': random.choice(['beginner', 'intermediate'])
                })

        return examples

    def generate_strategic_examples(self, count: int) -> List[Dict]:
        """Generate strategic examples."""
        examples = []
        strategies = [
            ('center control', 'occupying and influencing central squares'),
            ('king safety', 'protecting the king from attacks'),
            ('piece development', 'bringing pieces into active positions'),
            ('pawn structure', 'arranging pawns optimally'),
            ('initiative', 'keeping the opponent on defensive'),
            ('material balance', 'understanding piece values'),
            ('weak squares', 'exploiting undefended squares'),
            ('outposts', 'strong squares for pieces'),
            ('space advantage', 'controlling more territory'),
            ('piece coordination', 'making pieces work together')
        ]

        question_templates = [
            "Why is {} important in chess?",
            "How can I achieve {} in my games?",
            "What are common mistakes related to {}?",
            "Can you give examples of good {}?",
            "How does {} affect the game outcome?",
            "When should I prioritize {}?",
            "What are the principles of {}?",
            "How do I improve my {}?",
            "What are the consequences of poor {}?",
            "How does {} relate to winning chess?"
        ]

        for _ in range(count):
            strategy_name, description = random.choice(strategies)
            template = random.choice(question_templates)
            question = template.format(strategy_name)

            if question not in self.questions_seen:
                self.questions_seen.add(question)

                answer = f"{description.capitalize()} is crucial for success in chess. Good {strategy_name} gives you {random.choice(['control of the game', 'safety', 'activity', 'structure', 'advantage', 'winning chances'])}. Master these principles to improve your chess strength."

                examples.append({
                    'question': question,
                    'answer': answer,
                    'source': 'massive_strategy',
                    'category': 'strategy',
                    'difficulty': random.choice(['intermediate', 'advanced'])
                })

        return examples

    def generate_endgame_examples(self, count: int) -> List[Dict]:
        """Generate endgame examples."""
        examples = []
        endgames = [
            ('pawn endgame', 'kings and pawns only'),
            ('rook endgame', 'featuring rooks'),
            ('queen endgame', 'with queens on board'),
            ('knight vs bishop', 'minor piece endings'),
            ('opposite bishops', 'bishops on opposite colors'),
            ('same color bishops', 'bishops on same color'),
            ('king and pawn vs king', 'basic pawn promotion'),
            ('rook vs pawns', 'defensive technique'),
            ('queen vs rook', 'material advantage'),
            ('two rooks vs queen', 'coordination advantage')
        ]

        question_templates = [
            "What are the key principles of {}s?",
            "How do I win a {}?",
            "What should I avoid in {}s?",
            "When should I trade pieces in {}s?",
            "How do I convert an advantage in {}s?",
            "What are the most important techniques in {}s?",
            "How do I defend in {}s?",
            "What are common mistakes in {}s?",
            "How does king activity matter in {}s?",
            "What are the winning methods in {}s?"
        ]

        for _ in range(count):
            endgame_name, description = random.choice(endgames)
            template = random.choice(question_templates)
            question = template.format(endgame_name)

            if question not in self.questions_seen:
                self.questions_seen.add(question)

                answer = f"In {endgame_name}s ({description}), the key principles are {random.choice(['king activity', 'pawn promotion', 'piece coordination', 'precise calculation', 'opposition', 'zugzwang'])}. {'King centralization is crucial' if 'pawn' in endgame_name else 'Rooks belong on open files' if 'rook' in endgame_name else 'Queens are powerful but can be vulnerable' if 'queen' in endgame_name else 'Piece coordination determines the outcome'}. Study these endings carefully!"

                examples.append({
                    'question': question,
                    'answer': answer,
                    'source': 'massive_endgames',
                    'category': 'endgame',
                    'difficulty': random.choice(['intermediate', 'advanced'])
                })

        return examples

    def generate_general_examples(self, count: int) -> List[Dict]:
        """Generate general chess knowledge examples."""
        examples = []
        topics = [
            'checkmate patterns',
            'defensive technique',
            'calculation method',
            'evaluation principles',
            'time management',
            'psychological factors',
            'opening repertoire',
            'study methods',
            'competition preparation',
            'rating improvement'
        ]

        question_templates = [
            "How should I approach {} systematically?",
            "What are the best practices for {}?",
            "How can I improve my {} skills?",
            "What are common mistakes in {}?",
            "How does {} affect chess performance?",
            "What are the principles of {}?",
            "How do I develop {} effectively?",
            "What are the most important aspects of {}?",
            "How should I practice {}?",
            "What are the consequences of poor {}?"
        ]

        for _ in range(count):
            topic = random.choice(topics)
            template = random.choice(question_templates)
            question = template.format(topic)

            if question not in self.questions_seen:
                self.questions_seen.add(question)

                answer = f"{topic.replace('_', ' ').title()} is an essential aspect of chess mastery. Systematic approach involves {random.choice(['practice', 'study', 'analysis', 'reflection', 'experimentation'])} and {random.choice(['patience', 'consistency', 'focus', 'discipline', 'creativity'])}. Success comes from understanding principles and applying them consistently."

                examples.append({
                    'question': question,
                    'answer': answer,
                    'source': 'massive_general',
                    'category': 'general',
                    'difficulty': random.choice(['beginner', 'intermediate', 'advanced'])
                })

        return examples

    def generate_massive_dataset(self, target_size: int = 2000) -> List[Dict]:
        """Generate a massive dataset by running multiple iterations."""
        print(f"🎯 Generating {target_size} massive chess examples...")

        # Distribution across categories
        category_counts = {
            'openings': int(target_size * 0.25),    # 25%
            'tactics': int(target_size * 0.30),     # 30%
            'strategy': int(target_size * 0.20),    # 20%
            'endgames': int(target_size * 0.15),    # 15%
            'general': int(target_size * 0.10)      # 10%
        }

        generators = {
            'openings': self.generate_opening_examples,
            'tactics': self.generate_tactical_examples,
            'strategy': self.generate_strategic_examples,
            'endgames': self.generate_endgame_examples,
            'general': self.generate_general_examples
        }

        total_generated = 0
        batch_size = 100  # Generate in batches

        for category, target_count in category_counts.items():
            print(f"\n🔄 Generating {target_count} examples for {category}...")
            generated_for_category = 0

            while generated_for_category < target_count and total_generated < target_size:
                # Generate a batch
                batch_examples = generators[category](batch_size)

                # Add unique examples
                for example in batch_examples:
                    if generated_for_category >= target_count or total_generated >= target_size:
                        break

                    if example['question'] not in self.questions_seen:
                        self.questions_seen.add(example['question'])
                        self.dataset.append(example)
                        generated_for_category += 1
                        total_generated += 1

                print(f"   ✓ Batch completed: {generated_for_category}/{target_count} for {category}")

        # Shuffle the dataset
        random.shuffle(self.dataset)

        print(f"\n✅ Massive dataset generation complete!")
        print(f"   Total examples: {len(self.dataset)}")
        print(f"   Unique questions: {len(self.questions_seen)}")

        return self.dataset

    def save_dataset(self, output_file: str):
        """Save the massive dataset."""
        print(f"💾 Saving {len(self.dataset)} examples to {output_file}")

        # Convert to training format
        training_data = []
        for example in self.dataset:
            training_example = {
                "text": f"Question: {example['question']}\nAnswer: {example['answer']}",
                "question": example['question'],
                "answer": example['answer'],
                "source": example['source'],
                "category": example['category'],
                "difficulty": example['difficulty']
            }
            training_data.append(training_example)

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in training_data:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')

        # Save metadata
        metadata_file = output_file.replace('.jsonl', '_metadata.json')
        metadata = {
            "total_examples": len(self.dataset),
            "categories": {},
            "sources": {},
            "difficulty_distribution": {},
            "generation_date": str(datetime.now()),
            "description": "Massive multi-category chess Q&A dataset"
        }

        for example in self.dataset:
            # Count by category
            category = example['category']
            metadata['categories'][category] = metadata['categories'].get(category, 0) + 1

            # Count by source
            source = example['source']
            metadata['sources'][source] = metadata['sources'].get(source, 0) + 1

            # Count by difficulty
            difficulty = example['difficulty']
            metadata['difficulty_distribution'][difficulty] = metadata['difficulty_distribution'].get(difficulty, 0) + 1

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"📊 Metadata saved to {metadata_file}")
        print("\n🎉 Massive dataset saved successfully!")
        print(f"   File: {output_file}")
        print(f"   Size: {len(self.dataset)} examples")


def main():
    """Generate massive chess dataset."""
    generator = MassiveDatasetGenerator()

    # Generate massive dataset
    dataset = generator.generate_massive_dataset(2000)

    # Save to file
    output_dir = Path(__file__).parent.parent / 'data' / 'finetune'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'chess_massive_2000.jsonl'

    generator.save_dataset(str(output_file))


if __name__ == "__main__":
    main()
