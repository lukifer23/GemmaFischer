#!/usr/bin/env python3
"""
Chain-of-Thought (CoT) Chess Reasoning System

This module implements comprehensive CoT prompting for chess analysis including:
- Step-by-step reasoning templates
- Chess-specific reasoning patterns
- Dataset generation with reasoning traces
- Evaluation metrics for reasoning quality
"""

import json
import random
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import re


class ChessCoTGenerator:
    """Generate Chain-of-Thought examples for chess reasoning."""

    def __init__(self):
        # Define reasoning templates for different chess aspects
        self.reasoning_templates = {
            "opening": {
                "pawn_structure": "First, I need to examine the pawn structure. {pawn_analysis}. This creates {pawn_advantage}.",
                "piece_development": "Next, I should consider piece development. {development_analysis}. This means {development_conclusion}.",
                "center_control": "The center is crucial. {center_analysis}. This gives {center_control_advantage}.",
                "king_safety": "King safety is paramount. {king_safety_analysis}. Therefore, {castling_plan}.",
                "overall_evaluation": "Putting it all together: {overall_assessment}. The best move is {recommended_move} because {reasoning}."
            },

            "tactical": {
                "threat_recognition": "I need to identify potential threats. {threat_analysis}. This suggests {threat_response}.",
                "combination_analysis": "Let me look for tactical combinations. {combination_analysis}. The key sequence is {tactical_sequence}.",
                "material_evaluation": "Material balance is important. {material_analysis}. This gives {material_advantage}.",
                "defensive_resources": "Defensive possibilities include {defense_options}. The strongest defense is {best_defense}.",
                "conclusion": "Therefore, the tactical continuation is {recommended_line} because {tactical_reasoning}."
            },

            "positional": {
                "space_advantage": "Space control is key. {space_analysis}. This creates {space_benefit}.",
                "piece_activity": "Piece coordination matters. {activity_analysis}. This leads to {activity_conclusion}.",
                "weaknesses": "Structural weaknesses to consider: {weakness_analysis}. This can be exploited by {exploitation_plan}.",
                "compensation": "The position offers compensation through {compensation_factors}. This makes {compensation_assessment}.",
                "long_term": "Long-term considerations: {long_term_analysis}. The strategic plan should be {strategic_approach}."
            },

            "endgame": {
                "king_activity": "King activity is crucial in the endgame. {king_analysis}. This means {king_plan}.",
                "pawn_structure": "Pawn structure determines the outcome. {pawn_endgame_analysis}. This creates {pawn_endgame_outcome}.",
                "piece_coordination": "Piece coordination is essential. {coordination_analysis}. This leads to {coordination_strategy}.",
                "zugzwang": "Zugzwang possibilities: {zugzwang_analysis}. This affects {zugzwang_implications}.",
                "technique": "Endgame technique requires {technique_analysis}. The winning method is {winning_method}."
            }
        }

        # Chess concepts for generating diverse examples
        self.chess_concepts = {
            "openings": ["Italian Game", "Sicilian Defense", "French Defense", "Caro-Kann", "Queen's Gambit"],
            "tactics": ["fork", "pin", "skewer", "discovered attack", "double attack", "overloading"],
            "strategies": ["center control", "king safety", "piece development", "pawn structure", "space advantage"],
            "endgames": ["pawn endgame", "rook endgame", "queen endgame", "knight vs bishop", "opposite bishops"]
        }

    def generate_cot_example(self, category: str, difficulty: str = "intermediate") -> Dict[str, Any]:
        """Generate a single CoT example for chess reasoning."""

        if category == "opening":
            return self._generate_opening_cot()
        elif category == "tactical":
            return self._generate_tactical_cot()
        elif category == "positional":
            return self._generate_positional_cot()
        elif category == "endgame":
            return self._generate_endgame_cot()
        else:
            return self._generate_general_cot()

    def _generate_opening_cot(self) -> Dict[str, Any]:
        """Generate opening CoT example."""
        opening = random.choice(self.chess_concepts["openings"])

        # Step-by-step reasoning
        reasoning_steps = []

        # Step 1: Pawn structure analysis
        pawn_structures = ["solid pawn chain", "open center", "pawn majority on queenside", "symmetrical pawn structure"]
        pawn_structure = random.choice(pawn_structures)
        reasoning_steps.append(f"First, I examine the pawn structure. The {opening} typically creates a {pawn_structure}, which {'provides solid foundations' if 'solid' in pawn_structure else 'allows for active piece play'}.")

        # Step 2: Development analysis
        dev_patterns = ["rapid piece development", "kingside fianchetto", "central piece concentration", "flexible development"]
        dev_pattern = random.choice(dev_patterns)
        reasoning_steps.append(f"Next, I consider piece development. The {opening} encourages {dev_pattern}, which {'allows quick mobilization' if 'rapid' in dev_pattern else 'provides long-term positional advantages'}.")

        # Step 3: Center control
        center_controls = ["strong central presence", "flexible center", "pawn center occupation", "piece pressure on center"]
        center_control = random.choice(center_controls)
        reasoning_steps.append(f"The center is crucial in this opening. The {opening} achieves {center_control}, giving {'a spatial advantage' if 'strong' in center_control else 'flexibility for future maneuvers'}.")

        # Step 4: King safety
        safety_patterns = ["early castling", "pawn shield protection", "piece defense", "active king positioning"]
        safety_pattern = random.choice(safety_patterns)
        reasoning_steps.append(f"King safety considerations: The {opening} addresses this through {safety_pattern}, ensuring {'immediate security' if 'early' in safety_pattern else 'long-term protection'}.")

        # Step 5: Conclusion
        moves = ["e4", "d4", "Nf3", "Nc3", "Bc4", "Bb5"]
        best_move = random.choice(moves)
        reasoning_steps.append(f"Putting it all together: The {opening} is a solid choice that balances development, center control, and king safety. The best opening move for White is {best_move}, which initiates the {opening} and sets up the typical pawn structure.")

        question = f"What are the key principles and best first move for the {opening}?"

        return {
            "question": question,
            "reasoning_chain": reasoning_steps,
            "answer": reasoning_steps[-1],
            "category": "opening",
            "difficulty": "intermediate",
            "cot_format": True
        }

    def _generate_tactical_cot(self) -> Dict[str, Any]:
        """Generate tactical CoT example."""
        tactic = random.choice(self.chess_concepts["tactics"])

        reasoning_steps = []

        # Step 1: Recognize the tactical pattern
        pattern_recognition = {
            "fork": "I notice a piece attacking two enemy pieces simultaneously",
            "pin": "I see a piece that can't move because it would expose a more valuable piece",
            "skewer": "I identify a valuable piece attacked, forcing it to move and exposing another piece",
            "discovered attack": "I spot a piece moving to reveal an attack by another piece",
            "double attack": "I find two pieces attacking the same target",
            "overloading": "I see a piece doing too many defensive tasks"
        }

        reasoning_steps.append(f"First, I need to identify tactical opportunities. {pattern_recognition[tactic]}. This is a classic {tactic} pattern.")

        # Step 2: Analyze the consequences
        consequence_analysis = {
            "fork": "This creates immediate threats on two pieces, potentially winning material",
            "pin": "This restricts the opponent's mobility and creates positional pressure",
            "skewer": "This can force the opponent to lose a more valuable piece",
            "discovered attack": "This combines attack with development or another threat",
            "double attack": "This overloads the opponent's defensive resources",
            "overloading": "This creates multiple defensive weaknesses to exploit"
        }

        reasoning_steps.append(f"Next, I analyze the tactical consequences. {consequence_analysis[tactic]}. The key is to calculate the opponent's defensive options carefully.")

        # Step 3: Consider defensive responses
        reasoning_steps.append("Now I must consider the opponent's defensive possibilities. They might try to capture the attacking piece, interpose, or move the attacked piece. I need to evaluate if these defenses work.")

        # Step 4: Calculate the best continuation
        tactical_moves = ["Qxh7+", "Nxe5", "Bxf7+", "Rxd8+", "Qxg7+", "Nxf7"]
        best_tactical_move = random.choice(tactical_moves)
        reasoning_steps.append(f"After analyzing all possibilities, the strongest tactical continuation is {best_tactical_move}. This move exploits the {tactic} pattern and leads to a decisive advantage.")

        question = f"How should I analyze and execute a {tactic} in this chess position?"

        return {
            "question": question,
            "reasoning_chain": reasoning_steps,
            "answer": reasoning_steps[-1],
            "category": "tactical",
            "difficulty": "advanced",
            "cot_format": True
        }

    def _generate_positional_cot(self) -> Dict[str, Any]:
        """Generate positional CoT example."""
        strategy = random.choice(self.chess_concepts["strategies"])

        reasoning_steps = []

        # Step 1: Evaluate current position
        position_eval = {
            "center control": "I assess how well each side controls the central squares (e4, e5, d4, d5)",
            "king safety": "I examine the king's position and protection from potential attacks",
            "piece development": "I check how many pieces are developed and their activity level",
            "pawn structure": "I analyze pawn chains, weaknesses, and pawn breaks",
            "space advantage": "I compare territorial control and available squares"
        }

        reasoning_steps.append(f"First, I evaluate the current positional factors. {position_eval[strategy]}. This gives me a clear picture of the position's character.")

        # Step 2: Identify strengths and weaknesses
        reasoning_steps.append("Next, I identify each side's strengths and weaknesses. The better side typically has more active pieces, safer king, and fewer weaknesses.")

        # Step 3: Consider long-term plans
        reasoning_steps.append("Now I consider long-term strategic plans. This might involve improving piece placement, creating weaknesses, or preparing pawn breaks.")

        # Step 4: Choose optimal continuation
        positional_moves = ["Bf4", "Re1", "Nd2", "h3", "g4", "Bd3"]
        best_positional_move = random.choice(positional_moves)
        reasoning_steps.append(f"After careful analysis, the best positional move is {best_positional_move}. This improves {strategy} and strengthens the overall position.")

        question = f"What positional considerations are most important regarding {strategy}?"

        return {
            "question": question,
            "reasoning_chain": reasoning_steps,
            "answer": reasoning_steps[-1],
            "category": "positional",
            "difficulty": "intermediate",
            "cot_format": True
        }

    def _generate_endgame_cot(self) -> Dict[str, Any]:
        """Generate endgame CoT example."""
        endgame = random.choice(self.chess_concepts["endgames"])

        reasoning_steps = []

        # Step 1: Assess material and position
        reasoning_steps.append(f"First, I assess the endgame characteristics. In a {endgame}, the key factors are king activity, pawn structure, and piece coordination.")

        # Step 2: Identify winning/losing factors
        reasoning_steps.append("Next, I identify the winning and losing factors. These include passed pawns, king position, piece activity, and potential zugzwang positions.")

        # Step 3: Plan the technical execution
        reasoning_steps.append("Now I plan the technical execution. This involves precise calculation of moves, triangulation, and opposition when relevant.")

        # Step 4: Execute the winning method
        endgame_moves = ["Kd4", "h4", "Kf5", "g5", "Ke6", "f5"]
        best_endgame_move = random.choice(endgame_moves)
        reasoning_steps.append(f"After analyzing all factors, the correct endgame technique requires {best_endgame_move}. This move demonstrates proper {endgame} technique.")

        question = f"What are the key principles for handling a {endgame}?"

        return {
            "question": question,
            "reasoning_chain": reasoning_steps,
            "answer": reasoning_steps[-1],
            "category": "endgame",
            "difficulty": "advanced",
            "cot_format": True
        }

    def _generate_general_cot(self) -> Dict[str, Any]:
        """Generate general chess CoT example."""
        topics = ["checkmate patterns", "defensive technique", "calculation method", "evaluation principles"]
        topic = random.choice(topics)

        reasoning_steps = [
            f"First, I need to understand the fundamentals of {topic} in chess.",
            "Next, I consider the specific position and how general principles apply.",
            "Then, I analyze concrete variations and calculate accurately.",
            "Finally, I evaluate the resulting positions and choose the best continuation."
        ]

        question = f"How should I approach {topic} systematically?"

        return {
            "question": question,
            "reasoning_chain": reasoning_steps,
            "answer": reasoning_steps[-1],
            "category": "general",
            "difficulty": "intermediate",
            "cot_format": True
        }

    def generate_cot_dataset(self, num_examples: int = 500) -> List[Dict[str, Any]]:
        """Generate a dataset of CoT examples."""
        dataset = []
        categories = ["opening", "tactical", "positional", "endgame"]

        print(f"Generating {num_examples} CoT examples...")

        for i in range(num_examples):
            category = random.choice(categories)
            example = self.generate_cot_example(category)

            # Convert to training format
            training_format = {
                "text": f"Question: {example['question']}\nReasoning: {' '.join(example['reasoning_chain'])}\nAnswer: {example['answer']}",
                "question": example['question'],
                "reasoning": example['reasoning_chain'],
                "answer": example['answer'],
                "category": example['category'],
                "difficulty": example['difficulty'],
                "cot_format": True
            }

            dataset.append(training_format)

            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1} examples...")

        print(f"Generated {len(dataset)} CoT examples successfully!")
        return dataset

    def save_cot_dataset(self, dataset: List[Dict[str, Any]], output_file: str):
        """Save CoT dataset in JSONL format."""
        print(f"Saving {len(dataset)} CoT examples to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')

        # Save metadata
        metadata_file = output_file.replace('.jsonl', '_metadata.json')
        metadata = {
            "total_examples": len(dataset),
            "categories": {},
            "difficulty_distribution": {},
            "cot_format": True,
            "generation_date": str(datetime.now()),
            "description": "Chain-of-Thought chess reasoning dataset"
        }

        for example in dataset:
            category = example['category']
            difficulty = example['difficulty']

            metadata['categories'][category] = metadata['categories'].get(category, 0) + 1
            metadata['difficulty_distribution'][difficulty] = metadata['difficulty_distribution'].get(difficulty, 0) + 1

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {metadata_file}")
        print("CoT dataset generation complete!")

    def evaluate_cot_quality(self, response: str, question: str) -> Dict[str, Any]:
        """Evaluate the quality of CoT reasoning in a response."""
        evaluation = {
            "step_by_step_reasoning": False,
            "logical_progression": False,
            "chess_concepts_used": 0,
            "conclusion_supported": False,
            "overall_quality": "poor"
        }

        # Check for step-by-step reasoning
        step_indicators = ["first", "next", "then", "after", "finally", "therefore", "conclusion"]
        evaluation["step_by_step_reasoning"] = any(indicator in response.lower() for indicator in step_indicators)

        # Check for logical progression
        logical_connectors = ["because", "therefore", "so", "thus", "hence", "as a result"]
        evaluation["logical_progression"] = any(connector in response.lower() for connector in logical_connectors)

        # Count chess concepts
        chess_concepts = [
            "pawn", "knight", "bishop", "rook", "queen", "king", "castle", "checkmate",
            "center", "development", "safety", "structure", "advantage", "control"
        ]
        evaluation["chess_concepts_used"] = sum(1 for concept in chess_concepts if concept in response.lower())

        # Check if conclusion is supported by reasoning
        evaluation["conclusion_supported"] = len(response.split()) > 50 and evaluation["logical_progression"]

        # Overall quality assessment
        score = sum([
            evaluation["step_by_step_reasoning"],
            evaluation["logical_progression"],
            min(evaluation["chess_concepts_used"] / 5, 1),  # Normalize to 0-1
            evaluation["conclusion_supported"]
        ])

        if score >= 3.5:
            evaluation["overall_quality"] = "excellent"
        elif score >= 2.5:
            evaluation["overall_quality"] = "good"
        elif score >= 1.5:
            evaluation["overall_quality"] = "fair"
        else:
            evaluation["overall_quality"] = "poor"

        return evaluation


def main():
    """Generate CoT chess reasoning dataset."""
    generator = ChessCoTGenerator()

    # Generate CoT dataset
    dataset = generator.generate_cot_dataset(300)  # Start with 300 for testing

    # Save to file
    output_dir = Path(__file__).parent.parent / 'data' / 'finetune'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'chess_cot_reasoning.jsonl'

    generator.save_cot_dataset(dataset, str(output_file))

    # Test evaluation
    test_response = "First, I need to examine the pawn structure. The position has a solid pawn chain on the kingside. Next, I should consider piece development - the knight on b1 hasn't moved yet. Then, king safety is important - castling should happen soon. Therefore, the best move is Nf3 to develop the knight and prepare for kingside castling."
    evaluation = generator.evaluate_cot_quality(test_response, "What should White do in this position?")
    print("\nCoT Quality Evaluation Test:")
    print(f"Overall quality: {evaluation['overall_quality']}")
    print(f"Step-by-step reasoning: {evaluation['step_by_step_reasoning']}")
    print(f"Logical progression: {evaluation['logical_progression']}")
    print(f"Chess concepts used: {evaluation['chess_concepts_used']}")


if __name__ == "__main__":
    main()
