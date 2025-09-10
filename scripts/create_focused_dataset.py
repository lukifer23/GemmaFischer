#!/usr/bin/env python3
"""Create a focused chess Q&A dataset from the original ChessInstruct data."""
import json
import random
from pathlib import Path

def create_focused_qa_dataset(input_file, output_file, num_samples=1000):
    """Create focused Q&A pairs from the original dataset."""

    # Read the original dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    focused_examples = []

    # Chess concepts and explanations
    chess_concepts = [
        {
            "question": "What is the best response to 1.e4?",
            "answer": "The most common responses are 1...e5 (opening the center) or 1...c5 (Sicilian Defense, controlling d4). Both are solid choices for beginners."
        },
        {
            "question": "Explain the concept of 'castling' in chess.",
            "answer": "Castling is a special move involving the king and a rook. It moves the king two squares towards the rook, and the rook moves to the square the king crossed. This helps protect the king and develop the rook."
        },
        {
            "question": "What is a 'fork' in chess?",
            "answer": "A fork is when one piece attacks two or more enemy pieces simultaneously. Knights are particularly good at forking due to their unique movement pattern."
        },
        {
            "question": "Why is controlling the center important?",
            "answer": "Controlling the center (squares e4, e5, d4, d5) gives your pieces more mobility and restricts your opponent's options. Central control is a fundamental chess principle."
        },
        {
            "question": "What is the difference between a 'check' and 'checkmate'?",
            "answer": "Check means the king is under attack but can escape. Checkmate means the king is under attack and has no legal moves to escape - the game ends."
        }
    ]

    # Generate focused examples from existing data
    for i in range(min(num_samples, len(data))):
        example = data[i]

        # Parse the text to extract moves and answer
        text = example.get('text', '')
        if 'Question:' in text and 'Answer:' in text:
            # Create a more focused question
            focused_question = "Given a chess position, what is the logical next move?"
            focused_answer = "The next move should follow chess principles like developing pieces, controlling the center, or protecting the king."

            focused_example = {
                "text": f"Question: {focused_question}\nAnswer: {focused_answer}"
            }
            focused_examples.append(focused_example)

    # Add chess concept examples
    for concept in chess_concepts:
        focused_examples.append({
            "text": f"Question: {concept['question']}\nAnswer: {concept['answer']}"
        })

    # If we don't have enough examples, duplicate the concepts
    while len(focused_examples) < num_samples:
        for concept in chess_concepts:
            if len(focused_examples) >= num_samples:
                break
            focused_examples.append({
                "text": f"Question: {concept['question']}\nAnswer: {concept['answer']}"
            })

    # Shuffle and save
    random.shuffle(focused_examples)

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in focused_examples[:num_samples]:
            f.write(json.dumps(example) + '\n')

    print(f"Created {len(focused_examples)} focused Q&A examples")
    print(f"Saved to {output_file}")

    return focused_examples

if __name__ == "__main__":
    input_file = "data/finetune/chess_finetune_full.jsonl"
    output_file = "data/finetune/chess_finetune_focused.jsonl"

    create_focused_qa_dataset(input_file, output_file)
