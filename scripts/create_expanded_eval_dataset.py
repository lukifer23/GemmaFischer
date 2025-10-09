#!/usr/bin/env python3
"""
Create an expanded, high-quality evaluation dataset for MoE router training.

Generates diverse chess positions and queries that clearly distinguish between
UCI (move generation), Tutor (analysis), and Director (strategic Q&A) experts.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Diverse chess positions covering different game phases and complexities
CHESS_POSITIONS = [
    # Opening positions
    {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "phase": "opening",
        "description": "Starting position"
    },
    {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "phase": "opening",
        "description": "After e4"
    },
    {
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "phase": "opening",
        "description": "Basic pawn structure"
    },

    # Middlegame tactical positions
    {
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
        "phase": "middlegame",
        "description": "Complex tactical position"
    },
    {
        "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/1BBP1P1q/3P4/Pp1P1PPp/RNBQ1RK1 w kq - 0 1",
        "phase": "middlegame",
        "description": "Chaotic middlegame with many pieces"
    },
    {
        "fen": "r4rk1/pp3ppp/2n1b3/q1pp2B1/8/P1Q2NP1/1PP1PP1P/2KR3R w - - 0 15",
        "phase": "middlegame",
        "description": "Queen and minor pieces middlegame"
    },

    # Endgame positions
    {
        "fen": "8/4R3/1p2P3/p4r2/P6p/1P3Pk1/4K3/8 w - - 1 64",
        "phase": "endgame",
        "description": "Rook endgame"
    },
    {
        "fen": "8/8/4k1p1/2KpP2p/5PP1/8/8/8 w - - 0 53",
        "phase": "endgame",
        "description": "King and pawn endgame"
    },
    {
        "fen": "6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - - 2 37",
        "phase": "endgame",
        "description": "Queen vs knight and pawns"
    },

    # Additional diverse positions
    {
        "fen": "5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27",
        "phase": "middlegame",
        "description": "Rook and queen middlegame"
    },
    {
        "fen": "r1bqk2r/pp1nbNp1/2p1p2p/8/2BP4/1PN3P1/P3QP1P/3R1RK1 b kq - 0 19",
        "phase": "middlegame",
        "description": "Unusual material balance"
    },
    {
        "fen": "4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35",
        "phase": "endgame",
        "description": "Double rook endgame"
    },
    {
        "fen": "r2qr1k1/b1p2ppp/pp4n1/P1P1p3/4P1n1/B2P2Pb/3NBP1P/RN1QR1K1 b - - 1 16",
        "phase": "middlegame",
        "description": "Heavy piece middlegame"
    }
]

def create_uci_queries() -> List[Dict[str, Any]]:
    """Create queries that clearly require UCI move generation."""
    queries = []

    for pos in CHESS_POSITIONS:
        # Pure move generation queries
        queries.append({
            "id": f"uci_move_{len(queries)+1}",
            "category": "pure_move",
            "question": f"FEN: {pos['fen']}\nWhat is the best move?",
            "expected_format": "uci_move_only",
            "expert": "uci",
            "phase": pos["phase"],
            "description": f"UCI move query for {pos['description']}"
        })

        # Tactical move queries
        if pos["phase"] in ["middlegame", "endgame"]:
            queries.append({
                "id": f"uci_tactical_{len(queries)+1}",
                "category": "tactical_move",
                "question": f"FEN: {pos['fen']}\nFind the strongest tactical move in this position.",
                "expected_format": "uci_move_only",
                "expert": "uci",
                "phase": pos["phase"],
                "description": f"Tactical UCI query for {pos['description']}"
            })

    # Add some standard opening moves
    standard_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", "Standard opening move"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "e7e5", "Symmetric response"),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "g1f3", "Knight development"),
        ("rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2", "b8c6", "Developing the knight"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4PP2/8/PPPP2PP/RNBQKBNR w KQkq - 1 3", "f1c4", "Italian Game bishop"),
    ]

    for fen, move, desc in standard_positions:
        queries.append({
            "id": f"uci_standard_{len(queries)+1}",
            "category": "standard_move",
            "question": f"FEN: {fen}\nWhat is the standard move to play here?",
            "expected_format": "uci_move_only",
            "expert": "uci",
            "phase": "opening",
            "description": desc
        })

    # Add complex tactical positions
    complex_tactical = [
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3B1N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Complex middlegame with tactics"),
        ("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 7", "Rich middlegame position"),
        ("r3r1k1/pppq1ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPPQ1PPP/R3R1K1 w - - 2 12", "Double queen middlegame"),
    ]

    for fen, desc in complex_tactical:
        queries.append({
            "id": f"uci_complex_{len(queries)+1}",
            "category": "complex_tactical",
            "question": f"FEN: {fen}\nWhat is the strongest move in this complex position?",
            "expected_format": "uci_move_only",
            "expert": "uci",
            "phase": "middlegame",
            "description": desc
        })

    return queries

def create_tutor_queries() -> List[Dict[str, Any]]:
    """Create queries that clearly require detailed analysis."""
    queries = []

    for pos in CHESS_POSITIONS:
        # Step-by-step analysis queries
        queries.append({
            "id": f"tutor_analysis_{len(queries)+1}",
            "category": "position_analysis",
            "question": f"FEN: {pos['fen']}\nAnalyze this position step by step. What are the key threats and opportunities?",
            "expected_format": "step_by_step_analysis",
            "expert": "tutor",
            "phase": pos["phase"],
            "description": f"Step-by-step analysis for {pos['description']}"
        })

        # Candidate moves evaluation
        queries.append({
            "id": f"tutor_candidates_{len(queries)+1}",
            "category": "candidate_evaluation",
            "question": f"FEN: {pos['fen']}\nWhat are the candidate moves to consider here? Evaluate their strengths and weaknesses.",
            "expected_format": "candidate_analysis",
            "expert": "tutor",
            "phase": pos["phase"],
            "description": f"Candidate moves analysis for {pos['description']}"
        })

    # Tactical analysis queries
    tactical_positions = [
        ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1", "Pin and tactics"),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/1BBP1P1q/3P4/Pp1P1PPp/RNBQ1RK1 w kq - 0 1", "Complex tactics"),
        ("5rk1/p5p1/3bpr1p/1Pp4q/3pR3/1P1Q1N2/P4PPP/4R1K1 w - - 4 22", "Rook and queen tactics")
    ]

    for fen, desc in tactical_positions:
        queries.append({
            "id": f"tutor_tactical_{len(queries)+1}",
            "category": "tactical_patterns",
            "question": f"FEN: {fen}\nIdentify and analyze the tactical patterns present in this position.",
            "expected_format": "tactical_analysis",
            "expert": "tutor",
            "phase": "middlegame",
            "description": f"Tactical pattern analysis for {desc}"
        })

    return queries

def create_director_queries() -> List[Dict[str, Any]]:
    """Create queries that clearly require strategic/conceptual understanding."""
    queries = []

    # Strategic concepts - expanded list
    strategic_questions = [
        "What are the main ideas behind the Sicilian Defense?",
        "Explain the concept of minority attack in chess.",
        "What is the purpose of fianchettoing a bishop?",
        "When should you consider a pawn break in the center?",
        "Explain the principle of two weaknesses.",
        "What are the key principles for rook and pawn endgames?",
        "How should you handle king and pawn endgames?",
        "What is the opposition in chess endgames?",
        "Explain the concept of zugzwang.",
        "When should you promote a pawn to a queen vs other pieces?",
        "How does castling work in chess?",
        "What is en passant and when can it be played?",
        "Explain the rules of pawn promotion.",
        "What are the conditions for a stalemate?",
        "How do you claim a draw by repetition?",
        "What is the concept of piece activity in chess?",
        "Explain the principle of centralization.",
        "What are the goals of the opening phase?",
        "How do you evaluate material imbalances?",
        "What is the concept of tempo in chess?",
        "Explain the idea of prophylactic thinking.",
        "What are the characteristics of a good pawn structure?",
        "How do you handle isolated pawns?",
        "What is backward pawn development?",
        "Explain the concept of a space advantage.",
        "What are the principles of king safety?",
        "How do you evaluate trading pieces?",
        "What is the concept of initiative in chess?",
        "Explain the principle of least resistance.",
        "What are the key ideas in the French Defense?"
    ]

    for i, question in enumerate(strategic_questions):
        queries.append({
            "id": f"director_strategy_{i+1}",
            "category": "strategic_explanation",
            "question": question,
            "expected_format": "strategic_explanation",
            "expert": "director",
            "phase": "general",
            "description": "Strategic chess concept explanation"
        })

    # Position-based strategic questions
    for pos in CHESS_POSITIONS:
        if pos["phase"] == "opening":
            queries.append({
                "id": f"director_opening_{len(queries)+1}",
                "category": "opening_strategy",
                "question": f"FEN: {pos['fen']}\nWhat are the strategic ideas and plans for both sides in this opening position?",
                "expected_format": "strategic_overview",
                "expert": "director",
                "phase": pos["phase"],
                "description": f"Opening strategy for {pos['description']}"
            })
        elif pos["phase"] == "endgame":
            queries.append({
                "id": f"director_endgame_{len(queries)+1}",
                "category": "endgame_principles",
                "question": f"FEN: {pos['fen']}\nWhat are the key endgame principles that apply to this position?",
                "expected_format": "endgame_analysis",
                "expert": "director",
                "phase": pos["phase"],
                "description": f"Endgame principles for {pos['description']}"
            })

    # Add specific middlegame strategic questions
    middlegame_strategy = [
        ("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 7", "Strategic planning in complex middlegame"),
        ("r3r1k1/pppq1ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPPQ1PPP/R3R1K1 w - - 2 12", "Long-term strategic decisions"),
        ("r1b2rk1/ppppqppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 4 8", "Strategic pawn structure decisions"),
    ]

    for fen, desc in middlegame_strategy:
        queries.append({
            "id": f"director_middlegame_{len(queries)+1}",
            "category": "middlegame_strategy",
            "question": f"FEN: {fen}\nWhat are the long-term strategic considerations in this middlegame position?",
            "expected_format": "strategic_analysis",
            "expert": "director",
            "phase": "middlegame",
            "description": desc
        })

    return queries

def balance_dataset(queries: List[Dict[str, Any]], target_per_expert: int = 50) -> List[Dict[str, Any]]:
    """Balance the dataset to have roughly equal representation from each expert."""
    expert_counts = {}
    balanced_queries = []

    # First pass: collect all queries by expert
    for query in queries:
        expert = query["expert"]
        if expert not in expert_counts:
            expert_counts[expert] = []
        expert_counts[expert].append(query)

    # Second pass: balance by taking up to target_per_expert from each expert
    for expert, expert_queries in expert_counts.items():
        # Shuffle to get random selection
        random.shuffle(expert_queries)

        # Take up to target_per_expert queries
        selected = expert_queries[:target_per_expert]
        balanced_queries.extend(selected)

        print(f"Expert {expert}: {len(expert_queries)} total, selected {len(selected)}")

    # Final shuffle to mix experts
    random.shuffle(balanced_queries)
    return balanced_queries

def create_expanded_evaluation_dataset():
    """Create an expanded, high-quality evaluation dataset."""
    print("🎯 Creating Expanded MoE Router Evaluation Dataset")
    print("=" * 60)

    # Generate queries for each expert
    print("\n📝 Generating queries...")
    uci_queries = create_uci_queries()
    tutor_queries = create_tutor_queries()
    director_queries = create_director_queries()

    print(f"UCI queries: {len(uci_queries)}")
    print(f"Tutor queries: {len(tutor_queries)}")
    print(f"Director queries: {len(director_queries)}")

    # Combine all queries
    all_queries = uci_queries + tutor_queries + director_queries

    # Balance the dataset
    print(f"\n⚖️  Balancing dataset (total: {len(all_queries)} queries)...")
    balanced_queries = balance_dataset(all_queries, target_per_expert=50)

    print(f"Balanced dataset: {len(balanced_queries)} queries")

    # Show final distribution
    final_counts = {}
    for query in balanced_queries:
        expert = query["expert"]
        final_counts[expert] = final_counts.get(expert, 0) + 1

    print("\n📊 Final expert distribution:")
    for expert, count in final_counts.items():
        print(f"   {expert}: {count}")

    # Save to file
    output_file = Path("data/validation/expanded_eval_suite.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving to {output_file}...")
    with open(output_file, 'w') as f:
        for query in balanced_queries:
            json.dump(query, f)
            f.write('\n')

    print("✅ Expanded evaluation dataset created!")

    # Show some samples
    print("\n📝 Sample queries from each expert:")
    samples_shown = {"uci": 0, "tutor": 0, "director": 0}

    for query in balanced_queries:
        expert = query["expert"]
        if samples_shown[expert] < 2:
            question_preview = query['question'][:60] + "..." if len(query['question']) > 60 else query['question']
            print(f"   [{expert.upper()}] {question_preview}")
            samples_shown[expert] += 1

        if all(count >= 2 for count in samples_shown.values()):
            break

    return str(output_file)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    output_file = create_expanded_evaluation_dataset()

    print(f"\n🎯 Next steps:")
    print(f"1. Use the expanded dataset: --eval_file {output_file}")
    print("2. Train MoE router: python scripts/train_moe_router.py --epochs 30 --eval_file " + output_file)
    print("3. The 30 epochs should be sufficient with this larger dataset")
