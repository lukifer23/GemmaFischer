#!/usr/bin/env python3
"""
Patch existing director data to focus on strategic concepts instead of tactics.

Converts puzzle-solving responses into strategic analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any
import re
import random

STRATEGIC_CONCEPTS = [
    "piece activity",
    "king safety",
    "pawn structure",
    "control of key squares",
    "development",
    "center control",
    "initiative",
    "material balance",
    "space advantage",
    "weaknesses exploitation"
]

OPENING_PRINCIPLES = [
    "Control the center with pawns and pieces",
    "Develop knights before bishops",
    "Castle early to ensure king safety",
    "Don't move the same piece twice in the opening",
    "Bring your queen out too early",
    "Create a strong pawn structure"
]

MIDDLLEGAME_CONCEPTS = [
    "Improve your worst piece",
    "Centralize your pieces",
    "Create and exploit weaknesses",
    "Maintain the initiative",
    "Coordinate your pieces for attack",
    "Defend actively, not passively"
]

ENDGAME_PRINCIPLES = [
    "Activate your king in the endgame",
    "Push passed pawns",
    "Cut off the enemy king from your pawns",
    "Use your king to support pawn advances",
    "Exchange pieces when ahead in material",
    "Keep your pieces active"
]

def patch_director_response(response: str, fen: str, rating: int) -> str:
    """
    Convert tactical response into strategic analysis.

    Example:
    Input: "Strategic briefing: Forcing sequence... Key continuation: play g5e7"
    Output: "This position illustrates the importance of piece coordination. White has achieved a strong initiative by coordinating their pieces effectively. The key strategic concept here is centralizing pieces to maximize their activity. Always consider how your pieces work together rather than individually."
    """

    # Determine game phase based on FEN
    piece_count = sum(1 for c in fen if c.isalpha() and c not in 'Kk')
    if piece_count > 20:
        phase = "opening"
        concepts = OPENING_PRINCIPLES + STRATEGIC_CONCEPTS[:5]
    elif piece_count > 10:
        phase = "middlegame"
        concepts = MIDDLLEGAME_CONCEPTS + STRATEGIC_CONCEPTS[3:8]
    else:
        phase = "endgame"
        concepts = ENDGAME_PRINCIPLES + STRATEGIC_CONCEPTS[5:]

    # Select appropriate strategic concept
    concept = random.choice(concepts)

    # Create strategic analysis
    if "opening" in phase:
        analysis = f"This {phase} position demonstrates key opening principles. {concept}. At this stage of the game, focus on rapid development, king safety, and central control. The position shows how proper opening play creates a solid foundation for the middlegame. Remember: strong openings lead to winning middlegames."

    elif "middlegame" in phase:
        analysis = f"This {phase} position illustrates important strategic concepts. {concept}. The middlegame is about converting your opening advantages into winning positions. Pay attention to piece coordination, weak squares, and pawn structure. Strong middlegame play requires both tactical awareness and strategic planning."

    else:  # endgame
        analysis = f"This {phase} position shows critical endgame principles. {concept}. Endgames are about precise calculation and king activity. Even with fewer pieces, small advantages can be decisive. Focus on pawn promotion, king centralization, and piece activity. Technical endgame play separates strong from weak players."

    return analysis

def main():
    """Patch director data to focus on strategy."""
    input_file = Path("data/standardized/standardized_director_expert_v2.jsonl")
    output_file = Path("data/standardized/enhanced_director_expert.jsonl")

    print("🎯 Patching director data for strategic focus...")

    patched_count = 0
    total_count = 0

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            total_count += 1
            try:
                data = json.loads(line.strip())

                # Only patch director data
                if data.get('task') == 'director_qa':
                    original_response = data['response']
                    fen = data.get('meta', {}).get('fen', '')
                    rating = data.get('meta', {}).get('rating', 1500)

                    # Patch the response to be strategic
                    strategic_response = patch_director_response(original_response, fen, rating)
                    data['response'] = strategic_response

                    # Update metadata
                    if 'meta' not in data:
                        data['meta'] = {}
                    data['meta']['strategic_enhanced'] = True
                    data['meta']['original_response'] = original_response

                    patched_count += 1

                # Write the (possibly patched) data
                f_out.write(json.dumps(data) + '\n')

            except json.JSONDecodeError as e:
                print(f"❌ Error parsing line {total_count}: {e}")
                continue

    print("✅ Director data patching complete!")
    print(f"   Total examples: {total_count}")
    print(f"   Patched examples: {patched_count}")
    print(f"   Output file: {output_file}")

if __name__ == "__main__":
    main()
