from __future__ import annotations

from typing import Optional

from src.inference.inference import ChessGemmaInference


class DirectorExpert:
    """Expert specialized in rules, theory, strategy, and general Q&A (no FEN required)."""

    def __init__(self, inference: ChessGemmaInference):
        self.inference = inference

    def answer(self, question: str, temperature: float = 0.6, top_p: float = 0.9, max_new_tokens: int = 256) -> str:
        if not self.inference.load_model():
            return "Model unavailable."
        prompt = (
            "Chess Director: You are a concise and accurate teacher. Answer clearly and avoid fabrications.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        try:
            return self.inference.generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.05,
            )
        except Exception:
            return "An error occurred while generating the answer."


