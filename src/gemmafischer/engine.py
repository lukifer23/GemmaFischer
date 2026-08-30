from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import chess
import chess.engine

from .domain import (
    WDL,
    BoardFact,
    CandidateEvidence,
    EngineEvidence,
    EngineMetadata,
    canonical_hash,
    normalize_fen,
)

NODE_BUDGET = 250_000
ENGINE_OPTIONS: dict[str, int | bool] = {
    "Threads": 1,
    "Hash": 256,
    "UCI_ShowWDL": True,
}


class EngineUnavailable(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_stockfish(explicit: str | None = None) -> Path:
    candidates = [explicit, os.environ.get("GEMMAFISCHER_STOCKFISH"), shutil.which("stockfish")]
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if path.is_file() and os.access(path, os.X_OK):
                return path
    raise EngineUnavailable(
        "Stockfish was not found. Install Stockfish 18 and set GEMMAFISCHER_STOCKFISH."
    )


def _fact(fact_type: str, value: bool | int | str) -> BoardFact:
    payload = {"fact_type": fact_type, "value": value, "source": "python-chess"}
    return BoardFact(evidence_id=canonical_hash(payload), fact_type=fact_type, value=value)  # type: ignore[arg-type]


def extract_board_facts(board: chess.Board) -> tuple[BoardFact, ...]:
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
    }
    material = sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in values.items()
    )
    castling = "".join(
        symbol
        for allowed, symbol in (
            (board.has_kingside_castling_rights(chess.WHITE), "K"),
            (board.has_queenside_castling_rights(chess.WHITE), "Q"),
            (board.has_kingside_castling_rights(chess.BLACK), "k"),
            (board.has_queenside_castling_rights(chess.BLACK), "q"),
        )
        if allowed
    ) or "-"
    return (
        _fact("side_to_move", "white" if board.turn else "black"),
        _fact("in_check", board.is_check()),
        _fact("legal_move_count", board.legal_moves.count()),
        _fact("material_balance_cp", material),
        _fact("castling_rights", castling),
    )


class StockfishProvider:
    def __init__(self, path: str | None = None, node_budget: int = NODE_BUDGET) -> None:
        self.path = resolve_stockfish(path)
        self.node_budget = node_budget
        self.binary_sha256 = sha256_file(self.path)

    def analyze(self, fen: str, considered_move_uci: str | None = None) -> EngineEvidence:
        board, normalized_fen = normalize_fen(fen)
        position_id = canonical_hash({"schema_version": "1.0", "normalized_fen": normalized_fen})
        if board.is_game_over(claim_draw=False):
            return EngineEvidence(
                position_id=position_id,
                fen=normalized_fen,
                side_to_move="white" if board.turn else "black",
                engine=self._metadata(name="Stockfish", author=None),
                terminal_reason=board.outcome(claim_draw=False).termination.name.lower(),  # type: ignore[union-attr]
                candidates=(),
                board_facts=extract_board_facts(board),
            )

        considered = None
        if considered_move_uci:
            try:
                considered = chess.Move.from_uci(considered_move_uci)
            except ValueError as exc:
                raise ValueError("The considered move is not valid UCI notation") from exc
            if considered not in board.legal_moves:
                raise ValueError("The considered move is illegal in this position")

        engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
        try:
            supported = engine.options
            applied = {key: value for key, value in ENGINE_OPTIONS.items() if key in supported}
            engine.configure(applied)
            identity = engine.id
            infos = engine.analyse(
                board,
                chess.engine.Limit(nodes=self.node_budget),
                multipv=min(3, board.legal_moves.count()),
                info=chess.engine.INFO_ALL,
            )
            candidates = [self._candidate(board, info, rank) for rank, info in enumerate(infos, 1)]
            if considered is not None:
                best_move = chess.Move.from_uci(candidates[0].move_uci)
                best_constrained = engine.analyse(
                    board,
                    chess.engine.Limit(nodes=self.node_budget),
                    root_moves=[best_move],
                    info=chess.engine.INFO_ALL,
                )
                candidates[0] = self._candidate(board, best_constrained, 1)
                if considered != best_move:
                    considered_rank = next(
                        (
                            index
                            for index, item in enumerate(candidates, 1)
                            if item.move_uci == considered.uci()
                        ),
                        min(3, len(candidates) + 1),
                    )
                    considered_info = engine.analyse(
                        board,
                        chess.engine.Limit(nodes=self.node_budget),
                        root_moves=[considered],
                        info=chess.engine.INFO_ALL,
                    )
                    compared = self._candidate(board, considered_info, considered_rank)
                    existing = next(
                        (
                            index
                            for index, item in enumerate(candidates)
                            if item.move_uci == considered.uci()
                        ),
                        None,
                    )
                    if existing is None:
                        candidates = (candidates[:2] + [compared])[:3]
                    else:
                        candidates[existing] = compared
            metadata = self._metadata(identity.get("name", "Stockfish"), identity.get("author"))
        finally:
            engine.quit()

        return EngineEvidence(
            position_id=position_id,
            fen=normalized_fen,
            side_to_move="white" if board.turn else "black",
            engine=metadata,
            candidates=tuple(candidates),
            board_facts=extract_board_facts(board),
        )

    def _metadata(self, name: str, author: str | None) -> EngineMetadata:
        return EngineMetadata(
            name=name,
            author=author,
            binary_sha256=self.binary_sha256,
            options={**ENGINE_OPTIONS, "MultiPV": 3, "nodes": self.node_budget},
            node_budget=self.node_budget,
        )

    def _candidate(
        self, board: chess.Board, info: chess.engine.InfoDict, rank: int
    ) -> CandidateEvidence:
        pv = info.get("pv") or []
        if not pv:
            raise RuntimeError("Stockfish returned no principal variation")
        first = pv[0]
        score = info["score"].pov(board.turn)
        mate = score.mate()
        cp = None if mate is not None else score.score(mate_score=100_000)
        wdl_value = info.get("wdl")
        wdl = None
        if wdl_value is not None:
            relative = wdl_value.pov(board.turn)
            wdl = WDL(win=relative.wins, draw=relative.draws, loss=relative.losses)
        payload = {
            "rank": rank,
            "move_uci": first.uci(),
            "score_cp": cp,
            "mate_in": mate,
            "nodes": int(info.get("nodes", self.node_budget)),
            "pv_uci": [move.uci() for move in pv[:16]],
        }
        return CandidateEvidence(
            evidence_id=canonical_hash(payload),
            rank=rank,
            move_uci=first.uci(),
            move_san=board.san(first),
            score_cp=cp,
            mate_in=mate,
            wdl_permille=wdl,
            depth=info.get("depth"),
            seldepth=info.get("seldepth"),
            nodes=int(info.get("nodes", self.node_budget)),
            pv_uci=tuple(move.uci() for move in pv[:16]),
        )
