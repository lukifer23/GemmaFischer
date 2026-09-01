from __future__ import annotations

import hashlib
import os
import shutil
import threading
import uuid
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import chess
import chess.engine

from .domain import (
    WDL,
    BoardFact,
    BoardMoveResult,
    CandidateEvidence,
    CandidateSet,
    ConceptEvidence,
    EngineEvidence,
    EngineMetadata,
    EngineTurnResult,
    GameDifficulty,
    LegalMovesResult,
    MoveComparisonEvidence,
    canonical_hash,
    normalize_fen,
)

NODE_BUDGET = 250_000
ENGINE_OPTIONS: dict[str, int | bool] = {
    "Threads": 1,
    "Hash": 64,
    "UCI_ShowWDL": True,
}
GAME_SKILL_LEVEL = {
    GameDifficulty.CASUAL: 4,
    GameDifficulty.CLUB: 10,
    GameDifficulty.STRONG: 18,
}
GAME_NODE_RATIO = {
    GameDifficulty.CASUAL: 0.08,
    GameDifficulty.CLUB: 0.32,
    GameDifficulty.STRONG: 1.0,
}
ANALYSIS_SKILL_LEVEL = 20


class EngineUnavailable(RuntimeError):
    pass


class EngineOperationCancelled(RuntimeError):
    pass


class EngineOperationPreempted(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineOperation:
    token: str
    kind: Literal["analysis", "gameplay"]


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


def _fact(position_id: str, fact_type: str, value: bool | int | str) -> BoardFact:
    payload = {
        "schema_version": "2.0",
        "position_id": position_id,
        "fact_type": fact_type,
        "value": value,
        "source": "python-chess",
    }
    return BoardFact(evidence_id=canonical_hash(payload), fact_type=fact_type, value=value)  # type: ignore[arg-type]


def extract_board_facts(board: chess.Board, position_id: str) -> tuple[BoardFact, ...]:
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
    castling = (
        "".join(
            symbol
            for allowed, symbol in (
                (board.has_kingside_castling_rights(chess.WHITE), "K"),
                (board.has_queenside_castling_rights(chess.WHITE), "Q"),
                (board.has_kingside_castling_rights(chess.BLACK), "k"),
                (board.has_queenside_castling_rights(chess.BLACK), "q"),
            )
            if allowed
        )
        or "-"
    )
    return (
        _fact(position_id, "side_to_move", "white" if board.turn else "black"),
        _fact(position_id, "in_check", board.is_check()),
        _fact(position_id, "legal_move_count", board.legal_moves.count()),
        _fact(position_id, "material_balance_cp", material),
        _fact(position_id, "castling_rights", castling),
    )


def legal_moves_for_square(fen: str, from_square: str) -> LegalMovesResult:
    board, _ = normalize_fen(fen)
    source = chess.parse_square(from_square)
    moves = tuple(move.uci() for move in board.legal_moves if move.from_square == source)
    destinations = tuple(dict.fromkeys(move[2:4] for move in moves))
    return LegalMovesResult(
        from_square=from_square,
        moves_uci=moves,
        destinations=destinations,
    )


def validate_player_move(fen: str, move_uci: str) -> None:
    board, _ = normalize_fen(fen)
    if board.is_game_over(claim_draw=True):
        raise ValueError("The game is already over")
    normalized_move = move_uci.lower()
    if len(normalized_move) == 4:
        source: int | None
        target: int | None
        try:
            source = chess.parse_square(normalized_move[:2])
            target = chess.parse_square(normalized_move[2:])
        except ValueError:
            source = target = None
        piece = board.piece_at(source) if source is not None else None
        if (
            piece
            and piece.piece_type == chess.PAWN
            and target is not None
            and chess.square_rank(target) in {0, 7}
        ):
            raise ValueError("Promotion move requires q, r, b, or n suffix")
    try:
        move = chess.Move.from_uci(normalized_move)
    except ValueError as exc:
        raise ValueError("The move is not valid UCI notation") from exc
    if move not in board.legal_moves:
        raise ValueError("That move is illegal in this position")


def extract_concepts(
    board: chess.Board,
    position_id: str,
    candidates: list[CandidateEvidence],
) -> tuple[ConceptEvidence, ...]:
    concepts: list[ConceptEvidence] = []
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
    }
    for candidate in candidates:
        move = chess.Move.from_uci(candidate.move_uci)
        captured = board.piece_at(move.to_square)
        if board.is_en_passant(move):
            captured = chess.Piece(chess.PAWN, not board.turn)
        raw: list[tuple[str, bool | int]] = [
            ("capture", board.is_capture(move)),
            ("promotion", move.promotion is not None),
            ("castling", board.is_castling(move)),
        ]
        moving = board.piece_at(move.from_square)
        home_rank = chess.square_rank(move.from_square) in ({0} if board.turn else {7})
        raw.append(
            (
                "development",
                bool(
                    moving
                    and moving.piece_type in {chess.KNIGHT, chess.BISHOP}
                    and home_rank
                ),
            )
        )
        if captured is not None:
            raw.append(("material_change", piece_values.get(captured.piece_type, 0)))
        after = board.copy(stack=False)
        after.push(move)
        raw.append(("check", after.is_check()))
        if len(candidate.pv_uci) > 1:
            reply = chess.Move.from_uci(candidate.pv_uci[1])
            if reply in after.legal_moves:
                after.push(reply)
                raw.append(("opponent_check", after.is_check()))
        for concept, value in raw:
            if value is False or value == 0:
                continue
            payload = {
                "schema_version": "2.0",
                "position_id": position_id,
                "candidate_id": candidate.evidence_id,
                "concept": concept,
                "value": value,
            }
            concepts.append(
                ConceptEvidence(
                    evidence_id=canonical_hash(payload),
                    position_id=position_id,
                    candidate_id=candidate.evidence_id,
                    concept=concept,  # type: ignore[arg-type]
                    value=value,
                )
            )
    return tuple(concepts)


class StockfishProvider:
    def __init__(self, path: str | None = None, node_budget: int = NODE_BUDGET) -> None:
        self.path = resolve_stockfish(path)
        self.node_budget = node_budget
        self.binary_sha256 = sha256_file(self.path)
        self._engine: chess.engine.SimpleEngine | None = None
        self._condition = threading.Condition()
        self._active_operation: EngineOperation | None = None
        self._analysis_waiters: set[str] = set()
        self._gameplay_waiters: deque[str] = deque()
        self._interrupt_reasons: dict[str, Literal["cancelled", "preempted"]] = {}
        self._closed = False
        self._started_at: datetime | None = None
        self._applied_options: dict[str, int | str | bool | None] = {}

    def __enter__(self) -> StockfishProvider:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Explicit service ownership/context managers are preferred. This is a
        # final guard for abandoned short-lived callers so python-chess does not
        # leave its transport thread and Stockfish child alive.
        with suppress(Exception):
            self.close()

    def close(self) -> None:
        with self._condition:
            if self._closed and self._engine is None:
                return
            self._closed = True
            engine, self._engine = self._engine, None
            active = self._active_operation is not None
            self._condition.notify_all()
        if engine is not None:
            if active:
                engine.close()
                return
            try:
                engine.quit()
            except (BrokenPipeError, TimeoutError, chess.engine.EngineTerminatedError):
                engine.close()

    def interrupt_analysis(
        self, operation_id: str, reason: Literal["cancelled", "preempted"] = "cancelled"
    ) -> bool:
        """Interrupt an exact active or waiting analysis operation, never gameplay."""
        with self._condition:
            active = self._active_operation
            is_active = active == EngineOperation(operation_id, "analysis")
            if not is_active and operation_id not in self._analysis_waiters:
                return False
            self._interrupt_reasons[operation_id] = reason
            engine = None
            if is_active:
                engine, self._engine = self._engine, None
            self._condition.notify_all()
        if engine is not None:
            engine.close()
        return True

    def operation_status(self) -> EngineOperation | None:
        with self._condition:
            return self._active_operation

    def _acquire_analysis(self, operation_id: str) -> None:
        with self._condition:
            self._analysis_waiters.add(operation_id)
            try:
                while self._active_operation is not None or self._gameplay_waiters:
                    if self._closed:
                        raise EngineUnavailable("The Stockfish provider is closed")
                    self._raise_if_interrupted_locked(operation_id)
                    self._condition.wait()
                if self._closed:
                    raise EngineUnavailable("The Stockfish provider is closed")
                self._raise_if_interrupted_locked(operation_id)
                self._active_operation = EngineOperation(operation_id, "analysis")
            finally:
                self._analysis_waiters.discard(operation_id)

    def _acquire_gameplay(self, operation_id: str) -> None:
        engine_to_close: chess.engine.SimpleEngine | None = None
        with self._condition:
            if self._closed:
                raise EngineUnavailable("The Stockfish provider is closed")
            self._gameplay_waiters.append(operation_id)
            active = self._active_operation
            if active is not None and active.kind == "analysis":
                self._interrupt_reasons[active.token] = "preempted"
                engine_to_close, self._engine = self._engine, None
        if engine_to_close is not None:
            engine_to_close.close()
        with self._condition:
            while (
                self._active_operation is not None
                or not self._gameplay_waiters
                or self._gameplay_waiters[0] != operation_id
            ):
                if self._closed:
                    with suppress(ValueError):
                        self._gameplay_waiters.remove(operation_id)
                    raise EngineUnavailable("The Stockfish provider is closed")
                self._condition.wait()
            self._gameplay_waiters.popleft()
            self._active_operation = EngineOperation(operation_id, "gameplay")

    def _release_operation(self, operation_id: str) -> None:
        with self._condition:
            if self._active_operation and self._active_operation.token == operation_id:
                self._active_operation = None
            self._interrupt_reasons.pop(operation_id, None)
            self._condition.notify_all()

    def _raise_if_interrupted_locked(self, operation_id: str) -> None:
        reason = self._interrupt_reasons.pop(operation_id, None)
        if reason == "cancelled":
            raise EngineOperationCancelled(operation_id)
        if reason == "preempted":
            raise EngineOperationPreempted(operation_id)

    def _raise_if_interrupted(self, operation_id: str) -> None:
        with self._condition:
            self._raise_if_interrupted_locked(operation_id)

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        if self._closed:
            raise EngineUnavailable("The Stockfish provider is closed")
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
            applied = {
                key: value for key, value in ENGINE_OPTIONS.items() if key in self._engine.options
            }
            # Gameplay deliberately lowers Stockfish's Skill Level. Analysis must
            # always restore full strength when this persistent process is reused.
            if "Skill Level" in self._engine.options:
                applied["Skill Level"] = ANALYSIS_SKILL_LEVEL
            self._engine.configure(applied)
            self._applied_options = dict(applied)
            self._started_at = datetime.now(UTC)
        return self._engine

    def analyze(
        self,
        fen: str,
        considered_move_uci: str | None = None,
        *,
        operation_id: str | None = None,
    ) -> EngineEvidence:
        board, normalized_fen = normalize_fen(fen)
        position_id = canonical_hash({"schema_version": "2.0", "normalized_fen": normalized_fen})
        if board.is_game_over(claim_draw=False):
            return EngineEvidence(
                position_id=position_id,
                fen=normalized_fen,
                side_to_move="white" if board.turn else "black",
                engine=self._metadata(name="Stockfish", author=None),
                terminal_reason=board.outcome(claim_draw=False).termination.name.lower(),  # type: ignore[union-attr]
                candidate_set=None,
                board_facts=extract_board_facts(board, position_id),
            )

        considered = None
        if considered_move_uci:
            try:
                considered = chess.Move.from_uci(considered_move_uci)
            except ValueError as exc:
                raise ValueError("The considered move is not valid UCI notation") from exc
            if considered not in board.legal_moves:
                raise ValueError("The considered move is illegal in this position")

        operation_id = operation_id or uuid.uuid4().hex
        self._acquire_analysis(operation_id)
        try:
            engine = self._ensure_engine()
            # A gameplay request or cancellation can arrive while the UCI
            # process is starting, before there is an engine handle to close.
            # Honor the token before issuing the first command in that case.
            self._raise_if_interrupted(operation_id)
            analysis_options = {
                key: value for key, value in ENGINE_OPTIONS.items() if key in engine.options
            }
            if "Skill Level" in engine.options:
                analysis_options["Skill Level"] = ANALYSIS_SKILL_LEVEL
            engine.configure(analysis_options)
            self._applied_options = dict(analysis_options)
            if "Clear Hash" in engine.options:
                engine.configure({"Clear Hash": None})
            identity = engine.id
            infos = engine.analyse(
                board,
                chess.engine.Limit(nodes=self.node_budget),
                multipv=min(3, board.legal_moves.count()),
                info=chess.engine.INFO_ALL,
            )
            candidates = [
                self._candidate(board, info, rank, position_id)
                for rank, info in enumerate(infos, 1)
            ]
            comparison = None
            if considered is not None:
                best_move = chess.Move.from_uci(candidates[0].move_uci)
                best_constrained = engine.analyse(
                    board,
                    chess.engine.Limit(nodes=self.node_budget),
                    root_moves=[best_move],
                    info=chess.engine.INFO_ALL,
                )
                considered_info = (
                    best_constrained
                    if considered == best_move
                    else engine.analyse(
                        board,
                        chess.engine.Limit(nodes=self.node_budget),
                        root_moves=[considered],
                        info=chess.engine.INFO_ALL,
                    )
                )
                comparison = self._comparison(
                    board, position_id, best_move, best_constrained, considered, considered_info
                )
            metadata = self._metadata(identity.get("name", "Stockfish"), identity.get("author"))
            self._raise_if_interrupted(operation_id)
        except (BrokenPipeError, chess.engine.EngineTerminatedError):
            self._raise_if_interrupted(operation_id)
            raise
        finally:
            self._release_operation(operation_id)

        candidate_set_id = canonical_hash(
            {
                "schema_version": "2.0",
                "position_id": position_id,
                "engine_sha256": self.binary_sha256,
                "node_budget": self.node_budget,
                "candidate_ids": [item.evidence_id for item in candidates],
            }
        )

        return EngineEvidence(
            position_id=position_id,
            fen=normalized_fen,
            side_to_move="white" if board.turn else "black",
            engine=metadata,
            candidate_set=CandidateSet(
                evidence_id=candidate_set_id,
                position_id=position_id,
                candidates=tuple(candidates),
            ),
            move_comparison=comparison,
            board_facts=extract_board_facts(board, position_id),
            concepts=extract_concepts(board, position_id, candidates),
        )

    def play_move(
        self,
        fen: str,
        move_uci: str,
        *,
        engine_reply: bool,
        difficulty: GameDifficulty,
    ) -> BoardMoveResult:
        validate_player_move(fen, move_uci)
        board, normalized_fen = normalize_fen(fen)
        normalized_move = move_uci.lower()
        human_move = chess.Move.from_uci(normalized_move)

        human_san = board.san(human_move)
        board.push(human_move)
        fen_after_human = board.fen(en_passant="fen")
        engine_move_uci: str | None = None
        engine_move_san: str | None = None
        engine_name: str | None = None
        engine_nodes = 0

        if engine_reply and not board.is_game_over(claim_draw=True):
            engine_nodes = max(1, round(self.node_budget * GAME_NODE_RATIO[difficulty]))
            operation_id = uuid.uuid4().hex
            self._acquire_gameplay(operation_id)
            try:
                engine = self._ensure_engine()
                applied = dict(self._applied_options)
                if "Skill Level" in engine.options:
                    applied["Skill Level"] = GAME_SKILL_LEVEL[difficulty]
                engine.configure(applied)
                self._applied_options = applied
                engine_name = engine.id.get("name", "Stockfish")
                reply = engine.play(board, chess.engine.Limit(nodes=engine_nodes)).move
                if reply is None:
                    raise RuntimeError("Stockfish returned no move")
                engine_move_uci = reply.uci()
                engine_move_san = board.san(reply)
                board.push(reply)
            finally:
                self._release_operation(operation_id)

        outcome = board.outcome(claim_draw=True)
        return BoardMoveResult(
            fen_before=normalized_fen,
            fen_after_human=fen_after_human,
            fen=board.fen(en_passant="fen"),
            human_move_uci=normalized_move,
            human_move_san=human_san,
            engine_move_uci=engine_move_uci,
            engine_move_san=engine_move_san,
            engine_name=engine_name,
            engine_nodes=engine_nodes,
            game_over=outcome is not None,
            outcome=outcome.result() if outcome else None,
            turn="white" if board.turn else "black",
        )

    def play_engine_turn(
        self,
        fen: str,
        *,
        difficulty: GameDifficulty,
    ) -> EngineTurnResult:
        board, normalized_fen = normalize_fen(fen)
        if board.is_game_over(claim_draw=True):
            raise ValueError("The game is already over")
        engine_nodes = max(1, round(self.node_budget * GAME_NODE_RATIO[difficulty]))
        operation_id = uuid.uuid4().hex
        self._acquire_gameplay(operation_id)
        try:
            engine = self._ensure_engine()
            applied = dict(self._applied_options)
            if "Skill Level" in engine.options:
                applied["Skill Level"] = GAME_SKILL_LEVEL[difficulty]
            engine.configure(applied)
            engine_name = engine.id.get("name", "Stockfish")
            move = engine.play(board, chess.engine.Limit(nodes=engine_nodes)).move
            if move is None:
                raise RuntimeError("Stockfish returned no move")
            move_uci = move.uci()
            move_san = board.san(move)
            board.push(move)
        finally:
            self._release_operation(operation_id)
        outcome = board.outcome(claim_draw=True)
        return EngineTurnResult(
            fen_before=normalized_fen,
            fen=board.fen(en_passant="fen"),
            move_uci=move_uci,
            move_san=move_san,
            engine_name=engine_name,
            engine_nodes=engine_nodes,
            game_over=outcome is not None,
            outcome=outcome.result() if outcome else None,
            turn="white" if board.turn else "black",
        )

    def _metadata(self, name: str, author: str | None) -> EngineMetadata:
        return EngineMetadata(
            name=name,
            author=author,
            binary_sha256=self.binary_sha256,
            options={**self._applied_options, "MultiPV": 3, "nodes": self.node_budget},
            node_budget=self.node_budget,
            started_at=self._started_at,
        )

    def _candidate(
        self, board: chess.Board, info: chess.engine.InfoDict, rank: int, position_id: str
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
            "schema_version": "2.0",
            "position_id": position_id,
            "engine_sha256": self.binary_sha256,
            "node_budget": self.node_budget,
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

    def _comparison(
        self,
        board: chess.Board,
        position_id: str,
        engine_move: chess.Move,
        engine_info: chess.engine.InfoDict,
        considered_move: chess.Move,
        considered_info: chess.engine.InfoDict,
    ) -> MoveComparisonEvidence:
        engine_score = engine_info["score"].pov(board.turn)
        considered_score = considered_info["score"].pov(board.turn)
        engine_mate = engine_score.mate()
        considered_mate = considered_score.mate()
        engine_cp = None if engine_mate is not None else engine_score.score(mate_score=100_000)
        considered_cp = (
            None if considered_mate is not None else considered_score.score(mate_score=100_000)
        )
        engine_value = engine_score.score(mate_score=100_000)
        considered_value = considered_score.score(mate_score=100_000)
        assert engine_value is not None and considered_value is not None
        delta = engine_value - considered_value
        outcome: str = "equal" if abs(delta) <= 15 else (
            "engine_better" if delta > 0 else "considered_better"
        )
        payload = {
            "schema_version": "2.0",
            "position_id": position_id,
            "engine_sha256": self.binary_sha256,
            "node_budget_each": self.node_budget,
            "engine_move_uci": engine_move.uci(),
            "considered_move_uci": considered_move.uci(),
            "engine_score_cp": engine_cp,
            "engine_mate_in": engine_mate,
            "considered_score_cp": considered_cp,
            "considered_mate_in": considered_mate,
        }
        return MoveComparisonEvidence(
            evidence_id=canonical_hash(payload),
            position_id=position_id,
            engine_move_uci=engine_move.uci(),
            considered_move_uci=considered_move.uci(),
            engine_score_cp=engine_cp,
            engine_mate_in=engine_mate,
            considered_score_cp=considered_cp,
            considered_mate_in=considered_mate,
            outcome=outcome,  # type: ignore[arg-type]
            node_budget_each=self.node_budget,
        )
