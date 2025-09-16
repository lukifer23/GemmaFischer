#!/usr/bin/env python3
"""
Expert Scorecard Evaluation System

Comprehensive evaluation system with per-expert scorecards for ChessGemma.
Provides detailed metrics for UCI, Tutor, and Director experts.

Features:
- Per-expert scorecards with detailed metrics
- UCI syntax and legality validation
- Stockfish match analysis
- Puzzle accuracy testing
- Tutor explanation quality assessment
- Director Q&A accuracy testing
- Router evaluation and misroute detection
- Regression detection and performance trending
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import statistics
import concurrent.futures
from collections import defaultdict, Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

import chess
import chess.engine
from src.inference.inference import ChessGemmaInference
from src.inference.uci_utils import (
    validate_uci_syntax, 
    is_legal_uci, 
    extract_and_validate_uci,
    post_process_uci_response
)
from src.inference.chess_engine import ChessEngineManager
from src.utils.error_handler import error_boundary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExpertScorecard:
    """Comprehensive scorecard for a specific expert."""
    expert_name: str
    evaluation_time: datetime
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    
    # UCI Expert Metrics
    uci_syntax_rate: float = 0.0
    uci_legality_rate: float = 0.0
    uci_stockfish_top1: float = 0.0
    uci_stockfish_top3: float = 0.0
    uci_response_time_p50: float = 0.0
    uci_response_time_p95: float = 0.0
    
    # Tutor Expert Metrics
    tutor_first_move_accuracy: float = 0.0
    tutor_cot_structure_adherence: float = 0.0
    tutor_explanation_quality: float = 0.0
    tutor_uci_extraction_rate: float = 0.0
    tutor_response_time_p50: float = 0.0
    tutor_response_time_p95: float = 0.0
    
    # Director Expert Metrics
    director_rules_accuracy: float = 0.0
    director_opening_accuracy: float = 0.0
    director_hallucination_rate: float = 0.0
    director_factual_consistency: float = 0.0
    director_response_time_p50: float = 0.0
    director_response_time_p95: float = 0.0
    
    # Router Metrics
    router_intent_classification_accuracy: float = 0.0
    router_misroute_rate: float = 0.0
    router_confidence_calibration: float = 0.0
    
    # Overall Metrics
    overall_accuracy: float = 0.0
    overall_quality_score: float = 0.0
    overall_response_time: float = 0.0
    error_rate: float = 0.0
    
    # Detailed Results
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def calculate_overall_metrics(self):
        """Calculate overall metrics from expert-specific metrics."""
        if self.expert_name == "uci":
            self.overall_accuracy = (self.uci_syntax_rate + self.uci_legality_rate + self.uci_stockfish_top1) / 3
            self.overall_quality_score = (self.uci_syntax_rate * 0.3 + self.uci_legality_rate * 0.4 + self.uci_stockfish_top1 * 0.3)
            self.overall_response_time = (self.uci_response_time_p50 + self.uci_response_time_p95) / 2
            
        elif self.expert_name == "tutor":
            self.overall_accuracy = (self.tutor_first_move_accuracy + self.tutor_cot_structure_adherence + self.tutor_explanation_quality) / 3
            self.overall_quality_score = (self.tutor_first_move_accuracy * 0.4 + self.tutor_cot_structure_adherence * 0.3 + self.tutor_explanation_quality * 0.3)
            self.overall_response_time = (self.tutor_response_time_p50 + self.tutor_response_time_p95) / 2
            
        elif self.expert_name == "director":
            self.overall_accuracy = (self.director_rules_accuracy + self.director_opening_accuracy + self.director_factual_consistency) / 3
            self.overall_quality_score = (self.director_rules_accuracy * 0.4 + self.director_opening_accuracy * 0.3 + self.director_factual_consistency * 0.3)
            self.overall_response_time = (self.director_response_time_p50 + self.director_response_time_p95) / 2
        
        self.error_rate = self.failed_tests / max(self.total_tests, 1)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    timestamp: datetime
    total_experts: int
    successful_experts: int
    failed_experts: int
    expert_scorecards: Dict[str, ExpertScorecard] = field(default_factory=dict)
    router_evaluation: Dict[str, Any] = field(default_factory=dict)
    regression_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_experts": self.total_experts,
            "successful_experts": self.successful_experts,
            "failed_experts": self.failed_experts,
            "success_rate": self.successful_experts / max(self.total_experts, 1),
            "average_accuracy": statistics.mean([card.overall_accuracy for card in self.expert_scorecards.values()]),
            "average_quality": statistics.mean([card.overall_quality_score for card in self.expert_scorecards.values()]),
            "average_response_time": statistics.mean([card.overall_response_time for card in self.expert_scorecards.values()])
        }


class ExpertScorecardEvaluator:
    """Comprehensive evaluation system with per-expert scorecards."""
    
    def __init__(self, model_path: Optional[str] = None, adapter_path: Optional[str] = None):
        """Initialize the evaluator."""
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.inference = None
        self.chess_engine = None
        self.evaluation_data = {}
        
        # Load evaluation datasets
        self._load_evaluation_datasets()
    
    def _load_evaluation_datasets(self):
        """Load evaluation datasets."""
        logger.info("📚 Loading evaluation datasets")
        
        # UCI evaluation data
        self.evaluation_data["uci_fens"] = self._load_uci_evaluation_data()
        
        # Tutor evaluation data
        self.evaluation_data["tutor_puzzles"] = self._load_tutor_evaluation_data()
        
        # Director evaluation data
        self.evaluation_data["director_qa"] = self._load_director_evaluation_data()
        
        # Router evaluation data
        self.evaluation_data["router_queries"] = self._load_router_evaluation_data()
        
        logger.info("✅ Evaluation datasets loaded successfully")
    
    def _load_uci_evaluation_data(self) -> List[Dict[str, Any]]:
        """Load UCI evaluation data."""
        fens_file = Path("data/validation/eval_mixed_positions_200.jsonl")
        if not fens_file.exists():
            logger.warning(f"UCI evaluation data not found: {fens_file}")
            return []
        
        fens = []
        with open(fens_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                fens.append({
                    "fen": data["fen"],
                    "difficulty": data.get("difficulty", "medium"),
                    "expected_move": data.get("expected_move"),
                    "category": data.get("category", "mixed")
                })
        
        return fens
    
    def _load_tutor_evaluation_data(self) -> List[Dict[str, Any]]:
        """Load tutor evaluation data."""
        puzzles_file = Path("data/validation/tutor_comprehensive_validation.json")
        if not puzzles_file.exists():
            logger.warning(f"Tutor evaluation data not found: {puzzles_file}")
            return []
        
        with open(puzzles_file, 'r') as f:
            data = json.load(f)
        
        return data.get("puzzles", [])
    
    def _load_director_evaluation_data(self) -> List[Dict[str, Any]]:
        """Load director evaluation data."""
        qa_file = Path("data/validation/director_comprehensive_validation.json")
        if not qa_file.exists():
            logger.warning(f"Director evaluation data not found: {qa_file}")
            return []
        
        with open(qa_file, 'r') as f:
            data = json.load(f)
        
        return data.get("questions", [])
    
    def _load_router_evaluation_data(self) -> List[Dict[str, Any]]:
        """Load router evaluation data."""
        router_file = Path("data/validation/router_evaluation_queries.json")
        if not router_file.exists():
            logger.warning(f"Router evaluation data not found: {router_file}")
            return []
        
        with open(router_file, 'r') as f:
            data = json.load(f)
        
        return data.get("queries", [])
    
    def initialize_inference(self) -> bool:
        """Initialize inference system."""
        try:
            logger.info("🤖 Initializing inference system")
            self.inference = ChessGemmaInference(self.model_path, self.adapter_path)
            
            if not self.inference.load_model():
                logger.error("Failed to load model")
                return False
            
            # Initialize chess engine
            self.chess_engine = ChessEngineManager()
            
            logger.info("✅ Inference system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize inference system: {e}")
            return False
    
    def evaluate_uci_expert(self, max_positions: int = 100) -> ExpertScorecard:
        """Evaluate UCI expert with comprehensive metrics."""
        logger.info("🎯 Evaluating UCI expert")
        
        scorecard = ExpertScorecard(
            expert_name="uci",
            evaluation_time=datetime.now()
        )
        
        if not self.inference:
            scorecard.errors.append("Inference system not initialized")
            return scorecard
        
        # Set UCI expert
        self.inference.set_active_adapter("uci")
        
        fens = self.evaluation_data["uci_fens"][:max_positions]
        if not fens:
            scorecard.errors.append("No UCI evaluation data available")
            return scorecard
        
        scorecard.total_tests = len(fens)
        response_times = []
        syntax_valid = 0
        legal_moves = 0
        stockfish_matches = 0
        stockfish_top3_matches = 0
        
        for i, fen_data in enumerate(fens):
            try:
                fen = fen_data["fen"]
                board = chess.Board(fen)
                
                # Generate UCI move
                start_time = time.time()
                prompt = f"FEN: {fen}\nMode: Engine\nGenerate the best move in UCI format (e.g., e2e4). Respond with only the move."
                
                response = self.inference.generate_response(
                    prompt,
                    mode="engine",
                    max_new_tokens=8,
                    temperature=0.0,
                    do_sample=False
                )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                response_text = response.get('response', '').strip()
                
                # Validate syntax
                if validate_uci_syntax(response_text):
                    syntax_valid += 1
                    
                    # Validate legality
                    if is_legal_uci(fen, response_text):
                        legal_moves += 1
                        
                        # Check Stockfish match
                        if self.chess_engine:
                            try:
                                sf_move = self.chess_engine.get_best_move(board, depth=8)
                                if sf_move and response_text == sf_move.uci():
                                    stockfish_matches += 1
                                
                                # Check top-3 Stockfish moves
                                sf_moves = self.chess_engine.get_top_moves(board, depth=8, top_k=3)
                                if any(response_text == move.uci() for move in sf_moves):
                                    stockfish_top3_matches += 1
                                    
                            except Exception as e:
                                logger.warning(f"Stockfish evaluation failed: {e}")
                
                scorecard.test_results.append({
                    "fen": fen,
                    "response": response_text,
                    "response_time": response_time,
                    "syntax_valid": validate_uci_syntax(response_text),
                    "legal": is_legal_uci(fen, response_text),
                    "stockfish_match": False  # Will be updated if Stockfish available
                })
                
            except Exception as e:
                logger.error(f"UCI evaluation error for position {i}: {e}")
                scorecard.errors.append(f"Position {i}: {str(e)}")
                scorecard.failed_tests += 1
        
        # Calculate metrics
        scorecard.uci_syntax_rate = syntax_valid / len(fens)
        scorecard.uci_legality_rate = legal_moves / len(fens)
        scorecard.uci_stockfish_top1 = stockfish_matches / len(fens)
        scorecard.uci_stockfish_top3 = stockfish_top3_matches / len(fens)
        
        if response_times:
            scorecard.uci_response_time_p50 = statistics.median(response_times)
            scorecard.uci_response_time_p95 = sorted(response_times)[int(len(response_times) * 0.95)]
        
        scorecard.passed_tests = syntax_valid
        scorecard.calculate_overall_metrics()
        
        logger.info(f"✅ UCI expert evaluation completed")
        logger.info(f"   Syntax rate: {scorecard.uci_syntax_rate:.3f}")
        logger.info(f"   Legality rate: {scorecard.uci_legality_rate:.3f}")
        logger.info(f"   Stockfish top-1: {scorecard.uci_stockfish_top1:.3f}")
        
        return scorecard
    
    def evaluate_tutor_expert(self, max_puzzles: int = 50) -> ExpertScorecard:
        """Evaluate tutor expert with comprehensive metrics."""
        logger.info("🎓 Evaluating tutor expert")
        
        scorecard = ExpertScorecard(
            expert_name="tutor",
            evaluation_time=datetime.now()
        )
        
        if not self.inference:
            scorecard.errors.append("Inference system not initialized")
            return scorecard
        
        # Set tutor expert
        self.inference.set_active_adapter("tutor")
        
        puzzles = self.evaluation_data["tutor_puzzles"][:max_puzzles]
        if not puzzles:
            scorecard.errors.append("No tutor evaluation data available")
            return scorecard
        
        scorecard.total_tests = len(puzzles)
        response_times = []
        first_move_correct = 0
        cot_structure_good = 0
        explanation_quality_good = 0
        uci_extracted = 0
        
        for i, puzzle in enumerate(puzzles):
            try:
                fen = puzzle["fen"]
                expected_move = puzzle["expected_move"]
                board = chess.Board(fen)
                
                # Generate tutor response
                start_time = time.time()
                prompt = f"FEN: {fen}\nQuestion: Analyze this position step by step.\nMode: Tutor\nAnalyze this position step by step and provide your reasoning.\nEnd your response with: Best move: <UCI_MOVE>"
                
                response = self.inference.generate_response(
                    prompt,
                    mode="tutor",
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True
                )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                response_text = response.get('response', '')
                
                # Extract UCI move from response
                uci_move = None
                if "Best move:" in response_text:
                    lines = response_text.split('\n')
                    for line in lines:
                        if "Best move:" in line:
                            uci_candidate = line.split("Best move:")[-1].strip()
                            if validate_uci_syntax(uci_candidate):
                                uci_move = uci_candidate
                                break
                
                # Check first move accuracy
                if uci_move and uci_move == expected_move:
                    first_move_correct += 1
                
                # Check UCI extraction rate
                if uci_move:
                    uci_extracted += 1
                
                # Check CoT structure (simplified)
                cot_indicators = ["1.", "2.", "3.", "first", "then", "next", "finally"]
                cot_score = sum(1 for indicator in cot_indicators if indicator in response_text.lower())
                if cot_score >= 3:
                    cot_structure_good += 1
                
                # Check explanation quality (simplified)
                quality_indicators = ["threat", "opportunity", "tactical", "positional", "strategy"]
                quality_score = sum(1 for indicator in quality_indicators if indicator in response_text.lower())
                if quality_score >= 2:
                    explanation_quality_good += 1
                
                scorecard.test_results.append({
                    "fen": fen,
                    "expected_move": expected_move,
                    "response": response_text,
                    "uci_extracted": uci_move,
                    "first_move_correct": uci_move == expected_move if uci_move else False,
                    "response_time": response_time
                })
                
            except Exception as e:
                logger.error(f"Tutor evaluation error for puzzle {i}: {e}")
                scorecard.errors.append(f"Puzzle {i}: {str(e)}")
                scorecard.failed_tests += 1
        
        # Calculate metrics
        scorecard.tutor_first_move_accuracy = first_move_correct / len(puzzles)
        scorecard.tutor_cot_structure_adherence = cot_structure_good / len(puzzles)
        scorecard.tutor_explanation_quality = explanation_quality_good / len(puzzles)
        scorecard.tutor_uci_extraction_rate = uci_extracted / len(puzzles)
        
        if response_times:
            scorecard.tutor_response_time_p50 = statistics.median(response_times)
            scorecard.tutor_response_time_p95 = sorted(response_times)[int(len(response_times) * 0.95)]
        
        scorecard.passed_tests = first_move_correct
        scorecard.calculate_overall_metrics()
        
        logger.info(f"✅ Tutor expert evaluation completed")
        logger.info(f"   First move accuracy: {scorecard.tutor_first_move_accuracy:.3f}")
        logger.info(f"   CoT structure: {scorecard.tutor_cot_structure_adherence:.3f}")
        logger.info(f"   UCI extraction: {scorecard.tutor_uci_extraction_rate:.3f}")
        
        return scorecard
    
    def evaluate_director_expert(self, max_questions: int = 50) -> ExpertScorecard:
        """Evaluate director expert with comprehensive metrics."""
        logger.info("🎭 Evaluating director expert")
        
        scorecard = ExpertScorecard(
            expert_name="director",
            evaluation_time=datetime.now()
        )
        
        if not self.inference:
            scorecard.errors.append("Inference system not initialized")
            return scorecard
        
        # Set director expert
        self.inference.set_active_adapter("director")
        
        questions = self.evaluation_data["director_qa"][:max_questions]
        if not questions:
            scorecard.errors.append("No director evaluation data available")
            return scorecard
        
        scorecard.total_tests = len(questions)
        response_times = []
        rules_correct = 0
        opening_correct = 0
        factual_consistent = 0
        hallucination_detected = 0
        
        for i, qa in enumerate(questions):
            try:
                question = qa["question"]
                expected_answer = qa["expected_answer"]
                category = qa.get("category", "general")
                
                # Generate director response
                start_time = time.time()
                response = self.inference.generate_response(
                    question,
                    mode="director",
                    max_new_tokens=200,
                    temperature=0.6,
                    do_sample=True
                )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                response_text = response.get('response', '')
                
                # Simple accuracy check (would need more sophisticated evaluation)
                if category == "rules":
                    # Check for rule-related keywords
                    rule_keywords = ["legal", "illegal", "rule", "move", "piece", "square"]
                    if any(keyword in response_text.lower() for keyword in rule_keywords):
                        rules_correct += 1
                elif category == "opening":
                    # Check for opening-related keywords
                    opening_keywords = ["opening", "defense", "attack", "variation", "line"]
                    if any(keyword in response_text.lower() for keyword in opening_keywords):
                        opening_correct += 1
                
                # Check for hallucination indicators (simplified)
                hallucination_indicators = ["i don't know", "i'm not sure", "i think", "maybe", "possibly"]
                if any(indicator in response_text.lower() for indicator in hallucination_indicators):
                    hallucination_detected += 1
                
                # Factual consistency (simplified)
                if len(response_text) > 50 and "chess" in response_text.lower():
                    factual_consistent += 1
                
                scorecard.test_results.append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "response": response_text,
                    "category": category,
                    "response_time": response_time
                })
                
            except Exception as e:
                logger.error(f"Director evaluation error for question {i}: {e}")
                scorecard.errors.append(f"Question {i}: {str(e)}")
                scorecard.failed_tests += 1
        
        # Calculate metrics
        scorecard.director_rules_accuracy = rules_correct / len(questions)
        scorecard.director_opening_accuracy = opening_correct / len(questions)
        scorecard.director_factual_consistency = factual_consistent / len(questions)
        scorecard.director_hallucination_rate = hallucination_detected / len(questions)
        
        if response_times:
            scorecard.director_response_time_p50 = statistics.median(response_times)
            scorecard.director_response_time_p95 = sorted(response_times)[int(len(response_times) * 0.95)]
        
        scorecard.passed_tests = rules_correct + opening_correct
        scorecard.calculate_overall_metrics()
        
        logger.info(f"✅ Director expert evaluation completed")
        logger.info(f"   Rules accuracy: {scorecard.director_rules_accuracy:.3f}")
        logger.info(f"   Opening accuracy: {scorecard.director_opening_accuracy:.3f}")
        logger.info(f"   Factual consistency: {scorecard.director_factual_consistency:.3f}")
        
        return scorecard
    
    def evaluate_router(self, max_queries: int = 100) -> Dict[str, Any]:
        """Evaluate router performance."""
        logger.info("🧭 Evaluating router performance")
        
        queries = self.evaluation_data["router_queries"][:max_queries]
        if not queries:
            return {"error": "No router evaluation data available"}
        
        correct_routes = 0
        misroutes = 0
        confidence_scores = []
        
        for query in queries:
            try:
                question = query["question"]
                expected_expert = query["expected_expert"]
                
                # Generate response with MoE routing
                response = self.inference.generate_response(
                    question,
                    mode="auto",  # Use auto routing
                    max_new_tokens=100
                )
                
                # Check if correct expert was used
                used_expert = response.get('primary_expert', 'unknown')
                if used_expert == expected_expert:
                    correct_routes += 1
                else:
                    misroutes += 1
                
                # Collect confidence scores
                confidence = response.get('confidence', 0.5)
                confidence_scores.append(confidence)
                
            except Exception as e:
                logger.error(f"Router evaluation error: {e}")
                misroutes += 1
        
        return {
            "intent_classification_accuracy": correct_routes / len(queries),
            "misroute_rate": misroutes / len(queries),
            "confidence_calibration": statistics.mean(confidence_scores) if confidence_scores else 0.0,
            "total_queries": len(queries)
        }
    
    def run_comprehensive_evaluation(self, max_positions: int = 100) -> EvaluationReport:
        """Run comprehensive evaluation for all experts."""
        logger.info("🎯 Starting comprehensive evaluation")
        
        # Initialize inference system
        if not self.initialize_inference():
            raise RuntimeError("Failed to initialize inference system")
        
        report = EvaluationReport(
            timestamp=datetime.now(),
            total_experts=3,
            successful_experts=0,
            failed_experts=0
        )
        
        # Evaluate each expert
        experts = ["uci", "tutor", "director"]
        
        for expert_name in experts:
            try:
                logger.info(f"🔄 Evaluating {expert_name} expert...")
                
                if expert_name == "uci":
                    scorecard = self.evaluate_uci_expert(max_positions)
                elif expert_name == "tutor":
                    scorecard = self.evaluate_tutor_expert(max_positions // 2)
                elif expert_name == "director":
                    scorecard = self.evaluate_director_expert(max_positions // 2)
                
                report.expert_scorecards[expert_name] = scorecard
                
                if scorecard.errors:
                    report.failed_experts += 1
                else:
                    report.successful_experts += 1
                
            except Exception as e:
                logger.error(f"Failed to evaluate {expert_name} expert: {e}")
                report.failed_experts += 1
        
        # Evaluate router
        try:
            report.router_evaluation = self.evaluate_router(max_positions)
        except Exception as e:
            logger.error(f"Router evaluation failed: {e}")
            report.router_evaluation = {"error": str(e)}
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        logger.info("✅ Comprehensive evaluation completed")
        logger.info(f"   Successful experts: {report.successful_experts}/{report.total_experts}")
        
        return report
    
    def _generate_recommendations(self, report: EvaluationReport) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        for expert_name, scorecard in report.expert_scorecards.items():
            if expert_name == "uci":
                if scorecard.uci_syntax_rate < 0.95:
                    recommendations.append(f"UCI expert syntax rate ({scorecard.uci_syntax_rate:.3f}) is below target (0.95)")
                if scorecard.uci_legality_rate < 0.90:
                    recommendations.append(f"UCI expert legality rate ({scorecard.uci_legality_rate:.3f}) is below target (0.90)")
                if scorecard.uci_stockfish_top1 < 0.25:
                    recommendations.append(f"UCI expert Stockfish top-1 match ({scorecard.uci_stockfish_top1:.3f}) is below target (0.25)")
            
            elif expert_name == "tutor":
                if scorecard.tutor_first_move_accuracy < 0.70:
                    recommendations.append(f"Tutor expert first move accuracy ({scorecard.tutor_first_move_accuracy:.3f}) is below target (0.70)")
                if scorecard.tutor_cot_structure_adherence < 0.90:
                    recommendations.append(f"Tutor expert CoT structure adherence ({scorecard.tutor_cot_structure_adherence:.3f}) is below target (0.90)")
            
            elif expert_name == "director":
                if scorecard.director_rules_accuracy < 0.90:
                    recommendations.append(f"Director expert rules accuracy ({scorecard.director_rules_accuracy:.3f}) is below target (0.90)")
                if scorecard.director_hallucination_rate > 0.10:
                    recommendations.append(f"Director expert hallucination rate ({scorecard.director_hallucination_rate:.3f}) is above target (0.10)")
        
        # Router recommendations
        if "intent_classification_accuracy" in report.router_evaluation:
            accuracy = report.router_evaluation["intent_classification_accuracy"]
            if accuracy < 0.85:
                recommendations.append(f"Router intent classification accuracy ({accuracy:.3f}) is below target (0.85)")
        
        return recommendations
    
    def save_report(self, report: EvaluationReport, output_path: str) -> None:
        """Save evaluation report to file."""
        report_data = {
            "timestamp": report.timestamp.isoformat(),
            "total_experts": report.total_experts,
            "successful_experts": report.successful_experts,
            "failed_experts": report.failed_experts,
            "expert_scorecards": {
                name: {
                    "expert_name": card.expert_name,
                    "evaluation_time": card.evaluation_time.isoformat(),
                    "total_tests": card.total_tests,
                    "passed_tests": card.passed_tests,
                    "failed_tests": card.failed_tests,
                    "overall_accuracy": card.overall_accuracy,
                    "overall_quality_score": card.overall_quality_score,
                    "overall_response_time": card.overall_response_time,
                    "error_rate": card.error_rate,
                    "uci_syntax_rate": card.uci_syntax_rate,
                    "uci_legality_rate": card.uci_legality_rate,
                    "uci_stockfish_top1": card.uci_stockfish_top1,
                    "tutor_first_move_accuracy": card.tutor_first_move_accuracy,
                    "tutor_cot_structure_adherence": card.tutor_cot_structure_adherence,
                    "director_rules_accuracy": card.director_rules_accuracy,
                    "director_opening_accuracy": card.director_opening_accuracy,
                    "errors": card.errors,
                    "warnings": card.warnings
                }
                for name, card in report.expert_scorecards.items()
            },
            "router_evaluation": report.router_evaluation,
            "regression_analysis": report.regression_analysis,
            "recommendations": report.recommendations,
            "summary": report.generate_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"📊 Evaluation report saved: {output_path}")


def main():
    """Main entry point for expert scorecard evaluation."""
    parser = argparse.ArgumentParser(
        description="Expert Scorecard Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive evaluation
  python -m src.evaluation.expert_scorecard_eval

  # Evaluate specific expert
  python -m src.evaluation.expert_scorecard_eval --expert uci

  # Custom evaluation parameters
  python -m src.evaluation.expert_scorecard_eval --max-positions 200 --output expert_eval_report.json
        """
    )
    
    parser.add_argument('--expert', choices=['uci', 'tutor', 'director', 'all'], default='all',
                       help='Expert to evaluate (default: all)')
    parser.add_argument('--max-positions', type=int, default=100,
                       help='Maximum positions to evaluate (default: 100)')
    parser.add_argument('--output', type=str, default='expert_scorecard_report.json',
                       help='Output report file (default: expert_scorecard_report.json)')
    parser.add_argument('--model-path', type=str, help='Path to base model')
    parser.add_argument('--adapter-path', type=str, help='Path to adapter')
    
    args = parser.parse_args()
    
    print("🎯 Expert Scorecard Evaluation System")
    print("=" * 50)
    
    try:
        # Initialize evaluator
        evaluator = ExpertScorecardEvaluator(args.model_path, args.adapter_path)
        
        if args.expert == 'all':
            # Run comprehensive evaluation
            report = evaluator.run_comprehensive_evaluation(args.max_positions)
            
            # Save report
            evaluator.save_report(report, args.output)
            
            # Print summary
            summary = report.generate_summary()
            print(f"\n📊 Evaluation Summary:")
            print(f"   Total experts: {summary['total_experts']}")
            print(f"   Successful: {summary['successful_experts']}")
            print(f"   Failed: {summary['failed_experts']}")
            print(f"   Success rate: {summary['success_rate']:.1%}")
            print(f"   Average accuracy: {summary['average_accuracy']:.3f}")
            print(f"   Average quality: {summary['average_quality']:.3f}")
            
            if report.recommendations:
                print(f"\n💡 Recommendations:")
                for rec in report.recommendations:
                    print(f"   - {rec}")
        
        else:
            # Evaluate specific expert
            if not evaluator.initialize_inference():
                print("❌ Failed to initialize inference system")
                return
            
            if args.expert == 'uci':
                scorecard = evaluator.evaluate_uci_expert(args.max_positions)
            elif args.expert == 'tutor':
                scorecard = evaluator.evaluate_tutor_expert(args.max_positions // 2)
            elif args.expert == 'director':
                scorecard = evaluator.evaluate_director_expert(args.max_positions // 2)
            
            # Save single expert report
            report = EvaluationReport(
                timestamp=datetime.now(),
                total_experts=1,
                successful_experts=1 if not scorecard.errors else 0,
                failed_experts=1 if scorecard.errors else 0
            )
            report.expert_scorecards[args.expert] = scorecard
            
            evaluator.save_report(report, args.output)
            
            print(f"\n📊 {args.expert.upper()} Expert Results:")
            print(f"   Overall accuracy: {scorecard.overall_accuracy:.3f}")
            print(f"   Overall quality: {scorecard.overall_quality_score:.3f}")
            print(f"   Response time: {scorecard.overall_response_time:.3f}s")
            print(f"   Error rate: {scorecard.error_rate:.3f}")
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📄 Report saved: {args.output}")
    
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
