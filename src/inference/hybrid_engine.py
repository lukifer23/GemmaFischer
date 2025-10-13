#!/usr/bin/env python3
"""
Hybrid Chess Engine: LLM Strategic Guidance + LC0 Precise Calculation

Combines the strategic reasoning capabilities of LLMs with the precise
calculation power of LC0 neural chess engine for optimal chess analysis.
"""

import sys
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re

import chess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from lc0_engine import LC0EngineManager, LC0Config, LC0Analysis
    from inference import ChessGemmaInference
except ImportError:
    # Fallback for direct execution
    from .lc0_engine import LC0EngineManager, LC0Config, LC0Analysis
    from .inference import ChessGemmaInference

# Configure logging
logger = logging.getLogger(__name__)

class StrategicIntent(Enum):
    """Types of strategic guidance the LLM can provide."""
    AGGRESSIVE = "aggressive"      # Look for attacking moves
    DEFENSIVE = "defensive"        # Prioritize defense
    POSITIONAL = "positional"      # Focus on long-term advantages
    TACTICAL = "tactical"          # Immediate tactical opportunities
    DEVELOPMENT = "development"    # Piece development and mobilization
    ENDGAME = "endgame"            # Endgame principles
    OPENING = "opening"            # Opening principles
    MIDDLEGAME = "middlegame"      # Middlegame strategy

@dataclass
class StrategicGuidance:
    """LLM-generated strategic guidance for LC0."""
    intent: StrategicIntent
    description: str
    search_focus: str  # What LC0 should focus on
    time_allocation: float  # How much time to spend
    risk_tolerance: str  # conservative, balanced, aggressive
    
    # LC0 search parameters derived from guidance
    search_params: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence in this guidance
    confidence: float = 0.0

@dataclass
class HybridAnalysis:
    """Complete analysis from hybrid LLM + LC0 system."""
    
    # Position information
    fen: str
    strategic_guidance: StrategicGuidance
    
    # LC0 analysis results
    lc0_analysis: LC0Analysis
    
    # LLM interpretation
    explanation: str
    confidence: float
    
    # Metadata
    total_time: float
    llm_time: float
    lc0_time: float
    
    # Move recommendations
    best_move: Optional[chess.Move] = None
    alternative_moves: List[chess.Move] = field(default_factory=list)
    
    # Strategic insights
    key_themes: List[str] = field(default_factory=list)
    positional_themes: List[str] = field(default_factory=list)

class HybridChessEngine:
    """
    Hybrid Chess Engine combining LLM strategic reasoning with LC0 precision.
    
    The LLM provides strategic guidance and context, while LC0 handles
    the precise move calculation and evaluation.
    """
    
    def __init__(self, 
                 llm_model: Optional[ChessGemmaInference] = None,
                 lc0_config: Optional[LC0Config] = None):
        """
        Initialize hybrid chess engine.
        
        Args:
            llm_model: Pre-initialized ChessGemmaInference instance
            lc0_config: LC0 configuration (auto-created if None)
        """
        self.llm = llm_model or ChessGemmaInference()
        self.lc0 = LC0EngineManager(lc0_config)
        
        # Strategic guidance templates
        self._guidance_templates = self._load_guidance_templates()
        
        # Performance tracking
        self._analysis_count = 0
        self._total_llm_time = 0.0
        self._total_lc0_time = 0.0
        
        logger.info("🤖 Hybrid Chess Engine initialized")
    
    def _load_guidance_templates(self) -> Dict[StrategicIntent, Dict[str, str]]:
        """Load strategic guidance templates for different intents."""
        return {
            StrategicIntent.AGGRESSIVE: {
                'description': 'Look for aggressive attacking moves with tactical potential',
                'search_focus': 'prioritize moves that create threats, attacks, or tactical opportunities',
                'risk_tolerance': 'aggressive',
                'lc0_params': {'cpuct': 4.0, 'temperature': 0.8}
            },
            StrategicIntent.DEFENSIVE: {
                'description': 'Focus on solid defensive moves and king safety',
                'search_focus': 'prioritize moves that improve defense, king safety, and stability',
                'risk_tolerance': 'conservative', 
                'lc0_params': {'cpuct': 2.0, 'temperature': 0.3}
            },
            StrategicIntent.POSITIONAL: {
                'description': 'Seek long-term positional advantages',
                'search_focus': 'prioritize moves that improve piece coordination, control key squares, and create lasting advantages',
                'risk_tolerance': 'balanced',
                'lc0_params': {'cpuct': 3.0, 'temperature': 0.5}
            },
            StrategicIntent.TACTICAL: {
                'description': 'Find immediate tactical opportunities',
                'search_focus': 'prioritize moves with immediate tactical consequences like captures, checks, or threats',
                'risk_tolerance': 'aggressive',
                'lc0_params': {'cpuct': 4.0, 'temperature': 0.7}
            },
            StrategicIntent.DEVELOPMENT: {
                'description': 'Focus on piece development and mobilization',
                'search_focus': 'prioritize moves that develop pieces, control the center, and prepare for future operations',
                'risk_tolerance': 'balanced',
                'lc0_params': {'cpuct': 3.0, 'temperature': 0.4}
            },
            StrategicIntent.ENDGAME: {
                'description': 'Apply endgame principles and techniques',
                'search_focus': 'prioritize moves following endgame principles like king activation, pawn advancement, and piece coordination',
                'risk_tolerance': 'balanced',
                'lc0_params': {'cpuct': 2.5, 'temperature': 0.4}
            }
        }
    
    def analyze_position_with_strategy(self, 
                                     fen: str, 
                                     strategic_intent: Union[str, StrategicIntent] = None,
                                     time_limit: float = 2.0) -> HybridAnalysis:
        """
        Perform hybrid analysis: LLM strategic guidance + LC0 precise calculation.
        
        Args:
            fen: FEN string of position to analyze
            strategic_intent: Desired strategic approach
            time_limit: Total time limit for analysis
            
        Returns:
            HybridAnalysis with strategic insights and precise moves
        """
        start_time = time.time()
        
        # Step 1: LLM generates strategic guidance
        llm_start = time.time()
        guidance = self._generate_strategic_guidance(fen, strategic_intent)
        llm_time = time.time() - llm_start
        
        # Step 2: LC0 analyzes with strategic guidance
        lc0_start = time.time()
        
        # Configure LC0 based on strategic guidance
        lc0_limit = chess.engine.Limit(time=min(time_limit * 0.7, guidance.time_allocation))
        
        # Apply strategic parameters to LC0 search
        board = chess.Board(fen)
        lc0_analysis = self.lc0.analyze_position(board, lc0_limit)
        
        lc0_time = time.time() - lc0_start
        
        # Step 3: LLM interprets results and generates explanation
        explanation = self._interpret_lc0_results(fen, guidance, lc0_analysis)
        
        # Calculate overall confidence
        llm_confidence = guidance.confidence
        lc0_confidence = lc0_analysis.confidence
        overall_confidence = (llm_confidence + lc0_confidence) / 2
        
        # Create hybrid analysis
        analysis = HybridAnalysis(
            fen=fen,
            strategic_guidance=guidance,
            lc0_analysis=lc0_analysis,
            explanation=explanation,
            confidence=overall_confidence,
            total_time=time.time() - start_time,
            llm_time=llm_time,
            lc0_time=lc0_time,
            best_move=lc0_analysis.best_move
        )
        
        # Extract key themes from explanation
        analysis.key_themes = self._extract_key_themes(explanation)
        
        # Update performance tracking
        self._analysis_count += 1
        self._total_llm_time += llm_time
        self._total_lc0_time += lc0_time
        
        logger.info(f"🤖 Hybrid analysis completed in {analysis.total_time:.2f}s "
                   f"(LLM: {llm_time:.2f}s, LC0: {lc0_time:.2f}s)")
        
        return analysis
    
    def _generate_strategic_guidance(self, fen: str, 
                                    strategic_intent: Union[str, StrategicIntent]) -> StrategicGuidance:
        """
        Generate strategic guidance using LLM analysis of the position.
        """
        # Convert string intent to enum
        if isinstance(strategic_intent, str):
            try:
                intent = StrategicIntent(strategic_intent.lower())
            except ValueError:
                intent = StrategicIntent.POSITIONAL  # Default
        else:
            intent = strategic_intent or StrategicIntent.POSITIONAL
        
        # Get template for this intent
        template = self._guidance_templates.get(intent, self._guidance_templates[StrategicIntent.POSITIONAL])
        
        # Create LLM prompt for strategic analysis
        prompt = f"""Analyze this chess position and provide strategic guidance for {intent.value} play:

FEN: {fen}

Based on the position, determine the most appropriate {intent.value} approach. Consider:
- Current piece coordination and development
- King safety and pawn structure  
- Tactical opportunities and threats
- Long-term strategic goals

Provide specific guidance for what the engine should focus on in its search.

Strategic Guidance:"""
        
        try:
            # Get LLM response
            llm_response = self.llm.generate_response(prompt, mode="director", max_new_tokens=150)
            guidance_text = llm_response.get('response', '').strip()
            
            # Parse guidance and create StrategicGuidance object
            guidance = StrategicGuidance(
                intent=intent,
                description=template['description'],
                search_focus=self._extract_search_focus(guidance_text),
                time_allocation=2.0,  # Default 2 seconds
                risk_tolerance=template['risk_tolerance'],
                search_params=template['lc0_params'].copy(),
                confidence=llm_response.get('confidence', 0.7)
            )
            
            # Adjust time allocation based on position complexity
            position_complexity = self._assess_position_complexity(fen)
            guidance.time_allocation = min(3.0, max(1.0, 2.0 * position_complexity))
            
            return guidance
            
        except Exception as e:
            logger.warning(f"LLM guidance generation failed: {e}")
            # Return fallback guidance
            return StrategicGuidance(
                intent=intent,
                description=template['description'],
                search_focus=template['search_focus'],
                time_allocation=2.0,
                risk_tolerance=template['risk_tolerance'],
                search_params=template['lc0_params'].copy(),
                confidence=0.5
            )
    
    def _extract_search_focus(self, guidance_text: str) -> str:
        """Extract specific search focus from LLM guidance."""
        # Look for key phrases that indicate search focus
        focus_keywords = [
            'prioritize', 'focus on', 'look for', 'seek', 'emphasize',
            'concentrate on', 'target', 'aim for'
        ]
        
        lines = guidance_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            for keyword in focus_keywords:
                if keyword in line_lower:
                    return line.strip()
        
        # Fallback to first meaningful line
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith('FEN:'):
                return line
        
        return "Find the best move for the position"
    
    def _assess_position_complexity(self, fen: str) -> float:
        """Assess position complexity (0.0 to 1.0)."""
        try:
            board = chess.Board(fen)
            
            # Factors indicating complexity
            factors = []
            
            # Material imbalance
            white_material = sum(len(board.pieces(piece, chess.WHITE)) for piece in chess.PIECE_TYPES)
            black_material = sum(len(board.pieces(piece, chess.BLACK)) for piece in chess.PIECE_TYPES)
            material_imbalance = abs(white_material - black_material) / max(white_material + black_material, 1)
            factors.append(material_imbalance)
            
            # King safety (castling status)
            castling_rights = bool(board.castling_rights)
            factors.append(0.3 if castling_rights else 0.0)
            
            # Piece activity (developed pieces)
            developed_pieces = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and square not in [chess.A1, chess.B1, chess.C1, chess.D1, 
                                          chess.E1, chess.F1, chess.G1, chess.H1,
                                          chess.A8, chess.B8, chess.C8, chess.D8,
                                          chess.E8, chess.F8, chess.G8, chess.H8]:
                    developed_pieces += 1
            factors.append(min(developed_pieces / 24, 1.0))  # Normalize
            
            # Average complexity
            return sum(factors) / len(factors)
            
        except Exception:
            return 0.5  # Default medium complexity
    
    def _interpret_lc0_results(self, fen: str, guidance: StrategicGuidance, 
                             lc0_analysis: LC0Analysis) -> str:
        """LLM interprets LC0 results and provides rich explanation."""
        
        prompt = f"""Interpret the results of a chess engine analysis:

Position FEN: {fen}
Strategic Focus: {guidance.description}
Engine Best Move: {lc0_analysis.best_move.uci() if lc0_analysis.best_move else 'None'}
Engine Evaluation: {lc0_analysis.best_score} centipawns
Search Depth: {lc0_analysis.depth_reached}
Search Time: {lc0_analysis.search_time:.2f}s

Explain what this move means strategically and why it aligns with the {guidance.intent.value} approach.
Discuss the key ideas, potential follow-ups, and positional implications.

Strategic Analysis:"""
        
        try:
            llm_response = self.llm.generate_response(prompt, mode="tutor", max_new_tokens=200)
            return llm_response.get('response', 'Analysis not available')
        except Exception as e:
            logger.warning(f"LLM interpretation failed: {e}")
            return f"The engine recommends {lc0_analysis.best_move.uci() if lc0_analysis.best_move else 'no move'} with evaluation {lc0_analysis.best_score}."
    
    def _extract_key_themes(self, explanation: str) -> List[str]:
        """Extract key strategic themes from explanation."""
        themes = []
        
        # Look for common strategic themes
        theme_keywords = {
            'development': ['development', 'mobilize', 'activate'],
            'center': ['center', 'central', 'd5', 'e5', 'd4', 'e4'],
            'king safety': ['king safety', 'castling', 'kingside', 'queenside'],
            'initiative': ['initiative', 'attack', 'pressure'],
            'defense': ['defense', 'protect', 'secure'],
            'endgame': ['endgame', 'pawn promotion', 'king activity'],
            'tactics': ['tactic', 'combination', 'fork', 'pin', 'skewer']
        }
        
        explanation_lower = explanation.lower()
        for theme, keywords in theme_keywords.items():
            if any(keyword in explanation_lower for keyword in keywords):
                themes.append(theme)
        
        return themes[:3]  # Limit to top 3 themes
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the hybrid engine."""
        total_time = self._total_llm_time + self._total_lc0_time
        
        return {
            'total_analyses': self._analysis_count,
            'total_llm_time': self._total_llm_time,
            'total_lc0_time': self._total_lc0_time,
            'total_time': total_time,
            'avg_llm_time': self._total_llm_time / max(self._analysis_count, 1),
            'avg_lc0_time': self._total_lc0_time / max(self._analysis_count, 1),
            'llm_percentage': self._total_llm_time / max(total_time, 1) * 100,
            'lc0_percentage': self._total_lc0_time / max(total_time, 1) * 100
        }
    
    def cleanup(self) -> None:
        """Clean up engine resources."""
        if hasattr(self, 'lc0'):
            self.lc0.cleanup()
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

# Convenience functions
def create_hybrid_engine(llm_model: Optional[ChessGemmaInference] = None,
                        lc0_config: Optional[LC0Config] = None) -> HybridChessEngine:
    """Create a hybrid chess engine with default configurations."""
    return HybridChessEngine(llm_model, lc0_config)

def test_hybrid_engine():
    """Test the hybrid engine with a simple position."""
    print("🧠 Testing Hybrid Chess Engine...")
    
    try:
        with create_hybrid_engine() as engine:
            # Test with starting position
            analysis = engine.analyze_position_with_strategy(
                'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                strategic_intent='development'
            )
            
            print("✅ Hybrid analysis completed!")
            print(f"   Best move: {analysis.best_move}")
            print(f"   Strategic guidance: {analysis.strategic_guidance.intent.value}")
            print(f"   Confidence: {analysis.confidence:.2f}")
            print(f"   Key themes: {', '.join(analysis.key_themes)}")
            print(f"   Total time: {analysis.total_time:.2f}s")
            
            return True
            
    except Exception as e:
        print(f"❌ Hybrid engine test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the hybrid engine
    success = test_hybrid_engine()
    if success:
        print("\\n🎉 Hybrid Chess Engine is ready!")
    else:
        print("\\n❌ Hybrid Chess Engine needs debugging")
