#!/bin/bash

# Run ChessGemma Web App with Hybrid LC0 System Enabled

echo "🤖 Starting ChessGemma with Hybrid LC0 Integration"
echo "=================================================="

# Enable hybrid engine
export CHESSGEMMA_HYBRID_ENGINE=true

echo "✅ Hybrid LC0 engine enabled (CHESSGEMMA_HYBRID_ENGINE=true)"
echo ""
echo "🎯 Features:"
echo "   • Strategic intent selector (6 options)"
echo "   • Hybrid AI move button in Play Mode"
echo "   • Enhanced analysis display with LLM+LC0 details"
echo "   • 87.5% LLM strategic guidance + 12.5% LC0 precision"
echo ""
echo "🚀 Starting web application..."

# Run the web app
cd src/web && python app.py
