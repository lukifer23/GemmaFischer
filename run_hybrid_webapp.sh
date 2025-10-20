#!/bin/bash

# GemmaFischer hybrid web app launcher

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🤖 Starting GemmaFischer with Hybrid LC0 Integration"
echo "===================================================="

echo "ℹ️  This launcher keeps the LC0 pool enabled by default."
echo "   Set GEMMAFISCHER_DISABLE_LC0_POOL=1 to force a fresh LC0 instance per run."
echo ""

if [[ "${GEMMAFISCHER_DISABLE_LC0_POOL:-0}" == "1" ]]; then
  export CHESSGEMMA_LC0_USE_POOL=0
  echo "⚙️  LC0 engine pool disabled for this session."
else
  export CHESSGEMMA_LC0_USE_POOL=1
fi

export CHESSGEMMA_MOE_ENABLED="${CHESSGEMMA_MOE_ENABLED:-1}"

echo "🎯 Features:"
echo "   • LC0 + Gemma hybrid analysis with live explanations"
echo "   • MoE routing across UCI, Tutor, and Director experts"
echo "   • System metrics panel with engine health checks"
echo ""
echo "🚀 Launching web application..."

cd "$PROJECT_ROOT"
python -m src.web.run_web_app
