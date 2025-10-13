# MoE Router Assessment (2025-10-13)

## Overview
We ran the curated router validation suite (`data/validation/eval_suite.jsonl`) using the lightweight routing harness added to `scripts/run_evaluation_suite.py`. The harness operates in `--router-only` mode so it can exercise `ChessMoERouter` without requiring the full Gemma weights, which are unavailable in this environment. LC0 remains configured as the primary engine in `configs/default.yaml`, but the evaluation focuses on routing latency and target expert selection.

Two configurations were executed:

1. **MoE Enabled:** Router decides among UCI, tutor, and director experts.
2. **MoE Disabled:** All requests are forced to the UCI/LC0 expert to emulate the fallback configuration.

Results are persisted in `reports/router_eval_lc0.json` and `reports/router_eval_lc0_nomoe.json` for reproducibility.

## Key Metrics
| Configuration | Routing Accuracy | Overall Format Accuracy | Avg Latency (s) | Avg Confidence |
| ------------- | ---------------- | ----------------------- | --------------- | -------------- |
| MoE Enabled   | 82.9%            | 85.7%                   | 0.002           | 0.87           |
| MoE Disabled  | 0.0%             | 14.3%                   | ~0.0001         | 0.00           |

*Metrics extracted from the JSON reports produced by the evaluation harness.*

### Category-Level Observations
- The router is reliable on **pure move**, **tactical patterns**, **mixed analysis**, and **position analysis** queries (100% routing accuracy).
- **Rules explanation** prompts are consistently misrouted to the UCI expert, yielding only 40% routing accuracy and zero estimated format compliance. Targeted retraining is needed here.
- **Endgame principles** accuracy drops to 60%, indicating the director/tutor balance needs refinement.

These insights come directly from the category breakdowns in the MoE-enabled report.

### MoE vs. LC0-Only Quality
When MoE is disabled, non-move categories show 0% routing and format accuracy because the single-engine fallback cannot satisfy explanatory prompts. This underscores the necessity of the router for any experience beyond raw move suggestions.

## Recommendation
**Action:** Retrain the router rather than pruning or sunsetting it.

**Justification:**
- MoE provides a dramatic improvement over the LC0-only baseline across every non-move category.
- Misroutes cluster in a few semantic buckets (rules explanation, endgame principles, some opening strategy), which can be addressed with additional labeled data.
- The updated configuration flags these categories for targeted retraining and marks the action plan as `targeted_retrain`, aligning operational settings with the remediation plan.

## Next Steps
1. Curate additional labeled prompts covering rules explanation, endgame principles, and opening strategy.
2. Retrain the router checkpoint using the new supervision, then update `router_checkpoint_path` once validated.
3. Re-run the evaluation harness to verify routing accuracy exceeds the 90% threshold on the problematic categories before promoting the checkpoint.
