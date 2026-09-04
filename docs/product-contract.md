# Product Contract

GemmaFischer turns one of the learner's games into a short, evidence-backed practice loop.
The default surface is Learn. Position Lab preserves the live position, player-versus-engine,
engine exhibition, and single-position tutor workflows from 0.2.

## Game-to-mastery loop

1. Import exactly one standard-chess PGN, choose White or Black, or identify the player name.
2. Parse and validate the complete mainline before starting engine work.
3. Screen only the selected player's decision positions with a bounded Stockfish search.
4. Rank material mistakes by forced-mate loss, centipawn loss, and source-ply order.
5. Re-run up to six shortlisted positions at the configured full node budget.
6. Publish at most three learning moments. The public job never contains the preferred move.
7. Ask the learner to move on the board. Legal moves come from python-chess on the server.
8. Grade with a fresh, equal-budget Stockfish comparison. A first wrong answer does not reveal
   the answer. Retry reveals it. An engine-equivalent move is accepted as equivalent.
9. Offer a near-transfer position only when another deeply analyzed moment shares a detected
   concept. Do not substitute a synthetic or unrelated position.
10. Schedule delayed review. Successful delayed reviews expand to 3, 7, 14, then 30 days;
    a lapse returns the interval to one day. Two successful delayed reviews mark mastery.

## Authority and privacy

Stockfish 18 owns chess facts, ranking, grading, and evidence. Deterministic code owns visible
factual prose. Optional Gemma may select only supplied IDs and must degrade cleanly. Imported
PGNs, hidden answers, attempts, and review state stay in the local SQLite database. Study,
review, and progress reads require the per-launch capability cookie or header.

## Honest boundaries

No adapter is qualified or shipped. Production training remains unauthorized until the required
two-reviewer, adjudicated pedagogy labels exist and the candidate beats both deterministic and
untuned baselines. Browser automation is evidence for browser behavior, not physical-device,
VoiceOver, endurance, or human teaching value.
