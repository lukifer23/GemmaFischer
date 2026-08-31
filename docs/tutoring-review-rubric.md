# Blind Tutoring Review Rubric

This is the human acceptance gate. Automated scores can reject unsupported or
internally inconsistent lessons, but they cannot decide whether a lesson helps a
player understand the position.

## Review protocol

1. Freeze the position IDs, rating buckets, engine evidence, and both lesson
   variants before review.
2. Randomize variant order and replace profile names with opaque labels.
3. Use at least two independent reviewers who can read chess notation.
4. Reviewers may inspect the board and cited line, but not the producing model or
   other reviewers' scores.
5. Record every score and comment. Do not average away a correctness failure.

## Per-lesson scores

Score each dimension from 1 to 5.

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Correctness | Materially false or illegal | Correct but incomplete | Fully correct within cited evidence |
| Clarity | Confusing or opaque | Understandable with effort | Immediately understandable |
| Relevance | Distracts from the decision | Addresses part of it | Focuses on the decisive idea |
| Rating fit | Clearly mismatched | Generally appropriate | Precisely calibrated |
| Actionability | No usable next step | Some calculation guidance | Clear reusable thinking process |
| Harmful omission | Omits decisive danger | Misses secondary context | No important omission |

For harmful omission, 5 means no harmful omission and 1 means a severe omission.

Each row also records:

- `position_id`
- `rating_bucket`
- opaque `variant_id`
- six integer scores
- `preferred_variant` after both are scored
- free-text correctness and teaching comments
- reviewer ID and timestamp

## Gates

- No correctness or harmful-omission score below 3.
- Mean candidate correctness must equal deterministic correctness within 0.1.
- Mean candidate usefulness, the mean of clarity, relevance, rating fit, and
  actionability, must not be below deterministic usefulness.
- Inter-reviewer disagreements of two or more points require adjudication, not
  automatic averaging.
- Any cited evidence mismatch is an automated failure and never reaches human
  review.

Until a completed blind artifact satisfies these gates, the strongest allowed
claim is `automated-qualified-human-open`.
