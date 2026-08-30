"""
08_evaluate_jfleg.py

STAGE 8 of the RAG pipeline (evaluation-only, additive): Independent
Evaluation on JFLEG.

Purpose
-------
Measure how well the EXISTING RAG grammar-correction pipeline
(06_retrieve_context.py + 07_prompting.py, unmodified) performs against
JFLEG, a held-out human-annotated grammatical-error-correction benchmark
(jhu-clsp/jfleg) that is NOT part of the retrieval knowledge base.

Strict data separation (see README / task spec)
-------------------------------------------------
    agentlans/grammar-correction  -> Retrieval Knowledge Base (ChromaDB)
    jhu-clsp/jfleg                -> Independent Evaluation/Test Dataset

This script:
    - NEVER inserts JFLEG examples into ChromaDB.
    - NEVER creates embeddings for JFLEG for retrieval purposes.
    - NEVER puts JFLEG reference corrections into the LLM prompt.
    - Only ever sends 07_prompting.analyze_sentence() the raw JFLEG
      source sentence, exactly like a brand-new, unseen user query
      typed into the Streamlit app. Retrieved context still comes
      exclusively from the existing agentlans/grammar-correction
      ChromaDB collection, via the existing, untouched retrieval code.

Pipeline implemented here
--------------------------
    JFLEG test dataset
        -> original ungrammatical sentence
        -> 07_prompting.analyze_sentence(sentence)     [existing, untouched]
             -> 06_retrieve_context.retrieve_context()  [existing, untouched]
             -> OpenRouter (openai/gpt-4o-mini)          [existing, untouched]
        -> predicted corrected_sentence
        -> compare against JFLEG's 4 human reference corrections
        -> per-sentence + aggregate metrics
        -> results saved to JSON and CSV

This stage deliberately does NOT:
    - modify 06_retrieve_context.py or 07_prompting.py
    - modify config.py
    - write anything to ChromaDB
    - re-embed anything
    - change the Streamlit UI

Run independently with:
    python 08_evaluate_jfleg.py
    python 08_evaluate_jfleg.py --split test --limit 20
"""

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import config

# 07_prompting.py (and, transitively, 06_retrieve_context.py) starts with a
# digit, so it cannot be imported with a normal `import` statement. Loaded
# once at module scope via importlib, exactly the same pattern
# 07_prompting.py itself already uses to import 06_retrieve_context.py.
# This script only ever calls the existing public analyze_sentence()
# function - it never reaches into ChromaDB, the embedding model, or
# OpenRouter directly, and never duplicates that logic.
_prompting_module = importlib.import_module("07_prompting")

# JFLEG dataset identifier. Kept local to this evaluation script (rather
# than added to config.py) because it is an evaluation-only concern, not
# something any retrieval/generation stage needs to read.
JFLEG_DATASET_NAME: str = "jhu-clsp/jfleg"
JFLEG_VALID_SPLITS: Tuple[str, ...] = ("validation", "test")

# Output locations for this evaluation run.
EVAL_OUTPUT_DIR: str = os.path.join(config.PROJECT_ROOT, "evaluation_results")
DEFAULT_PREDICTIONS_JSON: str = os.path.join(
    EVAL_OUTPUT_DIR, "jfleg_predictions.json"
)
DEFAULT_PREDICTIONS_CSV: str = os.path.join(
    EVAL_OUTPUT_DIR, "jfleg_predictions.csv"
)
DEFAULT_SUMMARY_JSON: str = os.path.join(
    EVAL_OUTPUT_DIR, "jfleg_evaluation_summary.json"
)
# Append-only checkpoint used for resumable evaluation (--resume). One JSON
# object per line - the exact same shape build_result_row() produces. This
# file is additive/optional: it does not change the final predictions
# JSON/CSV or summary JSON schema in any way.
DEFAULT_CHECKPOINT_JSONL: str = os.path.join(
    EVAL_OUTPUT_DIR, "jfleg_predictions.checkpoint.jsonl"
)


# ---------------------------------------------------------------------------
# Step 1: Load JFLEG (independent evaluation dataset - never touches Chroma)
# ---------------------------------------------------------------------------


def load_jfleg_dataset(split: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load the JFLEG evaluation dataset from Hugging Face.

    JFLEG record shape (confirmed from the dataset card for
    jhu-clsp/jfleg):
        {
            "sentence": "<original ungrammatical sentence>",
            "corrections": [
                "<human reference 1>",
                "<human reference 2>",
                "<human reference 3>",
                "<human reference 4>",
            ],
        }

    This function ONLY reads JFLEG into memory for evaluation. It never
    writes JFLEG data anywhere near ChromaDB and never generates
    embeddings from it.

    Args:
        split: "validation" or "test".
        limit: Optional cap on the number of examples to load (useful
            for a quick smoke-test run before a full evaluation).

    Returns:
        A list of dicts: {"sentence": str, "references": List[str]}.

    Raises:
        RuntimeError: If the `datasets` package is missing or the
            dataset cannot be loaded.
        ValueError: If `split` is not a recognized JFLEG split.
    """
    if split not in JFLEG_VALID_SPLITS:
        raise ValueError(
            f"Invalid JFLEG split '{split}'. Must be one of {JFLEG_VALID_SPLITS}."
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is not installed. "
            "Install it with: pip install datasets"
        ) from exc

    try:
        hf_dataset = load_dataset(JFLEG_DATASET_NAME, split=split)
    except Exception as exc:  # noqa: BLE001 - surface any loader failure clearly
        raise RuntimeError(
            f"Failed to load JFLEG dataset '{JFLEG_DATASET_NAME}' "
            f"(split='{split}'). Original error: {exc}"
        ) from exc

    examples: List[Dict[str, Any]] = []
    for row in hf_dataset:
        sentence = row.get("sentence", "")
        corrections = row.get("corrections", [])

        if not isinstance(sentence, str) or sentence.strip() == "":
            continue
        if not isinstance(corrections, list) or len(corrections) == 0:
            continue

        references = [c for c in corrections if isinstance(c, str) and c.strip() != ""]
        if not references:
            continue

        examples.append({"sentence": sentence, "references": references})

        if limit is not None and len(examples) >= limit:
            break

    if not examples:
        raise RuntimeError(
            f"No usable JFLEG examples were loaded from split '{split}'."
        )

    return examples


# ---------------------------------------------------------------------------
# Step 2: Run each sentence through the EXISTING RAG pipeline (read-only)
# ---------------------------------------------------------------------------


def correct_sentence_via_existing_pipeline(
    sentence: str,
    api_key: Optional[str],
    model: str,
    top_k: int,
) -> Dict[str, Any]:
    """
    Send exactly the raw JFLEG source sentence through the existing,
    unmodified 07_prompting.analyze_sentence() function - the same
    function the Streamlit app calls for a live user query.

    JFLEG's human reference corrections are NOT passed in here and are
    NOT visible to analyze_sentence() in any way. Retrieved context
    still comes only from the existing agentlans/grammar-correction
    ChromaDB collection via the existing 06_retrieve_context.py.

    Args:
        sentence: The raw JFLEG "sentence" field (ungrammatical input).
        api_key: OpenRouter API key (or None to read from environment).
        model: OpenRouter model identifier.
        top_k: Number of retrieved reference examples.

    Returns:
        A dict:
            {
                "prediction": str or None,
                "has_error": bool or None,
                "num_retrieved": int,
                "parse_error": str or None,
                "call_error": str or None,
            }
        Exactly one of a usable prediction, a parse_error, or a
        call_error will be set, so failures are visible in the saved
        results rather than silently producing a fabricated prediction.
    """
    try:
        output = _prompting_module.analyze_sentence(
            sentence, api_key=api_key, model=model, top_k=top_k
        )
    except (ValueError, RuntimeError) as exc:
        return {
            "prediction": None,
            "has_error": None,
            "num_retrieved": 0,
            "parse_error": None,
            "call_error": str(exc),
        }

    result = output.get("result", {})
    num_retrieved = len(output.get("retrieved_examples", []) or [])

    if "parse_error" in result:
        return {
            "prediction": None,
            "has_error": None,
            "num_retrieved": num_retrieved,
            "parse_error": result["parse_error"],
            "call_error": None,
        }

    return {
        "prediction": result.get("corrected_sentence"),
        "has_error": result.get("has_error"),
        "num_retrieved": num_retrieved,
        "parse_error": None,
        "call_error": None,
    }


# ---------------------------------------------------------------------------
# Step 3: Metrics
# ---------------------------------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_comparison(text: str) -> str:
    """
    Normalize text for exact-match comparison: lowercase, collapse
    whitespace, strip leading/trailing whitespace, and drop a single
    trailing period if present (JFLEG references are space-tokenized,
    e.g. "... trees ." - the model's output is normal prose punctuation,
    so a raw string match would fail on tokenization differences alone
    rather than on genuine correction differences).

    Args:
        text: Raw sentence text.

    Returns:
        A normalized string suitable for exact-match comparison.
    """
    normalized = text.strip().lower()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    # JFLEG tokenizes punctuation with a leading space (e.g. "word ."),
    # which would otherwise make an otherwise-identical sentence fail
    # exact match purely on tokenization style.
    normalized = re.sub(r"\s+([.,!?;:])", r"\1", normalized)
    return normalized


def exact_match(prediction: str, references: List[str]) -> bool:
    """
    Check whether the normalized prediction exactly matches ANY of the
    normalized human reference corrections (JFLEG provides 4 references
    per sentence; matching any one of them counts as correct, which is
    the standard convention for multi-reference GEC evaluation).

    Args:
        prediction: The model's predicted corrected sentence.
        references: JFLEG's human reference corrections.

    Returns:
        True if the normalized prediction matches any normalized
        reference exactly, False otherwise.
    """
    normalized_prediction = normalize_for_comparison(prediction)
    return any(
        normalized_prediction == normalize_for_comparison(reference)
        for reference in references
    )


def compute_sentence_gleu(prediction: str, references: List[str]) -> Optional[float]:
    """
    Compute a sentence-level GLEU score for one prediction against
    JFLEG's multiple human references.

    IMPORTANT scope note (see SECTION 9 / the written explanation in the
    chat response for full detail): this uses NLTK's
    `sentence_gleu`, which implements the Google-GLEU metric from Wu et
    al. (2016) - a general MT fluency metric, adapted here with NLTK's
    native support for multiple references. It is NOT a re-implementation
    of the JFLEG-paper-specific GLEU (Napoles et al., 2015), which
    additionally weights n-grams by comparing against the *source*
    sentence to reward correctly-preserved spans and penalize
    unnecessary edits. That source-aware variant requires the
    `pip install gec-metrics`-style tooling from the original GEC-ranking
    repository, which is not a PyPI-installable dependency and is out of
    scope for a minimal, reproducible addition to this project. NLTK's
    GLEU is used here as the closest reliable, pip-installable, n-gram
    fluency proxy, and is reported as "GLEU (NLTK, Wu et al.)" throughout
    to avoid overstating precision against the original JFLEG benchmark
    numbers.

    Args:
        prediction: The model's predicted corrected sentence.
        references: JFLEG's human reference corrections.

    Returns:
        The sentence-level GLEU score in [0, 1], or None if the `nltk`
        package is not installed (caller falls back to reporting only
        exact-match and token-overlap metrics in that case).
    """
    try:
        from nltk.translate.gleu_score import sentence_gleu
    except ImportError:
        return None

    hypothesis_tokens = prediction.strip().split()
    reference_token_lists = [reference.strip().split() for reference in references]

    if not hypothesis_tokens or not any(reference_token_lists):
        return 0.0

    try:
        return float(sentence_gleu(reference_token_lists, hypothesis_tokens))
    except (ZeroDivisionError, ValueError):
        return 0.0


def compute_token_f1(prediction: str, references: List[str]) -> float:
    """
    Compute a simple, dependency-free token-overlap F1 between the
    prediction and the best-matching reference (max over references).

    This is included as a robust, always-available fallback/companion
    metric alongside GLEU: it needs no extra package, is easy to sanity
    check by hand, and degrades gracefully if `nltk` is unavailable.

    Args:
        prediction: The model's predicted corrected sentence.
        references: JFLEG's human reference corrections.

    Returns:
        The best token-level F1 score in [0, 1] across all references.
    """
    prediction_tokens = normalize_for_comparison(prediction).split()

    best_f1 = 0.0
    for reference in references:
        reference_tokens = normalize_for_comparison(reference).split()
        if not prediction_tokens or not reference_tokens:
            continue

        pred_counts: Dict[str, int] = {}
        for token in prediction_tokens:
            pred_counts[token] = pred_counts.get(token, 0) + 1

        overlap = 0
        for token in reference_tokens:
            if pred_counts.get(token, 0) > 0:
                overlap += 1
                pred_counts[token] -= 1

        if overlap == 0:
            continue

        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


# ---------------------------------------------------------------------------
# Step 3b: Resumability - checkpoint helpers (additive only)
#
# None of this changes dataset loading, retrieval, generation, or metric
# methodology. It only adds a way to persist per-example results as they
# are produced and to recognize, on a later run, which examples already
# have a genuinely successful result so they are not re-sent to the LLM.
# ---------------------------------------------------------------------------


_KEY_WHITESPACE_RE = re.compile(r"\s+")


def make_example_key(sentence: str) -> str:
    """
    Build a stable identity key for a JFLEG example from its source
    sentence text.

    JFLEG rows have no native ID, and relying on list position would be
    fragile (e.g. if a future run used a different --limit or the
    upstream dataset ordering ever changed). Hashing the sentence text
    itself instead ties the key to the actual content. Whitespace is
    collapsed and case is normalized first purely so two occurrences of
    "the same" sentence that differ only in incidental whitespace still
    map to the same key.

    Args:
        sentence: The raw JFLEG source sentence.

    Returns:
        A deterministic hex-digest key.
    """
    normalized = _KEY_WHITESPACE_RE.sub(" ", sentence.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_row_successful(row: Dict[str, Any]) -> bool:
    """
    Decide whether a saved result row counts as a completed, reusable
    success for resume purposes.

    Deliberately strict: this mirrors exactly the shape
    build_result_row() produces on its success path. A row missing any
    of these is treated as NOT done, so it will be retried rather than
    silently accepted - this is what makes previously-failed rows
    (OpenRouter errors, parse errors, etc.) automatically retried on
    --resume instead of being permanently skipped.

    Args:
        row: A result row, e.g. loaded from the checkpoint file.

    Returns:
        True only if the row has a real prediction, no recorded error,
        and syntactically valid evaluation metrics attached.
    """
    if row.get("prediction") is None:
        return False
    if row.get("error") is not None:
        return False
    if not isinstance(row.get("exact_match"), bool):
        return False
    if not isinstance(row.get("token_f1"), (int, float)):
        return False
    # "gleu" may legitimately be None (nltk not installed) - that alone
    # does not make the row a failure, so only check the key is present.
    if "gleu" not in row:
        return False
    return True


def load_checkpoint_rows(path: str) -> List[Dict[str, Any]]:
    """
    Load all valid rows from a JSONL checkpoint file, tolerating a
    truncated/corrupted final line left behind by a hard interruption
    (killed process, crash, power loss, etc.).

    Each line is parsed independently. A line that fails to parse as
    JSON is simply skipped rather than raising - every complete line
    before it was already fsync'd to disk by append_checkpoint_row()
    and remains valid, and the example the bad line was for will just
    be retried, same as if it had never been attempted.

    Args:
        path: Path to the checkpoint JSONL file.

    Returns:
        A list of parsed row dicts (possibly empty if the file does
        not exist yet or contains no valid lines).
    """
    if not os.path.exists(path):
        return []

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                # Incomplete/corrupted trailing line from an interrupted
                # write - safe to ignore, see docstring above.
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def build_completed_map(
    checkpoint_rows: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Reduce raw checkpoint rows (which may contain multiple attempts per
    example across failures/retries/duplicate sentences) down to a map
    of only the examples that are genuinely done.

    Rows are processed in file order. A failed row never overwrites an
    existing success for the same key, and a later success overwrites
    an earlier failure - so the map always reflects the best known
    outcome per example key regardless of how many attempts it took.

    Args:
        checkpoint_rows: Output of load_checkpoint_rows().

    Returns:
        A dict mapping example key -> its last known successful row.
    """
    completed: Dict[str, Dict[str, Any]] = {}
    for row in checkpoint_rows:
        sentence = row.get("sentence")
        if not isinstance(sentence, str) or not sentence.strip():
            continue
        if is_row_successful(row):
            completed[make_example_key(sentence)] = row
    return completed


def append_checkpoint_row(row: Dict[str, Any], path: str) -> None:
    """
    Append one complete result row to the JSONL checkpoint, durably.

    Writes exactly one JSON object per line, then flushes and fsyncs
    before returning, so that once this call completes the row is
    guaranteed to survive a subsequent crash/interruption - the
    checkpoint is only ever missing the example that was in flight at
    the moment of interruption, never anything already finished.

    Args:
        row: A single result row (see build_result_row()).
        path: Path to the checkpoint JSONL file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def reset_checkpoint_file(path: str) -> None:
    """
    Start a fresh, empty checkpoint file.

    Used when running WITHOUT --resume, so a stale checkpoint left over
    from an earlier/unrelated run can never leak into a from-scratch
    run's results.

    Args:
        path: Path to the checkpoint JSONL file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass


# ---------------------------------------------------------------------------
# Step 4: Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    examples: List[Dict[str, Any]],
    api_key: Optional[str],
    model: str,
    top_k: int,
    sleep_seconds: float,
    checkpoint_path: str,
    completed_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Run the full evaluation loop over all loaded JFLEG examples,
    reporting progress to stdout as it goes.

    Resumability (additive - does not change what gets computed for any
    example that is actually run, only which examples get skipped):
    before calling the existing pipeline for an example, its key
    (make_example_key(sentence)) is looked up in completed_map. If a
    prior *successful* result exists there - either loaded from an
    earlier --resume checkpoint, or produced earlier in this very same
    run for a duplicate sentence - it is reused as-is and the LLM is
    NOT called again. Every newly computed row (success or failure) is
    appended to the JSONL checkpoint immediately after it is built, so
    progress already made survives an interruption.

    Args:
        examples: Output of load_jfleg_dataset().
        api_key: OpenRouter API key (or None to read from environment).
        model: OpenRouter model identifier.
        top_k: Number of retrieved reference examples per query.
        sleep_seconds: Optional delay between OpenRouter calls (be kind
            to rate limits on larger runs).
        checkpoint_path: Path to the JSONL checkpoint file to append
            newly computed rows to.
        completed_map: Map of example key -> previously successful row,
            as produced by build_completed_map(). Pass an empty dict for
            a from-scratch run. Mutated in place as new successes occur,
            so duplicate sentences later in `examples` are also skipped.

    Returns:
        A list of per-example result dicts (see build_result_row()), in
        the same order as `examples` - a mix of reused checkpoint rows
        and newly computed ones.
    """
    total = len(examples)
    rows: List[Dict[str, Any]] = []

    for index, example in enumerate(examples, start=1):
        sentence = example["sentence"]
        references = example["references"]
        key = make_example_key(sentence)

        existing = completed_map.get(key)
        if existing is not None:
            # Already has a successful result (from a prior --resume'd
            # checkpoint, or an earlier duplicate in this same run) -
            # reuse it and skip the LLM call entirely.
            print(
                f"[{index}/{total}] SKIP (already completed): {sentence[:70]}",
                flush=True,
            )
            rows.append(existing)
            continue

        print(f"[{index}/{total}] {sentence[:70]}", flush=True)

        pipeline_output = correct_sentence_via_existing_pipeline(
            sentence, api_key=api_key, model=model, top_k=top_k
        )

        row = build_result_row(sentence, references, pipeline_output)
        rows.append(row)

        # Persist immediately, regardless of success/failure: an
        # interruption after this point loses at most the example
        # currently in flight, never anything already finished.
        append_checkpoint_row(row, checkpoint_path)
        if is_row_successful(row):
            # Prevents a duplicate sentence appearing later in `examples`
            # from triggering a second, unnecessary LLM call.
            completed_map[key] = row

        status = (
            "OK"
            if row["prediction"] is not None
            else f"FAILED ({row['error']})"
        )
        gleu_display = (
            f"{row['gleu']:.3f}" if row["gleu"] is not None else "n/a"
        )
        print(
            f"    -> {status} | exact_match={row['exact_match']} "
            f"| gleu={gleu_display} | token_f1={row['token_f1']:.3f}",
            flush=True,
        )

        if sleep_seconds > 0 and index < total:
            time.sleep(sleep_seconds)

    return rows


def build_result_row(
    sentence: str,
    references: List[str],
    pipeline_output: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Assemble one saved result row from a JFLEG example and the existing
    pipeline's output for it.

    Args:
        sentence: The original JFLEG ungrammatical sentence.
        references: JFLEG's human reference corrections.
        pipeline_output: Return value of
            correct_sentence_via_existing_pipeline().

    Returns:
        A flat dict ready to serialize to JSON/CSV.
    """
    prediction = pipeline_output["prediction"]

    error = pipeline_output["call_error"] or pipeline_output["parse_error"]

    if prediction is None:
        return {
            "sentence": sentence,
            "references": references,
            "prediction": None,
            "has_error": pipeline_output["has_error"],
            "num_retrieved": pipeline_output["num_retrieved"],
            "exact_match": False,
            "gleu": None,
            "token_f1": 0.0,
            "error": error,
        }

    return {
        "sentence": sentence,
        "references": references,
        "prediction": prediction,
        "has_error": pipeline_output["has_error"],
        "num_retrieved": pipeline_output["num_retrieved"],
        "exact_match": exact_match(prediction, references),
        "gleu": compute_sentence_gleu(prediction, references),
        "token_f1": compute_token_f1(prediction, references),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Step 5: Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute corpus-level aggregate metrics from all per-sentence rows.

    Args:
        rows: Output of evaluate().

    Returns:
        A summary dict with counts, success/failure rates, and averaged
        metrics computed only over successfully-predicted sentences.
    """
    total = len(rows)
    successful_rows = [row for row in rows if row["prediction"] is not None]
    failed_rows = [row for row in rows if row["prediction"] is None]
    num_successful = len(successful_rows)

    gleu_scores = [row["gleu"] for row in successful_rows if row["gleu"] is not None]

    summary: Dict[str, Any] = {
        "total_examples": total,
        "successful_predictions": num_successful,
        "failed_predictions": len(failed_rows),
        "failure_rate": (len(failed_rows) / total) if total else None,
        "exact_match_rate": (
            sum(1 for row in successful_rows if row["exact_match"]) / num_successful
            if num_successful
            else None
        ),
        "mean_token_f1": (
            sum(row["token_f1"] for row in successful_rows) / num_successful
            if num_successful
            else None
        ),
        "mean_gleu": (sum(gleu_scores) / len(gleu_scores)) if gleu_scores else None,
        "gleu_available": len(gleu_scores) > 0,
        "gleu_scored_examples": len(gleu_scores),
        "mean_retrieved_examples": (
            sum(row["num_retrieved"] for row in successful_rows) / num_successful
            if num_successful
            else None
        ),
    }
    return summary


# ---------------------------------------------------------------------------
# Step 6: Saving results
# ---------------------------------------------------------------------------


def save_predictions_json(rows: List[Dict[str, Any]], path: str) -> None:
    """Save all per-sentence results as a JSON array."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def save_predictions_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Save all per-sentence results as a flat CSV (references joined by '||')."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "sentence",
        "prediction",
        "references",
        "exact_match",
        "gleu",
        "token_f1",
        "has_error",
        "num_retrieved",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sentence": row["sentence"],
                    "prediction": row["prediction"],
                    "references": " || ".join(row["references"]),
                    "exact_match": row["exact_match"],
                    "gleu": row["gleu"],
                    "token_f1": row["token_f1"],
                    "has_error": row["has_error"],
                    "num_retrieved": row["num_retrieved"],
                    "error": row["error"],
                }
            )


def save_summary_json(summary: Dict[str, Any], run_config: Dict[str, Any], path: str) -> None:
    """Save the aggregate summary plus the exact run configuration used, for reproducibility."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"run_config": run_config, "summary": summary}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for this evaluation run."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the existing RAG grammar-correction pipeline against "
            "the independent JFLEG benchmark. JFLEG is never added to "
            "ChromaDB and never used for retrieval."
        )
    )
    parser.add_argument(
        "--split",
        choices=JFLEG_VALID_SPLITS,
        default="test",
        help="JFLEG split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of JFLEG examples to evaluate (for a quick run).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.DEFAULT_TOP_K,
        help=f"Number of retrieved examples per query (default: {config.DEFAULT_TOP_K}, "
        "same default the existing pipeline uses).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.OPENROUTER_MODEL,
        help=f"OpenRouter model identifier (default: {config.OPENROUTER_MODEL}, "
        "same as the existing pipeline).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between OpenRouter calls (default: 0.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=EVAL_OUTPUT_DIR,
        help=f"Directory to save results to (default: {EVAL_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previous run using the JSONL checkpoint file in "
            "--output-dir. Examples with a previously successful result "
            "are reused without calling the LLM again; examples that "
            "previously failed (or were never reached) are retried. "
            "Without this flag, any existing checkpoint is reset and "
            "every example is evaluated from scratch, exactly as before."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point: load JFLEG, run the existing pipeline, evaluate, save results."""
    args = parse_args()

    predictions_json_path = os.path.join(args.output_dir, "jfleg_predictions.json")
    predictions_csv_path = os.path.join(args.output_dir, "jfleg_predictions.csv")
    summary_json_path = os.path.join(args.output_dir, "jfleg_evaluation_summary.json")
    checkpoint_path = os.path.join(
        args.output_dir, "jfleg_predictions.checkpoint.jsonl"
    )

    print("=" * 70)
    print("STAGE 8: Independent Evaluation on JFLEG")
    print("=" * 70)
    print(f"JFLEG dataset:          {JFLEG_DATASET_NAME}")
    print(f"JFLEG split:            {args.split}")
    print(f"Retrieval knowledge base (unchanged): {config.DATASET_NAME}")
    print(f"ChromaDB collection (unchanged):      {config.CHROMA_COLLECTION_NAME}")
    print(f"OpenRouter model:       {args.model}")
    print(f"top_k:                  {args.top_k}")
    if args.limit is not None:
        print(f"Limit:                  {args.limit} examples")
    print("-" * 70)

    try:
        examples = load_jfleg_dataset(args.split, limit=args.limit)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(examples)} JFLEG examples from split '{args.split}'.")
    print("-" * 70)

    if args.resume:
        checkpoint_rows = load_checkpoint_rows(checkpoint_path)
        completed_map = build_completed_map(checkpoint_rows)
        print(
            f"Resuming: {len(completed_map)} example(s) already have a "
            f"successful result and will be skipped "
            f"({len(checkpoint_rows) - len(completed_map)} prior failed/"
            "duplicate attempt(s) found and will be retried if still pending)."
        )
    else:
        reset_checkpoint_file(checkpoint_path)
        completed_map = {}
    print("-" * 70)

    api_key = _prompting_module.load_environment()
    print(f"OPENROUTER_API_KEY set: {bool(api_key)}")
    if not api_key:
        print(
            "ERROR: OPENROUTER_API_KEY is not set. Set it as an environment "
            "variable, in a local .env file, or via Streamlit secrets before "
            "running this evaluation - the existing pipeline needs it to "
            "call OpenRouter for each JFLEG sentence.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from nltk.translate.gleu_score import sentence_gleu  # noqa: F401
    except ImportError:
        print(
            "NOTE: 'nltk' is not installed, so GLEU will be skipped and "
            "reported as unavailable in the summary. Install it with: "
            "pip install nltk"
        )
    print("-" * 70)

    rows = evaluate(
        examples,
        api_key=api_key,
        model=args.model,
        top_k=args.top_k,
        sleep_seconds=args.sleep,
        checkpoint_path=checkpoint_path,
        completed_map=completed_map,
    )

    summary = aggregate_results(rows)

    run_config = {
        "jfleg_dataset": JFLEG_DATASET_NAME,
        "jfleg_split": args.split,
        "num_examples_requested": args.limit,
        "retrieval_knowledge_base": config.DATASET_NAME,
        "chroma_collection": config.CHROMA_COLLECTION_NAME,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "openrouter_model": args.model,
        "top_k": args.top_k,
    }

    save_predictions_json(rows, predictions_json_path)
    save_predictions_csv(rows, predictions_csv_path)
    save_summary_json(summary, run_config, summary_json_path)

    print("-" * 70)
    print("STAGE 8 complete.")
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total examples:          {summary['total_examples']}")
    print(f"Successful predictions:  {summary['successful_predictions']}")
    print(f"Failed predictions:      {summary['failed_predictions']}")
    if summary["exact_match_rate"] is not None:
        print(f"Exact match rate:        {summary['exact_match_rate']:.4f}")
    if summary["mean_gleu"] is not None:
        print(f"Mean GLEU (NLTK):        {summary['mean_gleu']:.4f}")
    else:
        print("Mean GLEU (NLTK):        n/a (nltk not installed, or no successful predictions)")
    if summary["mean_token_f1"] is not None:
        print(f"Mean token F1:           {summary['mean_token_f1']:.4f}")
    if summary["mean_retrieved_examples"] is not None:
        print(f"Mean retrieved examples: {summary['mean_retrieved_examples']:.2f}")
    print("-" * 70)
    print(f"Predictions (JSON):      {predictions_json_path}")
    print(f"Predictions (CSV):       {predictions_csv_path}")
    print(f"Summary (JSON):          {summary_json_path}")
    print(f"Checkpoint (JSONL):      {checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
