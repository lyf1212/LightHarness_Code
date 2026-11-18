"""Structured visual advice and derived, host-owned per-object history."""
from __future__ import annotations

import copy
import math


ASPECTS = ("region", "material", "color")
DIFFERENCE_LEVELS = ("significant", "minor", "uncertain")
IMPLEMENTATION = ("implemented", "partial", "not_implemented", "unknown")
OUTCOMES = ("improved", "unchanged", "regressed", "unknown")


class ReviewValidationError(ValueError):
    pass


def object_memory(objects, history):
    """Rebuild from completed entries only; never infer structured advice from prose."""
    memory = {oid: [] for oid in objects}
    for entry in history:
        review = entry["review"]
        rows = review.get("object_reviews", {})
        for oid in memory:
            record = {"round": entry["round"], "structured": oid in rows}
            if oid in rows:
                record.update(review=copy.deepcopy(rows[oid]), history_feedback=copy.deepcopy([
                    item for item in review.get("history_feedback", []) if item["object_id"] == oid]))
            else:
                record["reason"] = "no_structured_record; consult original history"
            memory[oid].append(record)
    return memory


def _require(condition, message):
    if not condition:
        raise ReviewValidationError(message)


def _text(value, location):
    _require(isinstance(value, str) and bool(value.strip()), f"{location}: expected nonempty text")


def _advice_aspects(row):
    """Read selective advice and the previous mandatory three-aspect format."""
    if "difference_level" in row:
        return [aspect for aspect in ASPECTS if aspect in row] if row["difference_level"] == "significant" else []
    return [aspect for aspect in ASPECTS if row.get(aspect, {}).get("status") == "needs_change"]


def validate_review(review, objects, history):
    """Validate coverage and traceable advice, not the truth of visual judgments."""
    _require(isinstance(review, dict), "Review must be a JSON object")
    rows = review.get("object_reviews")
    _require(isinstance(rows, dict), "object_reviews must be an object keyed by stable object ID")
    expected, supplied = set(objects), set(rows)
    _require(expected == supplied,
             f"object_reviews coverage: missing={sorted(expected-supplied)}, unknown={sorted(supplied-expected)}")
    for oid, row in rows.items():
        _require(isinstance(row, dict), f"{oid}: expected object assessment")
        _require(row.get("difference_level") in DIFFERENCE_LEVELS,
                 f"{oid}: difference_level must be one of {DIFFERENCE_LEVELS}")
        _text(row.get("observation"), f"{oid}.observation")
        aspects = set(row) & set(ASPECTS)
        _require(not (set(row) - {"difference_level", "observation", *ASPECTS}),
                 f"{oid}: unexpected assessment fields")
        if row["difference_level"] == "significant":
            _require(bool(aspects), f"{oid}: significant difference requires at least one relevant diagnosis")
        else:
            _require(not aspects, f"{oid}: minor/uncertain differences need only a brief observation, no diagnosis")
        for aspect in aspects:
            item = row[aspect]
            where = f"object_reviews.{oid}.{aspect}"
            _require(isinstance(item, dict), f"{where}: expected object")
            for field in ("difference", "suggestion", "evidence"):
                _text(item.get(field), f"{where}.{field}")
            confidence = item.get("confidence")
            _require(type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1,
                     f"{where}.confidence: expected finite number in [0,1]")

    # The latest structured advice remains relevant even across legacy gaps.
    sources, required = {}, set()
    for oid, records in object_memory(objects, history).items():
        latest = None
        for record in records:
            if record["structured"]:
                latest = record
                for aspect in ASPECTS:
                    if aspect in record["review"]:
                        sources[(oid, record["round"], aspect)] = record["review"][aspect]
        if latest:
            required.update((oid, latest["round"], aspect) for aspect in _advice_aspects(latest["review"]))

    feedback = review.get("history_feedback")
    _require(isinstance(feedback, list), "history_feedback must be an array (empty without structured history)")
    seen = set()
    for item in feedback:
        _require(isinstance(item, dict), "history_feedback entries must be objects")
        oid, source_round, aspect = item.get("object_id"), item.get("source_round"), item.get("aspect")
        _require(isinstance(oid, str) and oid in expected, "history_feedback: unknown object_id")
        _require(type(source_round) is int and isinstance(aspect, str) and aspect in ASPECTS,
                 "history_feedback: invalid source_round or aspect")
        key = (oid, source_round, aspect)
        _require(key in sources, f"history_feedback: no structured source advice for {key}")
        _require(key not in seen, f"history_feedback: duplicate reference {key}")
        seen.add(key)
        _require(item.get("implementation") in IMPLEMENTATION,
                 f"history_feedback {key}: invalid implementation; use {IMPLEMENTATION}")
        _require(item.get("outcome") in OUTCOMES, f"history_feedback {key}: invalid outcome; use {OUTCOMES}")
        _text(item.get("evidence"), f"history_feedback {key}.evidence")
    _require(required <= seen, f"history_feedback: assess latest modification advice {sorted(required-seen)}")
    return review
