import copy
import json

import pytest
from PIL import Image

from lightagent import pipeline
from lightagent.demo import CODE, demo_review, inventory
from lightagent.io import read, write
from lightagent.llm import ModelOutputError, parse_json
from lightagent.object_review import ReviewValidationError, object_memory, validate_review
from lightagent.state import initialize
from lightagent.surface_geometry import CONTRACT


def entry(index=0):
    return {"round": index, "review": demo_review(index)}


def test_complete_reviews_and_memory_are_independent_of_source():
    objects = initialize(inventory())["objects"]
    history = [entry()]
    assert validate_review(demo_review(0), objects, [])
    assert validate_review(demo_review(1), objects, history)
    memory = object_memory(objects, history)
    assert set(memory) == set(objects)
    assert memory["obj_cube"][0]["review"]["difference_level"] == "significant"
    memory["obj_cube"][0]["review"]["region"]["suggestion"] = "modified"
    assert history == [entry()]
    later = object_memory(objects, [entry(), entry(1)])
    assert later["obj_cube"][1]["history_feedback"][0]["outcome"] == "improved"


@pytest.mark.parametrize("case", ["missing_object", "unknown_object", "missing_diagnosis", "difference_level",
                                  "confidence_bool", "confidence_nan", "confidence_range", "empty_evidence",
                                  "empty_observation", "missing_feedback", "future_round", "unknown_history_object",
                                  "bad_history_aspect", "duplicate_feedback", "implementation", "outcome"])
def test_invalid_review_contract(case):
    review = demo_review(0)
    review["history_feedback"] = demo_review(1)["history_feedback"]
    row = review["object_reviews"]["obj_cube"]
    if case == "missing_object":
        del review["object_reviews"]["obj_floor"]
    elif case == "unknown_object":
        review["object_reviews"]["unknown"] = copy.deepcopy(row)
    elif case == "missing_diagnosis":
        del row["region"]
        del row["color"]
    elif case == "difference_level":
        row["difference_level"] = "perfect"
    elif case.startswith("confidence_"):
        row["color"]["confidence"] = {"confidence_bool": True, "confidence_nan": float("nan"),
                                       "confidence_range": 1.01}[case]
    elif case == "empty_evidence":
        row["color"]["evidence"] = " "
    elif case == "empty_observation":
        row["observation"] = ""
    elif case == "missing_feedback":
        review["history_feedback"] = []
    elif case == "future_round":
        review["history_feedback"][0]["source_round"] = 1
    elif case == "unknown_history_object":
        review["history_feedback"][0]["object_id"] = "unknown"
    elif case == "bad_history_aspect":
        review["history_feedback"][0]["aspect"] = "lighting"
    elif case == "duplicate_feedback":
        review["history_feedback"].append(copy.deepcopy(review["history_feedback"][0]))
    else:
        review["history_feedback"][0][case] = "invalid"
    with pytest.raises(ReviewValidationError):
        validate_review(review, initialize(inventory())["objects"], [entry()])


def test_uncertainty_and_legacy_history():
    objects = initialize(inventory())["objects"]
    legacy = {"round": 0, "review": {"summary": "Old unstructured suggestions"}}
    review = demo_review(0)
    review["object_reviews"]["obj_cube"] = {
        "difference_level": "uncertain", "observation": "Object occluded and no reference mask"}
    validate_review(review, objects, [legacy])
    memory = object_memory(objects, [legacy])
    assert all(rows[0]["structured"] is False for rows in memory.values())
    assert "no_structured_record" in memory["obj_cube"][0]["reason"]
    review["history_feedback"] = demo_review(1)["history_feedback"]
    with pytest.raises(ReviewValidationError, match="no structured source"):
        validate_review(review, objects, [legacy])
    # First-round feedback cannot reference nonexistent earlier advice.
    with pytest.raises(ReviewValidationError, match="no structured source"):
        validate_review(review, objects, [])


def test_feedback_distinguishes_nonimplementation_and_unknown_outcome():
    review = demo_review(1)
    review["history_feedback"][0].update(implementation="not_implemented", outcome="unknown",
                                        evidence="Requested geometry edit absent; appearance confounded by lighting")
    validate_review(review, initialize(inventory())["objects"], [entry()])


def test_only_noticeable_differences_need_selective_advice():
    objects = initialize(inventory())["objects"]
    review = demo_review(1)
    review["history_feedback"] = []
    # All objects may be close enough; no region/material/color text is required.
    validate_review(review, objects, [])
    validate_review(review, objects, [entry(), entry(1)])
    review["object_reviews"]["obj_floor"] = {
        "difference_level": "uncertain", "observation": "Mask unavailable and too occluded to judge"}
    review["object_reviews"]["obj_cube"] = {
        "difference_level": "significant", "observation": "Only silhouette needs correction",
        "region": copy.deepcopy(demo_review(0)["object_reviews"]["obj_cube"]["region"])}
    validate_review(review, objects, [])
    # Similar objects should not be expanded into mandatory aspect diagnoses.
    review["object_reviews"]["obj_wall"]["color"] = demo_review(0)["object_reviews"]["obj_wall"]["color"]
    with pytest.raises(ReviewValidationError, match="no diagnosis"):
        validate_review(review, objects, [])


def test_prior_three_aspect_history_remains_usable():
    objects = initialize(inventory())["objects"]
    old = entry()
    for row in old["review"]["object_reviews"].values():
        del row["difference_level"], row["observation"]
        for aspect in ("region", "material", "color"):
            if aspect in row:
                row[aspect]["status"] = "needs_change"
            else:
                row[aspect] = {"status": "keep", "difference": "Matches", "suggestion": "Preserve",
                               "evidence": "Known fixture", "confidence": 1.}
    before = copy.deepcopy(old)
    validate_review(demo_review(1), objects, [old])
    memory = object_memory(objects, [old])
    assert memory["obj_cube"][0]["review"]["region"]["status"] == "needs_change"
    assert old == before


def prepare_mock_run(path):
    path.mkdir()
    write(path / "config.json", {"llm": {}, "rounds": 2, "max_repairs": 1})
    write(path / "initial_state.json", initialize(inventory()))
    evidence = path / "evidence"
    write(evidence / "manifest.json", {"normal_contract": CONTRACT, "files": {}})
    write(evidence / "surface_geometry.json", {"contract": CONTRACT})
    write(evidence / "inventory.json", inventory())
    write(evidence / "object_priors.json", {})
    write(evidence / "masks/index.json", {})
    return path


@pytest.fixture
def harness(monkeypatch):
    calls, renders = [], []
    reviews = []

    class FakeClient:
        def __init__(self, config):
            pass

        def ask(self, folder, system, context, images=(), **kwargs):
            folder.mkdir(parents=True, exist_ok=False)
            write(folder / "context.json", context)
            calls.append({"stage": "build" if folder.name == "build" else "review",
                          "context": copy.deepcopy(context), "images": list(images)})
            if folder.name == "build":
                answer = {"rationale": "fixture", "scene_code": CODE + f"\n# round {context['round']}\n",
                          "state_patch": {"base_revision": context["state"]["revision"], "reason": "fixture"}}
            else:
                answer = reviews.pop(0) if reviews else demo_review(context["round"])
                if isinstance(answer, Exception):
                    raise answer
            write(folder / "parsed.json", answer)
            return answer

    def render(code, state, evidence, out, config):
        renders.append(str(out))
        write(out / "metadata.json", {})
        Image.new("RGB", (2, 2)).save(out / "rgb.png")

    def evaluate(evidence, rendered, out, config):
        scores = {"selection_loss": 0.5, "metrics": {}}
        write(out / "scores.json", scores)
        Image.new("RGB", (2, 2)).save(out / "comparison.png")
        return scores

    monkeypatch.setattr(pipeline, "Client", FakeClient)
    monkeypatch.setattr(pipeline, "render", render)
    monkeypatch.setattr(pipeline, "evaluate", evaluate)
    monkeypatch.setattr(pipeline, "export_final", lambda *args: None)
    monkeypatch.setattr(pipeline, "evidence_images", lambda evidence: [])
    return calls, renders, reviews


def test_two_round_context_and_resume_equivalence(tmp_path, harness):
    calls, renders, _ = harness
    continuous = prepare_mock_run(tmp_path / "continuous")
    pipeline.iterate(continuous, 2)
    uninterrupted_contexts = copy.deepcopy(calls)
    second_build, second_review = calls[2], calls[3]
    assert second_build["context"]["object_memory"] == second_review["context"]["object_memory"]
    assert second_review["context"]["previous_code"].endswith("# round 0\n")
    assert second_review["context"]["scene_code"].endswith("# round 1\n")
    assert second_build["context"]["object_memory"]["obj_cube"][0]["review"] == demo_review(0)["object_reviews"]["obj_cube"]
    assert {label for label, _ in second_review["images"]} >= {"previous render", "current render"}
    assert "object_memory" not in read(continuous / "state.json")
    calls.clear()
    resumed = prepare_mock_run(tmp_path / "resumed")
    pipeline.iterate(resumed, 1)
    before = (resumed / "round_000/history_entry.json").read_bytes()
    # Root history is a derived cache, not the resume source of truth.
    write(resumed / "history.json", [])
    pipeline.iterate(resumed, 2)
    assert before == (resumed / "round_000/history_entry.json").read_bytes()
    assert [c["context"] for c in calls] == [c["context"] for c in uninterrupted_contexts]
    assert read(resumed / "history.json") == read(continuous / "history.json")
    assert len(renders) == 4


def test_review_repair_and_exhaustion_resume_without_render(tmp_path, harness):
    calls, renders, reviews = harness
    run = prepare_mock_run(tmp_path / "run")
    reviews.extend([{"summary": "missing objects"}, demo_review(0)])
    pipeline.iterate(run, 1)
    assert len(renders) == 1
    assert (run / "round_000/review_000/failure.json").exists()
    assert calls[-1]["context"]["failed_reviews"][0]["review"] == "review_000"
    reviews.extend([{"object_reviews": {}}, {"object_reviews": {}}])
    with pytest.raises(ReviewValidationError):
        pipeline.iterate(run, 2)
    assert len(renders) == 2
    assert not (run / "round_001/done.json").exists()
    assert not (run / "round_001/history_entry.json").exists()
    assert read(run / "state.json")["revision"] == 1
    assert len(read(run / "history.json")) == 1
    assert read(run / "progress.json")["stage"] == "review"
    assert (run / "round_001/accepted.json").exists()
    pipeline.iterate(run, 2)
    assert len(renders) == 2
    assert len([c for c in calls if c["stage"] == "build"]) == 2
    assert read(run / "round_001/history_entry.json")["review_archive"] == "review_002"


@pytest.mark.parametrize("text", ['{', '[]', '{"confidence": NaN}', '{"confidence": 1e999}'])
def test_model_output_parse_errors_are_distinct_from_transport_errors(text):
    with pytest.raises(ModelOutputError):
        parse_json(text)


@pytest.mark.parametrize("failure,attempts", [(ModelOutputError("bad model JSON"), 2),
                                             (json.JSONDecodeError("bad transport JSON", "{", 1), 1),
                                             (RuntimeError("transport failure"), 1)])
def test_parse_retry_but_no_transport_retry(tmp_path, harness, failure, attempts):
    calls, renders, reviews = harness
    run = prepare_mock_run(tmp_path / "run")
    reviews.append(failure)
    if not isinstance(failure, ModelOutputError):
        with pytest.raises(type(failure)):
            pipeline.iterate(run, 1)
        assert not (run / "round_000/done.json").exists()
    else:
        pipeline.iterate(run, 1)
    assert len([c for c in calls if c["stage"] == "review"]) == attempts
    assert len(renders) == 1


def test_legacy_completed_review_is_unchanged_on_resume(tmp_path, harness):
    calls, _, reviews = harness
    run = prepare_mock_run(tmp_path / "run")
    pipeline.iterate(run, 1)
    path = run / "round_000/history_entry.json"
    old = read(path)
    old["review"] = {"summary": "legacy prose only", "geometry_changes": ["Move cube left"]}
    write(path, old)
    original = path.read_bytes()
    next_review = demo_review(1)
    next_review["history_feedback"] = []
    reviews.append(next_review)
    pipeline.iterate(run, 2)
    assert path.read_bytes() == original
    for call in calls[-2:]:
        memory = call["context"]["object_memory"]
        assert all(not records[0]["structured"] for records in memory.values())
        history = call["context"].get("history", call["context"].get("previous_history"))
        assert history[0]["review"] == old["review"]
