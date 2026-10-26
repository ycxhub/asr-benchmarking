from asrbench.metrics.normalize import normalize
from asrbench.metrics.stats import bootstrap_ci, paired_bootstrap, pooled_wer


def test_english_lakh_and_informal():
    assert normalize("5 lakh rupees", "en-IN") == normalize("500000 rupees", "en-IN")
    assert "ok" in normalize("Okay, gonna transfer.", "en-US").split()


def test_hindi_lexicon_and_nukta():
    folded = normalize("account", "hi", script_normalize=True)
    assert folded == "अकाउंट"
    raw = normalize("फ़ोन", "hi", script_normalize=False)
    assert "फोन" in raw.split()


def test_failed_clip_is_all_deletions():
    from asrbench.metrics.wer import score_pair

    score = score_pair("one two three", "", "en-US", failed=True)
    assert score.wer == 1
    assert score.deletions == score.ref_words


def test_bootstrap_ci_contains_estimate():
    rows = [{"item_id": str(i), "speaker": "a" if i < 4 else "b", "substitutions": i % 2, "deletions": 0, "insertions": 0, "ref_words": 10} for i in range(8)]
    ci = bootstrap_ci(rows, n_boot=200)
    assert ci["low"] <= ci["estimate"] <= ci["high"]
    assert pooled_wer(rows) == ci["estimate"]


def test_paired_bootstrap_detects_a_gap():
    base = [{"item_id": str(i), "substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 10} for i in range(12)]
    other = [{"item_id": str(i), "substitutions": 2, "deletions": 0, "insertions": 0, "ref_words": 10} for i in range(12)]
    result = paired_bootstrap(base, other, n_boot=200)
    assert result["diff"] == 0.2
    assert result["low"] > 0
