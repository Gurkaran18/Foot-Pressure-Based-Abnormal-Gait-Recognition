"""Tests for frozen model fitting and external scoring (pre-freeze safety gates)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.external_transport.scripts import c_01_frozen_scoring as c01
from experiments.external_transport.tests.fixtures import fake_features


def fake_train(path, n=165, seed=7):
    """Fake gaitpdb subject_v2 table (one row per subject, 0/1 label, planted signal)."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(93, int), np.zeros(n - 93, int)]
    df = pd.DataFrame({"subject_id": [f"S{i:03d}" for i in range(n)], "study": rng.choice(["Ga", "Ju", "Si"], n),
                       "label": y, "qc_pass": True,
                       "stride_time_mean": rng.normal(1.1 + 0.05 * y, 0.1), "stride_time_cv": rng.normal(2 + y, 0.8),
                       "swing_time_cv": rng.normal(4 + y, 1.5), "stance_pct": rng.normal(62, 2, n),
                       "asym_swing": rng.exponential(3, n)})
    df.to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def tmp_outputs(tmp_path, monkeypatch):
    """Keep test outputs out of the real outputs/ folder."""
    monkeypatch.setattr(c01.paths, "outputs_root", lambda run_id: tmp_path / "outputs" / run_id)
    return tmp_path / "outputs"


def test_sha256_mismatch_fails_loudly(tmp_path):
    """Integrity gate: mismatch between actual and expected SHA-256 must halt scoring."""
    fake_csv = tmp_path / "fake_ext.csv"
    fake_features.make_fake_external_table(n=10, output_path=fake_csv)

    # Calling main without allow_sha_mismatch must raise RuntimeError
    with pytest.raises(RuntimeError, match="SHA-256 verification FAILED"):
        c01.main(["--external-csv", str(fake_csv), "--run-id", "test_fake",
                  "--train-csv", str(fake_train(tmp_path / "train.csv"))])


def test_label_column_fails_loudly(tmp_path):
    """Integrity gate: any presence of a label column must halt scoring."""
    fake_csv = tmp_path / "fake_ext.csv"
    df = fake_features.make_fake_external_table(n=10)
    df["label"] = 0
    df.to_csv(fake_csv, index=False)

    with pytest.raises(RuntimeError, match="contains 'label' column"):
        c01.main(["--external-csv", str(fake_csv), "--run-id", "test_fake", "--allow-sha-mismatch",
                  "--train-csv", str(fake_train(tmp_path / "train.csv"))])


def test_train_table_explicit_and_checked(tmp_path):
    """Pre-freeze fix 5: training table must be given, must be the 165-subject cohort, SHA can be pinned."""
    fake_csv = tmp_path / "fake_ext.csv"
    fake_features.make_fake_external_table(n=10, output_path=fake_csv)
    base = ["--external-csv", str(fake_csv), "--run-id", "t", "--allow-sha-mismatch"]
    with pytest.raises(SystemExit):
        c01.main(base)                                                    # --train-csv is required
    with pytest.raises(ValueError, match="expected 165"):
        c01.main(base + ["--train-csv", str(fake_train(tmp_path / "small.csv", n=150))])
    with pytest.raises(RuntimeError, match="Training table SHA-256 mismatch"):
        c01.main(base + ["--train-csv", str(fake_train(tmp_path / "t.csv")), "--expected-train-sha256", "0" * 64])


def test_frozen_scoring_fake_pipeline(tmp_path, tmp_outputs):
    """Verify frozen scoring fits M1-fb and M1, filters qc_pass, and scores properly."""
    fake_csv = tmp_path / "fake_ext.csv"
    df_fake = fake_features.make_fake_external_table(n=15, seed=123, output_path=fake_csv)

    n_pass = int(df_fake["qc_pass"].sum())
    assert n_pass < 15, "Test requires at least one failed QC row"

    train = fake_train(tmp_path / "train.csv")
    c01.main(["--external-csv", str(fake_csv), "--run-id", "test_scoring", "--allow-sha-mismatch",
              "--train-csv", str(train)])

    out_dir = tmp_outputs / "test_scoring" / "frozen_scores"
    scores_file = out_dir / "external_scores.csv"
    assert scores_file.exists(), "Scores file was not written"

    scores = pd.read_csv(scores_file)
    assert len(scores) == n_pass, "Must score ONLY qc_pass == True rows"
    assert "label" not in scores.columns

    # Probabilities in valid bounds
    assert (scores["m1_fb_score"] >= 0.0).all() and (scores["m1_fb_score"] <= 1.0).all()
    assert (scores["m1_score"] >= 0.0).all() and (scores["m1_score"] <= 1.0).all()

    # Manifest verification
    manifest_file = out_dir / "frozen_manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r") as f:
        meta = json.load(f)
    assert meta["n_external_scored"] == n_pass
    assert meta["primary_model"]["model_id"] == "M1-fb"
    assert meta["secondary_model"]["model_id"] == "M1"
    assert meta["train_sha256"] == c01.compute_sha256(train) and meta["primary_model"]["n_train_subjects"] == 165
