"""Tests for frozen model fitting and external scoring (pre-freeze safety gates)."""
import json
from pathlib import Path
import pytest
import pandas as pd

from experiments.external_transport.scripts import c_01_frozen_scoring as c01
from experiments.external_transport.tests.fixtures import fake_features


def test_sha256_mismatch_fails_loudly(tmp_path):
    """Integrity gate: mismatch between actual and expected SHA-256 must halt scoring."""
    fake_csv = tmp_path / "fake_ext.csv"
    fake_features.make_fake_external_table(n=10, output_path=fake_csv)

    # Calling main without allow_sha_mismatch must raise RuntimeError
    with pytest.raises(RuntimeError, match="SHA-256 verification FAILED"):
        c01.main(["--external-csv", str(fake_csv), "--run-id", "test_fake"])


def test_label_column_fails_loudly(tmp_path):
    """Integrity gate: any presence of a label column must halt scoring."""
    fake_csv = tmp_path / "fake_ext.csv"
    df = fake_features.make_fake_external_table(n=10)
    df["label"] = 0
    df.to_csv(fake_csv, index=False)

    with pytest.raises(RuntimeError, match="contains 'label' column"):
        c01.main(["--external-csv", str(fake_csv), "--run-id", "test_fake", "--allow-sha-mismatch"])


def test_frozen_scoring_fake_pipeline(tmp_path):
    """Verify frozen scoring fits M1-fb and M1, filters qc_pass, and scores properly."""
    fake_csv = tmp_path / "fake_ext.csv"
    df_fake = fake_features.make_fake_external_table(n=15, seed=123, output_path=fake_csv)

    n_pass = int(df_fake["qc_pass"].sum())
    assert n_pass < 15, "Test requires at least one failed QC row"

    c01.main(["--external-csv", str(fake_csv), "--run-id", "test_scoring", "--allow-sha-mismatch"])

    out_dir = Path("experiments/external_transport/outputs/test_scoring/frozen_scores")
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
