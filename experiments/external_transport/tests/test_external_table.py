"""B5 external table: subject_v2 column layout, no label, all codes kept, covariates attached."""
import pandas as pd
import pytest

from experiments.external_transport.scripts import b_05_external_table as b05


def test_build_layout_and_no_label():
    walks = pd.DataFrame({"code": ["WG002", "WG001"], "task": "SelfPace", "has_insole": [True, False],
                          "stride_time_mean": [1.1, None], "qc_pass": [True, False], "qc_flags": ["", "no_insole"],
                          "n_valid_L": [17, None], "n_valid_R": [18, None]})
    cov = pd.DataFrame({"code": ["WG001", "WG002"], "speed_selfpace_mps": [0.9, 1.1], "age_y": [70, 65],
                        "sex": ["F", "M"]})
    t = b05.build(walks, cov)
    assert list(t.columns) == b05.COLUMNS and "label" not in t.columns
    assert t.subject_id.tolist() == ["WG001", "WG002"] and (t.study == "WG").all()
    assert t.loc[1, "speed_01_mps"] == 1.1 and t.loc[1, "source_recordings"] == "WG002_SelfPace.csv"
    assert t.qc_pass.tolist() == [False, True]
    with pytest.raises(ValueError):
        b05.build(walks.assign(label=0), cov)
