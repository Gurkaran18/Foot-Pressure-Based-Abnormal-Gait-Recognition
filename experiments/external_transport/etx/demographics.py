"""Robust demographics parser + HTML cross-check. NEVER fills values; UPDRS/H&Y description-only.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md §5, §11 (T8).
"""
from html.parser import HTMLParser
from pathlib import Path
import re
import numpy as np
import pandas as pd


class DemoHTMLParser(HTMLParser):
    """Parse demographics.html to extract table rows."""
    def __init__(self):
        super().__init__()
        self.in_tr = False
        self.in_td = False
        self.cur_row = []
        self.rows = []
        self.cur_val = ""

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.in_tr = True
            self.cur_row = []
        elif tag == "td" and self.in_tr:
            self.in_td = True
            self.cur_val = ""
            attr_dict = dict(attrs)
            if "sdval" in attr_dict:
                self.cur_val = attr_dict["sdval"]

    def handle_endtag(self, tag):
        if tag == "td" and self.in_td:
            self.in_td = False
            self.cur_row.append(self.cur_val.strip())
        elif tag == "tr" and self.in_tr:
            self.in_tr = False
            if self.cur_row:
                self.rows.append(self.cur_row)

    def handle_data(self, data):
        if self.in_td and not self.cur_val:
            self.cur_val = data.strip()


def _to_float(val):
    if val is None:
        return np.nan
    s = str(val).strip()
    if s in ("", "-", "NaN", "nan", "None", "."):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_demographics(root) -> pd.DataFrame:
    """Columns per spec §5 demographics/clean.csv. Gender 1->M, 2->F; Ju height cm->m.
    
    Cross-checks against demographics.html if present in root.
    Never imputes or synthesizes missing values.
    """
    root = Path(root)
    txt_path = root / "demographics.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"demographics.txt not found under {root}")

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    header_tokens = [t.strip() for t in raw_lines[0].strip().split("\t") if t.strip()]
    col_idx = {name: i for i, name in enumerate(header_tokens)}

    # Parse HTML for cross-check if present
    html_data = {}
    html_path = root / "demographics.html"
    if html_path.exists():
        parser = DemoHTMLParser()
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            parser.feed(f.read())
        if len(parser.rows) > 1:
            html_header = [h.strip() for h in parser.rows[0]]
            h_col_idx = {name: i for i, name in enumerate(html_header)}
            for h_row in parser.rows[1:]:
                if h_row and len(h_row) > 0 and h_row[0]:
                    sid = h_row[0].strip()
                    html_data[sid] = {
                        html_header[i]: h_row[i] for i in range(min(len(h_row), len(html_header)))
                    }

    clean_rows = []
    for line_idx, line in enumerate(raw_lines[1:], start=2):
        s_line = line.strip()
        if not s_line:
            continue
        tokens = [t.strip() for t in s_line.split("\t")]
        if not tokens or not tokens[0]:
            continue

        subj_id = tokens[col_idx["ID"]]
        study = tokens[col_idx["Study"]]
        group_raw = tokens[col_idx["Group"]]
        label = 1 if group_raw in ("1", 1, "Pt") else 0

        # Gender: 1 -> M, 2 -> F
        gender_raw = tokens[col_idx.get("Gender", 4)]
        if gender_raw in ("1", 1, "male", "M"):
            sex = "M"
        elif gender_raw in ("2", 2, "female", "F"):
            sex = "F"
        else:
            sex = None

        age_y = _to_float(tokens[col_idx["Age"]]) if "Age" in col_idx else np.nan
        raw_height = _to_float(tokens[col_idx["Height"]]) if "Height" in col_idx else np.nan
        # Ju conversion: cm to m if height > 3.0
        if not np.isnan(raw_height) and raw_height > 3.0:
            height_m = raw_height / 100.0
        else:
            height_m = raw_height

        weight_kg = _to_float(tokens[col_idx["Weight"]]) if "Weight" in col_idx else np.nan
        speed_01_mps = _to_float(tokens[col_idx["Speed_01"]]) if "Speed_01" in col_idx else np.nan
        speed_method_status = "verified_stopwatch_mid10m" if study == "Si" else "unverified"
        tuag_s = _to_float(tokens[col_idx["TUAG"]]) if "TUAG" in col_idx else np.nan
        hy_desc_only = _to_float(tokens[col_idx["HoehnYahr"]]) if "HoehnYahr" in col_idx else np.nan
        updrs_desc_only = _to_float(tokens[col_idx["UPDRS"]]) if "UPDRS" in col_idx else np.nan

        # HTML Cross-Check
        html_match = True
        if subj_id in html_data:
            h_entry = html_data[subj_id]
            h_age = _to_float(h_entry.get("Age"))
            h_wt = _to_float(h_entry.get("Weight (kg)"))
            h_sp = _to_float(h_entry.get("Speed_01 (m/sec)"))
            h_tuag = _to_float(h_entry.get("TUAG"))
            h_raw_ht = _to_float(h_entry.get("Height (meters)"))
            h_ht = h_raw_ht / 100.0 if not np.isnan(h_raw_ht) and h_raw_ht > 3.0 else h_raw_ht

            for val_txt, val_html in [
                (age_y, h_age),
                (height_m, h_ht),
                (weight_kg, h_wt),
                (speed_01_mps, h_sp),
                (tuag_s, h_tuag),
            ]:
                if np.isnan(val_txt) and np.isnan(val_html):
                    continue
                elif np.isnan(val_txt) != np.isnan(val_html):
                    html_match = False
                    break
                elif abs(val_txt - val_html) > 0.01:
                    html_match = False
                    break

        clean_rows.append({
            "subject_id": subj_id,
            "study": study,
            "label": label,
            "sex": sex,
            "age_y": age_y,
            "height_m": height_m,
            "weight_kg": weight_kg,
            "speed_01_mps": speed_01_mps,
            "speed_method_status": speed_method_status,
            "tuag_s": tuag_s,
            "hy_desc_only": hy_desc_only,
            "updrs_desc_only": updrs_desc_only,
            "source_txt_row": line_idx,
            "html_match": html_match,
        })

    return pd.DataFrame(clean_rows)


def build_audit(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Build audit table showing missingness by variable, study, and group."""
    vars_to_audit = [
        ("sex", "Categorical (M/F)"),
        ("age_y", "Years"),
        ("height_m", "Meters (Ju converted from cm)"),
        ("weight_kg", "Kilograms"),
        ("speed_01_mps", "m/s (usual walk 01)"),
        ("tuag_s", "Timed Up and Go (s)"),
        ("hy_desc_only", "Hoehn & Yahr scale (PD descriptive only)"),
        ("updrs_desc_only", "UPDRS total score (PD descriptive only)"),
    ]

    audit_rows = []
    # Breakdown by study and group
    groups = clean_df.groupby(["study", "label"])
    for (study, label), gdf in groups:
        grp_name = "PD" if label == 1 else "Control"
        n = len(gdf)
        for var_name, unit_note in vars_to_audit:
            if var_name == "sex":
                n_miss = int(gdf[var_name].isna().sum())
            else:
                n_miss = int(gdf[var_name].isna().sum())
            pct_miss = 100.0 * n_miss / n if n > 0 else 0.0
            audit_rows.append({
                "variable": var_name,
                "study": study,
                "group": grp_name,
                "n": n,
                "n_missing": n_miss,
                "pct_missing": round(pct_miss, 2),
                "unit_note": unit_note,
            })

    # Overall summary per variable
    n_total = len(clean_df)
    for var_name, unit_note in vars_to_audit:
        n_miss = int(clean_df[var_name].isna().sum())
        pct_miss = 100.0 * n_miss / n_total if n_total > 0 else 0.0
        audit_rows.append({
            "variable": var_name,
            "study": "All",
            "group": "All",
            "n": n_total,
            "n_missing": n_miss,
            "pct_missing": round(pct_miss, 2),
            "unit_note": unit_note,
        })

    return pd.DataFrame(audit_rows)
