"""D2: cohort table + report figures (aggregate only) for the write-up. Post-freeze, descriptive.

Owner: Lead. Reads the frozen/sensitivity outputs and the two cohorts' demographics; writes
outputs/<run_id>/writeup/{cohort_table.csv, fig1_auc_forest.png, fig2_age_by_group.png, fig3_external_roc.png}.
Figures and the table contain group-level summaries only (no individual rows), so they can be copied into
research/writeup/figures/. Internal gaitpdb numbers that only exist in the Teammate's a2_06 run are taken
from PROJECT_LOG.md (2026-11-22 Teammate Day 2 entry) and marked as such.

    GAITPDB_ROOT=... python -m experiments.external_transport.scripts.d_02_writeup_figures --run-id d_writeup
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from experiments.external_transport.etx import demographics, guards, paths  # noqa: E402
from experiments.external_transport.etx.weargait_covariates import norm_id  # noqa: E402

O = paths.OUTPUTS_DIR
CO, PD, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"   # validated slots 1-2
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200})

# Internal gaitpdb results from the Teammate's a2_06 run (PROJECT_LOG 2026-11-22): mean over 20x5 CV and
# participant-level bootstrap-of-CV 95% CI.
INTERNAL_M1FB = (0.665, 0.551, 0.744)


def iqr(x):
    x = pd.Series(x).dropna()
    return f"{x.median():.1f} ({x.quantile(.25):.0f}–{x.quantile(.75):.0f})"


def cohort_table(g, w):
    rows = []
    for name, d in (("gaitpdb (training, n = 165)", g), ("WearGait-PD (external, scored n = 113)", w)):
        for lab, grp in (("Controls", d[d.label == 0]), ("PD", d[d.label == 1])):
            rows.append({"cohort": name, "group": lab, "n": len(grp), "age_y median (IQR)": iqr(grp.age_y),
                         "female %": round(100 * (grp.sex == "F").mean()),
                         "speed m/s median (IQR)": (f"{grp.speed_01_mps.median():.2f} "
                                                    f"({grp.speed_01_mps.quantile(.25):.2f}–"
                                                    f"{grp.speed_01_mps.quantile(.75):.2f})"),
                         "H&Y median (PD)": (f"{grp.hy.median():.1f}" if lab == "PD" else "–"),
                         "stride_time_cv median": round(grp.stride_time_cv.median(), 2)})
    return pd.DataFrame(rows)


def fig_forest(path, auc_tab, emu):
    def get(name):
        r = auc_tab[auc_tab.analysis == name].iloc[0]
        return r.auc, r.ci_lower, r.ci_upper
    b = emu["h3_internal_reference_m1_fb"]
    items = [
        ("Internal: gaitpdb CV, full walks", INTERNAL_M1FB, "internal"),
        ("Internal: gaitpdb CV, emulated SelfPace walks", (b["auc_mean"], b["bootstrap"]["ci_lower"],
                                                            b["bootstrap"]["ci_upper"]), "internal"),
        ("External: frozen model, WearGait-PD (primary)", get("REF frozen M1-fb, SelfPace, 113"), "primary"),
        ("Walkway strides instead of insoles", get("S3 walkway strides, frozen M1-fb, SelfPace"), "sens"),
        ("Mean of all QC-passing trials", get("S2 frozen M1-fb, mean of QC-pass trials"), "sens"),
        ("≥ 15 valid strides per foot", get("S4 >= 15 valid strides/foot, frozen M1-fb"), "sens"),
        ("PD without DBS", get("S5 PD without DBS, frozen M1-fb"), "sens"),
        ("PD medication ON", get("S5 PD medication ON, frozen M1-fb"), "sens"),
        ("PD H&Y ≥ 2.5", get("S5 PD H&Y >= 2.5, frozen M1-fb"), "sens"),
        ("Age-matched 1:1 (n = 52)", get("S5 age-matched 1:1 (caliper 3 y), frozen M1-fb"), "sens"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y = np.arange(len(items))[::-1].astype(float)
    y[3:] -= 0.6                                                    # gap before the sensitivity block
    for yi, (lab, (a, lo, hi), kind) in zip(y, items):
        col = INK if kind != "sens" else MUTED
        ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(a, yi, "o", ms=8 if kind == "primary" else 6.5, color=PD if kind == "primary" else col,
                mec="white", mew=1.5, zorder=3)
        ax.text(1.02, yi, f"{a:.2f} [{lo:.2f}, {hi:.2f}]", va="center", ha="left", fontsize=8.5, color=INK,
                transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(0.505, y.max() + 0.75, "chance (0.5)", ha="left", fontsize=8, color=MUTED)
    ax.set_yticks(y, [i[0] for i in items])
    ax.set_xlim(0.25, 0.85)
    ax.set_ylim(y.min() - 0.7, y.max() + 1.1)
    ax.set_xlabel("AUC, M1-fb (stride_time_mean + stride_time_cv), 95% CI")
    ax.text(0.25, y[3] + 0.55, "Sensitivity analyses (exploratory, after unblinding)", fontsize=8.5,
            color=MUTED, ha="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_age(path, g, w):
    bins = np.arange(35, 96, 5)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=False)
    for ax, (title, d) in zip(axes, (("gaitpdb (training)", g), ("WearGait-PD (external)", w))):
        for lab, col, name in ((0, CO, "Controls"), (1, PD, "PD")):
            a = d.loc[d.label == lab, "age_y"].dropna()
            ax.hist(a, bins=bins, histtype="step", lw=2, color=col, label=f"{name} (median {a.median():.1f})")
            ax.axvline(a.median(), color=col, lw=1.2, ls=(0, (2, 2)))
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Age (years)")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("Participants")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_roc(path, df):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
    # linear predictor (unrounded) so the AUC equals the confirmatory value exactly
    for col_score, colr, name in (("m1_fb_logit", CO, "M1-fb (primary)"), ("m1_logit", PD, "M1 (secondary)")):
        fpr, tpr, _ = roc_curve(df.label, df[col_score])
        from sklearn.metrics import roc_auc_score
        ax.step(fpr, tpr, where="post", color=colr, lw=2, label=f"{name}, AUC {roc_auc_score(df.label, df[col_score]):.3f}")
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Frozen models on WearGait-PD (n = 113)", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d_writeup")
    args = ap.parse_args(argv)
    out = paths.outputs_root(args.run_id) / "writeup"
    out.mkdir(parents=True, exist_ok=True)
    before = guards.snapshot()

    # gaitpdb: training table + demographics
    g = pd.read_csv(O / "c_prefreeze" / "features" / "subject_v2.csv")
    dm = demographics.load_demographics(paths.gaitpdb_root())
    g = g.merge(dm[["subject_id", "sex", "age_y", "speed_01_mps", "hy_desc_only"]], on="subject_id", how="left")
    g = g.rename(columns={"hy_desc_only": "hy"})

    # WearGait-PD: scored cohort + labels (post-freeze) + H&Y (description only)
    key = pd.read_csv(paths.LABELS_DIR / "weargait_key.csv", usecols=["code", "original_subject", "group", "task"])
    key = key[key.task == "SelfPace"].set_index("code")
    w = pd.read_csv(O / "b02_20261122" / "weargait" / "subject_external_v2.csv")
    w = w[w.qc_pass].copy()
    w["label"] = w.subject_id.map(key.group)
    clin = pd.read_csv(paths.weargait_root() / "_restricted" / "PD - Demographic+Clinical - datasetV1.csv", header=1)
    clin["sid"] = clin["Subject ID"].map(norm_id)
    hy = pd.to_numeric(clin.drop_duplicates("sid").set_index("sid")["Modified Hoehn & Yahr Score"], errors="coerce")
    w["hy"] = w.subject_id.map(key.original_subject).map(hy)

    tab = cohort_table(g, w)
    guards.safe_write(out / "cohort_table.csv", tab.to_csv(index=False))

    auc_tab = pd.read_csv(O / "d_sensitivity" / "sensitivity" / "auc_table.csv")
    emu = json.loads((O / "c_prefreeze" / "emulation" / "summary.json").read_text())
    fig_forest(out / "fig1_auc_forest.png", auc_tab, emu)
    fig_age(out / "fig2_age_by_group.png", g, w)
    sc = pd.read_csv(O / "c_frozen" / "frozen_scores" / "external_scores.csv")
    sc["label"] = sc.subject_id.map(key.group)
    fig_roc(out / "fig3_external_roc.png", sc)
    guards.verify(before)
    print(tab.to_string(index=False))
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
