"""B1: download WearGait-PD (Version 1) SelfPace trials from Synapse and blind them on arrival.

Owner: Lead (data/*). Needs a Synapse personal access token in ~/.synapseConfig (run `synapse config`)
or in $SYNAPSE_AUTH_TOKEN. Run from repo root:
    python -m experiments.external_transport.scripts.b_01_weargait_download [--tasks SelfPace ...]

Layout under $WEARGAIT_ROOT (default ~/Downloads/weargait-pd, outside the repo; never re-uploaded):
    blinded/WG###_<task>.csv   trial files, renamed, read-only        <- the only folder analysis code reads
    blinded/manifest.csv       code, task, file, sha256, n_bytes      (no Synapse ids: they reveal names)
    _restricted/               group-split demographics + PKMAS walkway metrics, original names
labels/weargait_key.csv        code, original subject, group, Synapse id/version, sha256 (repo-ignored)

Nothing printed by this script identifies anyone's group. It refuses to run twice (codes are random
and drawn once); to redo it, delete blinded/ and labels/weargait_key.csv by hand.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from experiments.external_transport.etx import guards, paths
from experiments.external_transport.etx import weargait_blind as wb
from experiments.external_transport.scripts.a2_00_env_and_guard import default_run_id


def _children(syn, parent, types):
    return list(syn.getChildren(parent, includeTypes=types))


def list_trials(syn, tasks):
    """Selected trial files from both group folders, with a folder-vs-prefix consistency check."""
    trials = []
    for folder_name, folder_id in wb.SYN_GROUP_FOLDERS.items():
        csv_dirs = [c for c in _children(syn, folder_id, ["folder"]) if c["name"] == wb.CSV_FOLDER_NAME]
        if len(csv_dirs) != 1:
            raise RuntimeError(f"expected one '{wb.CSV_FOLDER_NAME}' folder under {folder_id}")
        files = {c["name"]: c["id"] for c in _children(syn, csv_dirs[0]["id"], ["file"])}
        folder_group = 1 if folder_name.startswith("PD") else 0
        for t in wb.select(files, tasks):
            if t["group"] != folder_group:
                raise RuntimeError("file prefix disagrees with its group folder (check FILE_RE)")
            trials.append({**t, "synapse_id": files[t["name"]]})
    return trials


def download(syn, syn_id, dest: Path):
    ent = syn.get(syn_id, downloadLocation=str(dest), ifcollision="overwrite.local")
    return Path(ent.path), int(ent.versionNumber)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(wb.DEFAULT_TASKS))
    ap.add_argument("--run-id", default=default_run_id())
    args = ap.parse_args(argv)

    import synapseclient                                   # optional dependency, only needed here
    root = paths.weargait_root(create=True)
    blinded, staging, restricted = root / "blinded", root / "_staging", root / "_restricted"
    key_path = paths.LABELS_DIR / "weargait_key.csv"
    if key_path.exists() or (blinded.exists() and any(blinded.iterdir())):
        raise SystemExit(f"Already blinded ({key_path} or {blinded} exists). Not re-drawing codes.")

    before = guards.snapshot()
    syn = synapseclient.Synapse(silent=True)
    syn.login()                                            # token from ~/.synapseConfig or env
    print(f"[B1] logged in to Synapse as {syn.getUserProfile()['userName']}")

    # 1) download everything to staging first (codes are drawn only after all files arrived)
    trials = list_trials(syn, args.tasks)
    print(f"[B1] {len(trials)} trial files selected for tasks {args.tasks}; downloading ...")
    staging.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(trials, 1):
        t["staged"], t["synapse_version"] = download(syn, t["synapse_id"], staging)
        if i % 25 == 0 or i == len(trials):
            print(f"     {i}/{len(trials)}")

    # 2) draw codes, move + rename, make read-only
    codes = wb.assign_codes(t["subject"] for t in trials)
    blinded.mkdir(parents=True, exist_ok=True)
    key_rows, man_rows = [], []
    for t in trials:
        code = codes[t["subject"]]
        dst = blinded / wb.blinded_name(code, t["task"])
        shutil.move(str(t["staged"]), dst)
        os.chmod(dst, 0o444)
        sha = guards.sha256_file(dst)
        man_rows.append({"code": code, "task": t["task"], "file": dst.name, "sha256": sha,
                         "n_bytes": dst.stat().st_size})
        key_rows.append({"code": code, "original_subject": t["subject"], "group": t["group"],
                         "task": t["task"], "original_name": t["name"], "synapse_id": t["synapse_id"],
                         "synapse_version": t["synapse_version"], "sha256": sha})
    man = pd.DataFrame(man_rows).sort_values(["code", "task"])
    (blinded / "manifest.csv").write_text(man.to_csv(index=False))
    guards.safe_write(key_path, pd.DataFrame(key_rows).sort_values(["code", "task"]).to_csv(index=False))

    # 3) group-split / id-bearing tables -> _restricted (original names; blinded later via allow-list)
    restricted.mkdir(parents=True, exist_ok=True)
    extra = dict(wb.RESTRICTED_FILES)
    extra.update({c["id"]: c["name"] for c in _children(syn, wb.WALKWAY_METRICS_FOLDER, ["file"])})
    headers = {}
    for syn_id in extra:
        p, _ = download(syn, syn_id, restricted)
        df = pd.read_csv(p, nrows=0) if p.suffix.lower() == ".csv" else None
        headers[p.name] = list(df.columns) if df is not None else "(not csv)"

    if staging.exists() and not any(staging.iterdir()):
        staging.rmdir()

    # 4) leak check on the blinded files (first 20 lines)
    leaks = sorted(r["file"] for r in man_rows if wb.leak_check(blinded / r["file"]))   # sorted: trial order is by group
    guards.verify(before)

    summary = {"n_subjects": len(codes), "n_files": len(man_rows),
               "files_per_task": man.groupby("task").size().to_dict(),
               "n_files_with_possible_leak": len(leaks), "leak_files": leaks,
               "restricted_headers": headers, "root": str(root)}
    out = paths.outputs_root(args.run_id) / "weargait"
    guards.safe_write(out / "b01_download_summary.json", json.dumps(summary, indent=2))
    print(f"[B1] subjects={summary['n_subjects']} files={summary['n_files']} "
          f"per task={summary['files_per_task']}")
    print(f"[B1] files with a possible group string in the first 20 lines: {len(leaks)} {leaks[:10]}")
    for name, cols in headers.items():
        print(f"[B1] restricted header  {name}: {cols}")
    print(f"[B1] key written to {key_path} (git-ignored; do not open before `model-frozen`)")


if __name__ == "__main__":
    main()
