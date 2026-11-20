"""E1: recording manifest + checksum verification (T1, T2).

Owner: Lead (data/*). Run from repo root:
    GAITPDB_ROOT=... python -m experiments.external_transport.scripts.a2_01_manifest --run-id <id>
"""
import argparse

from experiments.external_transport.etx import guards, manifest, paths
from experiments.external_transport.scripts.a2_00_env_and_guard import default_run_id


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=default_run_id())
    args = ap.parse_args(argv)
    out = paths.outputs_root(args.run_id) / "manifest"

    before = guards.snapshot()
    rec, exc = manifest.build_manifest(paths.gaitpdb_root())
    guards.safe_write(out / "recordings.csv", rec.to_csv(index=False))
    guards.safe_write(out / "excluded.csv", exc.to_csv(index=False))
    guards.verify(before)

    prim = rec[rec.include_primary]
    print(f"[E1] recordings={len(rec)} subjects={rec.subject_id.nunique()} primary={len(prim)} "
          f"(checksums all match: {bool(rec.sha256_official_match.all())})")
    print("     primary by study:", prim.groupby("study").size().to_dict(),
          "| by label:", prim.groupby("label").size().to_dict())
    print("     excluded by condition:", exc.groupby("condition").size().to_dict())
    print("     fs (Hz):", sorted(rec.fs_hz.round(3).unique().tolist()),
          "| all uniform:", bool(rec.fs_uniform.all()))


if __name__ == "__main__":
    main()
