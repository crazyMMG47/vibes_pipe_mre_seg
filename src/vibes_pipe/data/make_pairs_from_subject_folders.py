"""
Json manager for paired files of subjects. 
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_one(subj_dir: Path, pattern: str) -> Optional[Path]:
    """Return the first matching file (sorted), else None."""
    hits = sorted(subj_dir.glob(pattern))
    return hits[0] if hits else None


def find_many(subj_dir: Path, pattern: str) -> List[Path]:
    """Return all matching files (sorted)."""
    return sorted(subj_dir.glob(pattern))


def collect_subjects(
    root: Path,
    *,
    t2stack_mat_pat: str = "{id}_t2stack.mat",
    t2stack_nii_pat: str = "{id}_t2stack.nii",
    mask_mat_pat: str = "{id}_mask.mat",
    mu_mat_pat: str = "{id}_Mu.mat",
    pred_mat_pat: str = "{id}_pred*.mat",
    strict: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Collect subject records from root/<subject_id>/... based on patterns.
    Returns: (pairs, report)
    """
    pairs: List[Dict[str, Any]] = []
    report = {"root": str(root), "included": [], "skipped": []}

    for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        subj_id = subj_dir.name

        x_path = find_one(subj_dir, t2stack_mat_pat.format(id=subj_id))
        gt_path = find_one(subj_dir, mask_mat_pat.format(id=subj_id))

        nifti_path = find_one(subj_dir, t2stack_nii_pat.format(id=subj_id))
        mu_path = find_one(subj_dir, mu_mat_pat.format(id=subj_id))
        pred_paths = find_many(subj_dir, pred_mat_pat.format(id=subj_id))

        missing = []
        if x_path is None:
            missing.append("t2stack_mat")
        if gt_path is None:
            missing.append("mask_mat")

        if missing:
            msg = {"id": subj_id, "missing": missing, "dir": str(subj_dir)}
            report["skipped"].append(msg)
            if strict:
                raise SystemExit(f"[STRICT] Missing {missing} for subject {subj_id} in {subj_dir}")
            continue
        
        scanner_type = None
        # Handle for Helen's Run only
        # For the future label on scanners, it is depending on the person who run "process_noise_batch..."
        if subj_id.startswith("G"):
            scanner_type = "GE"
        elif subj_id.startswith("S"):
            scanner_type = "SIEMENS"
        
        rec: Dict[str, Any] = {
            "id": subj_id,
            "split": "train",  # may be overwritten
            "scanner_type": scanner_type,   # updated scanner types 
            "t2stack": str(x_path.resolve()),
            "t2stack_nii": str(nifti_path.resolve()) if nifti_path else None,
            "GT(human)": str(gt_path.resolve()),
            "eligible_preds": [str(p.resolve()) for p in pred_paths] if pred_paths else [],
            "NLI_output": str(mu_path.resolve()) if mu_path else None,
            "meta": {},
        }

        pairs.append(rec)
        report["included"].append({"id": subj_id, "dir": str(subj_dir)})

    report["n_found"] = len(report["included"]) + len(report["skipped"])
    report["n_included"] = len(report["included"])
    report["n_skipped"] = len(report["skipped"])
    return pairs, report


def assign_splits(
    pairs: List[Dict[str, Any]],
    *,
    mode: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 0,
    shuffle: bool = True,
) -> None:
    """
    mode:
      - "auto": assign 80/10/10 (or provided ratios)
      - "all-train": all train
      - "keep": do nothing
    """
    if mode == "keep":
        return
    if mode == "all-train":
        for it in pairs:
            it["split"] = "train"
        return
    if mode != "auto":
        raise ValueError(f"Unknown split mode: {mode}")

    n = len(pairs)
    if n == 0:
        return

    idx = list(range(n))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    for rank, i in enumerate(idx):
        if rank < n_train:
            pairs[i]["split"] = "train"
        elif rank < n_train + n_val:
            pairs[i]["split"] = "val"
        else:
            pairs[i]["split"] = "test"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build/update pairs.json from a dataset directory.")
    ap.add_argument("--root", type=Path, required=True, help="Root directory containing subject folders.")
    ap.add_argument("--out", type=Path, default=Path("./pairs.json"), help="Output pairs.json path.")
    ap.add_argument("--report", type=Path, default=None, help="Optional report JSON path (skips, counts).")

    ap.add_argument("--split", choices=["auto", "all-train", "keep"], default="auto")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shuffle", action="store_true", help="Do not shuffle before splitting.")

    ap.add_argument("--strict", action="store_true", help="Error out if any subject is missing required files.")

    # Patterns (easy future-proofing)
    ap.add_argument("--t2stack-mat", default="{id}_t2stack.mat")
    ap.add_argument("--t2stack-nii", default="{id}_t2stack.nii")
    ap.add_argument("--mask-mat", default="{id}_mask.mat")
    ap.add_argument("--mu-mat", default="{id}_Mu.mat")
    ap.add_argument("--pred-mat", default="{id}_pred*.mat")

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    pairs, report = collect_subjects(
        root,
        t2stack_mat_pat=args.t2stack_mat,
        t2stack_nii_pat=args.t2stack_nii,
        mask_mat_pat=args.mask_mat,
        mu_mat_pat=args.mu_mat,
        pred_mat_pat=args.pred_mat,
        strict=args.strict,
    )

    if not pairs:
        raise SystemExit("No valid subject pairs found.")

    assign_splits(
        pairs,
        mode=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")

    if args.report:
        rep_path = args.report.expanduser().resolve()
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"✅ Wrote {len(pairs)} subjects to: {out_path}")
    print(f"   Skipped: {report['n_skipped']} / Found: {report['n_found']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())