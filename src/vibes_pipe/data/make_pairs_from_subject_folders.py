"""
Helper functions that can update the pairs.json.


"""

from __future__ import annotations
import json 
from pathlib import Path
from typing import List, Dict


def collect_subjects(root: Path) -> List[Dict]:
    """
    Collect subjects by looking at the subjectories from the input directory. 
    
    """
    pairs = []

    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir():
            continue

        subj_id = subj_dir.name

        x_path = subj_dir / f"{subj_id}_t2stack.mat"
        nifti_path = subj_dir / f"{subj_id}_t2stack.nii"
        
        gt_path = subj_dir / f"{subj_id}_mask.mat"

        mu_path = subj_dir / f"{subj_id}_Mu.mat"
        prob_path = subj_dir / f"{subj_id}_pred*.mat"

        if not x_path.exists():
            print(f"[SKIP] missing X for {subj_id}")
            continue

        if not gt_path.exists():
            print(f"[SKIP] missing GT for {subj_id}")
            continue

        pairs.append(
            {
                "id": subj_id,
                "split": "train",  # temporary, may be overwritten by splitting
                "t2stack": str(x_path.resolve()),
                "t2stack_nii": str(nifti_path.resolve()),
                "GT(human)": str(gt_path.resolve()),
                "eligible_preds": str(prob_path.resolve()) if prob_path.exists() else None,
                "NLI_output": str(mu_path.resolve()) if mu_path.exists() else None,
                "meta": {},
            }
        )

    return pairs


def assign_splits(pairs: List[Dict], train_ratio=0.8, val_ratio=0.1) -> None:
    """
    Assign split on all dataset 
    """
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    for i, item in enumerate(pairs):
        if i < n_train:
            item["split"] = "train"
        elif i < n_train + n_val:
            item["split"] = "val"
        else:
            item["split"] = "test"
            


def main(mode="bypass"):
    """
    Prompt user to choose mode. Defaulty mode is Bypass-training. 
    
    """
    
    print("Important! \n You are about to change our json file recording all the subject data. Please read and response to the prompt below carefully. ")
    root_input = input("Enter the directory containing subject folers [full path]: \n\n").strip()
    
    root = Path(root_input).expanduser().resolve()
    
    if not root.exists():
        raise SystemExit(f"Directory you entered doesn't exist: {root}")
    
    
    # ---- Collect subjects ----
    pairs = collect_subjects(root)

    if not pairs:
        raise SystemExit("No valid subject pairs found.")

    print(f"Found {len(pairs)} valid subjects.")

    # ---- Ask about split ----
    print("Below we will prompt you to enter your data splitting mode. Just press [enter] or type 'default' for regular segmentation OR type 're-training' to re-assign train/val/test data labels.\n\n")
    split_choice = input("Mode of training [default/training]: \n\n ").strip().lower()

    if split_choice in ("", "y", "yes", "default"): # assign typical training
        assign_splits(pairs, train_ratio=0.8, val_ratio=0.1)
        print("Applied 80/10/10 split.")
        
    elif split_choice in ("retraining", "retrain"):
        print("All subjects set to train.")
        
    else:
        print("I can't understand your choice of split. Try again! ")
        
    # ---- Output path ----
    out_input = input(
        "Enter output pairs.json path (default=./pairs.json): \n "
    ).strip()

    out_path = Path(out_input).expanduser().resolve() if out_input else Path("./pairs.json").resolve()

    out_path.write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")

    print(f"\n✅ Created pairs.json with {len(pairs)} subjects")
    print(f"Location: {out_path}")


if __name__ == "__main__":
    main()