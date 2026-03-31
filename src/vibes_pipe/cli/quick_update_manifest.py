import json
from pathlib import Path

pairs_path = Path("experiments/toast_data/pairs.json")
manifest_path = Path("experiments/toast_data/workspace_root/manifest.json")

pairs = json.load(open(pairs_path))
manifest = json.load(open(manifest_path))

# build lookup by subject id
pair_lookup = {p["id"]: p for p in pairs}

for entry in manifest["pairs"]:
    sid = entry["id"]

    files = entry.setdefault("files", {})

    pair_row = pair_lookup.get(sid)

    if pair_row and "noise" in pair_row:
        files["noise_mat"] = {"dst": pair_row["noise"]}
    else:
        files["noise_mat"] = None

# overwrite manifest
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print("manifest.json updated with noise paths")