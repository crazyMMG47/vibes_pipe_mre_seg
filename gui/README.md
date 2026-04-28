# vibes_pipe Prediction Viewer

Interactive GUI for reviewing MC-sampled segmentation predictions, comparing against GT, and selecting pseudo-GT candidates.

---

## Prerequisites

- **Python**: via `conda activate torch3090` (Python 3.9, with PyTorch + project deps)
- **Node.js 20**: installed into `torch3090` via `conda install -n torch3090 nodejs=20.17.0`
- The main `vibes_pipe` package must be accessible from the repo root (it is, since we launch uvicorn with `--app-dir <repo_root>`)

---

## 1 — Install

```bash
# Python deps (fastapi, uvicorn, pillow, python-multipart)
conda run -n torch3090 pip install -r gui/backend/requirements.txt

# Node deps (auto-installed on first run_dev.sh, but can do manually)
conda run -n torch3090 npm install --prefix gui/frontend
```

---

## 2 — Configure

```bash
cp gui/.env.example gui/.env
```

Edit `gui/.env`:

```env
OUTPUT_DIR=/path/to/vibes_pipe/output       # InferenceEngine save_dir (e.g. output/pub_m1)
MANIFEST_PATH=/path/to/workspace_root/manifest.json
WORKSPACE_ROOT=/path/to/workspace_root
PORT=8000
```

| Variable | Required | What it does |
|---|---|---|
| `OUTPUT_DIR` | **Yes** | Where `InferenceEngine` wrote subject subfolders (prob_map.npy, mc_samples.npy, meta.json) |
| `MANIFEST_PATH` | No* | Enables raw image, GT, and stiffness loading |
| `WORKSPACE_ROOT` | No* | Enables pseudo-GT export |
| `PORT` | No | Backend port (default 8000) |

\* Slice viewer for raw/GT/stiffness and pseudo-GT export are disabled without these.

---

## 3 — Run (development — two servers, hot reload)

```bash
./gui/run_dev.sh
```

- **Backend** starts at `http://localhost:8000` (auto-reloads on Python changes)
- **Frontend** starts at `http://localhost:5173` (hot module replacement)
- **Access the UI** at `http://localhost:5173`
- Swagger API docs at `http://localhost:8000/docs`
- Stop with `Ctrl-C`

---

## 4 — Run (production — single port)

```bash
./gui/run_prod.sh
```

Builds React, then serves everything from FastAPI on port `$PORT`.  
**Access** at `http://localhost:8000`

---

## 5 — Using the viewer

1. **Select a subject** from the left sidebar (sorted by ID; search box at top)
2. **Browse slices** using the scrubber bar — drag the slider or use `←` `→` arrow keys
3. **Switch orientation** with Axial / Sagittal / Coronal buttons
4. **Toggle GT Contour** to burn the ground-truth boundary onto every panel
5. **Select a candidate** — click any Sample column image, or click its bar in the Metrics chart
6. **Export** — hit "Export as Pseudo-GT" to write the chosen mask to `<workspace_root>/<split>/<id>/pseudo_mask.mat`
7. **Stiffness** — when `NLI_output.mat` is present in the manifest, a stiffness panel appears on the right (hot colormap)

---

## 6 — Troubleshooting

| Problem | Fix |
|---|---|
| `OUTPUT_DIR env var is required` | Create `gui/.env` from `.env.example` and set `OUTPUT_DIR` |
| Backend fails to import vibes_pipe | Run from repo root; uvicorn uses `--app-dir <repo_root>` |
| Port already in use | Change `PORT=` in `.env`, or `kill $(lsof -ti:8000)` |
| Frontend shows "Failed to load subjects" | Make sure backend is running; check `localhost:8000/api/health` |
| Raw/GT slices missing | Set `MANIFEST_PATH` and `WORKSPACE_ROOT` in `.env` |
| Stiffness always "not yet available" | Run NLI inference to generate `NLI_output.mat`; update manifest |
| `npm: command not found` | `conda install -n torch3090 nodejs=20.17.0` |
