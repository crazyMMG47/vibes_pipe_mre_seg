# GUI Roadmap — vibes_pipe Prediction Viewer

**Architecture:** FastAPI (Python) backend + React/Vite frontend  
**Root:** `gui/` subfolder inside this repo  
**Purpose:** Browse MC-sampled predictions per subject, compare against GT, inspect metrics, and promote the best prediction to pseudo-GT.

## Status

| Phase | Status |
|-------|--------|
| 0 — Scaffold | ✅ Complete |
| 1 — Backend data layer | ✅ Complete |
| 2 — Core slice viewer | ✅ Complete |
| 3 — Metrics panel | ✅ Complete (built into Phase 2 delivery) |
| 4 — Candidate selection + export | ✅ Complete (built into Phase 2 delivery) |
| 5 — Stiffness integration | ✅ Complete (built into Phase 2 delivery) |
| 6 — Hardening + docs | ✅ Complete |

---

## Final directory layout (target state)

```
gui/
├── backend/
│   ├── main.py                  # FastAPI app — mounts routers + serves built frontend
│   ├── config.py                # runtime config (output_dir, manifest, workspace_root, port)
│   ├── routers/
│   │   ├── subjects.py          # list / detail endpoints
│   │   ├── slices.py            # slice-image serving
│   │   ├── metrics.py           # per-subject and per-sample metrics
│   │   └── export.py            # pseudo-GT write-back
│   ├── services/
│   │   ├── output_reader.py     # reads npy/mat from InferenceEngine output dir
│   │   ├── manifest_reader.py   # reads + updates manifest.json
│   │   └── slice_renderer.py    # numpy 3D volume → PNG bytes (using existing viz logic)
│   └── requirements.txt         # fastapi uvicorn scipy numpy pillow python-multipart
├── frontend/
│   ├── package.json
│   ├── vite.config.ts           # dev proxy: /api → localhost:8000
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx              # top-level layout: sidebar + main panel
│       ├── api/
│       │   └── client.ts        # typed fetch wrappers for every endpoint
│       ├── types/
│       │   └── index.ts         # Subject, SliceParams, SampleMetrics, ExportResult
│       ├── hooks/
│       │   ├── useSubjects.ts   # fetches subject list, caches
│       │   ├── useSubject.ts    # fetches one subject detail
│       │   ├── useSlice.ts      # fetches slice image URL for given params
│       │   └── useMetrics.ts    # fetches per-sample metrics
│       └── components/
│           ├── SubjectSidebar.tsx     # scrollable subject list, search, scanner badge
│           ├── SliceViewer.tsx        # one 2D image panel (image bytes → <img>)
│           ├── CandidateGrid.tsx      # [Input | GT | Sample_1 .. Sample_K] row
│           ├── SliceScrubber.tsx      # axis selector + index slider
│           ├── MetricsPanel.tsx       # dice/entropy/ged badges + recharts bar chart
│           ├── StiffnessPanel.tsx     # conditional — shows only if stiffness_available
│           ├── CandidateSelector.tsx  # click-to-select card highlight + export button
│           └── StatusBar.tsx          # loading / error / success toasts
├── run_dev.sh          # starts backend (:8000) + frontend dev server (:5173)
├── run_prod.sh         # builds frontend, starts single uvicorn process
└── README.md           # setup + run instructions (written in Phase 0)
```

---

## API contract (complete, agreed upfront)

All endpoints are prefixed `/api`.

```
GET  /api/health
     → { "status": "ok" }

GET  /api/subjects
     → SubjectSummary[]
     SubjectSummary {
       id: str, scanner_type: str, dice: float | null,
       n_mc_samples: int, stiffness_available: bool,
       mean_entropy: float, ged: float, saved_at: str
     }

GET  /api/subjects/{id}
     → SubjectDetail (all of SubjectSummary + prob_shape, threshold, checkpoint_path)

GET  /api/subjects/{id}/slice
     query params:
       volume: "raw" | "gt" | "mean" | "sample_{k}" | "stiffness"
       axis:   0 | 1 | 2        (default 2 = axial)
       index:  int              (default = middle slice for that axis)
       overlay: "gt" | "none"   (default "none") — burns GT contour onto image
       threshold: float         (default 0.5)
     → PNG image bytes (Content-Type: image/png)
     → 404 if volume not found (e.g. stiffness not yet available)

GET  /api/subjects/{id}/metrics
     → PerSubjectMetrics {
         mean_dice: float, mean_entropy: float, ged: float, mean_std: float,
         per_sample: SampleMetric[]   // one entry per MC sample
       }
     SampleMetric { sample_index: int, dice: float, entropy: float, std: float }

POST /api/subjects/{id}/set-pseudo-gt
     body: { "sample_index": int }
     → { "written_path": str, "subject_id": str }
     Side-effect: writes chosen mc_sample (thresholded) as .mat to
       <workspace_root>/<split>/<id>/pseudo_mask.mat
     (matching ManifestDataset pseudo_suffix convention)
```

---

## Phase 0 — Scaffold

**Goal:** Both servers start cleanly; React shows a placeholder page; `/api/health` returns 200. No real data yet.

### Backend tasks
- Create `gui/backend/main.py` with a bare FastAPI app, `/api/health` endpoint, and CORS middleware allowing `localhost:5173`.
- Create `gui/backend/config.py` that reads four env vars: `OUTPUT_DIR`, `MANIFEST_PATH`, `WORKSPACE_ROOT`, `PORT` (default 8000). Raise a clear error on startup if `OUTPUT_DIR` is missing.
- Create `gui/backend/requirements.txt`.

### Frontend tasks
- Scaffold Vite + React + TypeScript: `npm create vite@latest frontend -- --template react-ts`.
- Add Tailwind CSS, recharts, and `@tanstack/react-query`.
- Configure `vite.config.ts` proxy: any request to `/api` is forwarded to `http://localhost:8000`.
- Replace default App content with a centered placeholder: "vibes_pipe · Prediction Viewer".

### Startup scripts
- `gui/run_dev.sh`: exports the four env vars from a local `.env` file (git-ignored), starts `uvicorn` in background, then `npm run dev` in foreground. On Ctrl-C, kills uvicorn.
- `gui/run_prod.sh`: runs `npm run build`, then starts uvicorn (backend will serve `frontend/dist/` as static).
- `gui/README.md`: complete setup and run instructions (see template at end of this document).

### Deliverables / acceptance criteria
- [ ] `./gui/run_dev.sh` starts without errors.
- [ ] `curl localhost:8000/api/health` → `{"status":"ok"}`.
- [ ] Browser at `localhost:5173` shows the placeholder page with no console errors.
- [ ] `./gui/run_prod.sh` builds and serves on `localhost:8000`; the same placeholder is visible.

### Explicitly out of scope
- No real data reading. No React components beyond the placeholder.

---

## Phase 1 — Backend data layer

**Goal:** All API endpoints return real data. The frontend is not wired yet; test via curl / browser / Swagger UI (`localhost:8000/docs`).

### Backend tasks

**`services/output_reader.py`**
- `list_subjects(output_dir) → list[SubjectSummary]`: walks `output_dir/`, reads `meta.json` from each subfolder. Skips folders missing `meta.json`.
- `get_subject(output_dir, subject_id) → SubjectDetail | None`.
- `load_volume(output_dir, subject_id, volume_key) → np.ndarray | None`: maps `volume_key` → file (`prob_map.npy`, `pred_mask.npy`, `mc_samples.npy[k]`, etc.). Returns `None` if file missing.

**`services/manifest_reader.py`**
- `get_subject_manifest_entry(manifest_path, subject_id) → dict | None`: finds the pair record by `id`.
- `get_raw_image_path(manifest_path, workspace_root, subject_id) → Path | None`: resolves `t2stack` dst.
- `get_gt_path(manifest_path, workspace_root, subject_id) → Path | None`: resolves `GT(human)` dst.
- `get_stiffness_path(manifest_path, workspace_root, subject_id) → Path | None`: resolves `NLI_output` dst; returns `None` if missing.
- `write_pseudo_gt(workspace_root, split, subject_id, mask_array) → Path`: writes `<workspace_root>/<split>/<subject_id>/pseudo_mask.mat` using `scipy.io.savemat`. Creates directories as needed.

**`services/slice_renderer.py`**
- `render_slice(volume_3d, axis, index, overlay_mask=None, threshold=0.5) → bytes`: takes a `[H,W,D]` or `[C,H,W,D]` numpy array, extracts the 2D slice using `src/vibes_pipe/viz/slices.get_slice`, applies percentile normalization, optionally burns a GT contour (white, 1 px), encodes to PNG bytes via Pillow. Returns raw bytes.
- Must handle `[C,H,W,D]` (squeeze channel dim) and `[K,C,H,W,D]` MC samples (index on first dim).

**`routers/subjects.py`**
- `GET /api/subjects` — calls `output_reader.list_subjects`, enriches each entry with `stiffness_available` flag (checks `manifest_reader.get_stiffness_path`).
- `GET /api/subjects/{id}` — full detail.

**`routers/slices.py`**
- `GET /api/subjects/{id}/slice` — resolves volume from `volume` query param, calls `slice_renderer.render_slice`, returns `Response(content=png_bytes, media_type="image/png")`. Returns `404` for missing volumes (e.g. stiffness).

**`routers/metrics.py`**
- `GET /api/subjects/{id}/metrics` — reads `mc_samples.npy` and `GT(human).mat`, computes per-sample Dice using `src/vibes_pipe/metric/eval_metrics.DiceScore`, returns `PerSubjectMetrics`.

**`routers/export.py`**
- `POST /api/subjects/{id}/set-pseudo-gt` — reads the chosen `mc_samples.npy[sample_index]`, thresholds at 0.5, calls `manifest_reader.write_pseudo_gt`.

### Deliverables / acceptance criteria
- [ ] `GET /api/subjects` returns a JSON array (may be empty if `OUTPUT_DIR` has no inference results yet, but must not 500).
- [ ] `GET /api/subjects/{id}/slice?volume=gt&axis=2&index=30` returns a valid PNG (verify in browser or `curl … | file -`).
- [ ] `GET /api/subjects/{id}/slice?volume=stiffness` returns 404 when `NLI_output` is absent.
- [ ] `GET /api/subjects/{id}/metrics` returns per-sample Dice array.
- [ ] `POST /api/subjects/{id}/set-pseudo-gt` with `{"sample_index": 0}` writes `pseudo_mask.mat` to the correct path.
- [ ] `/docs` (Swagger UI) documents all endpoints correctly.

### Explicitly out of scope
- No frontend changes. No authentication. No pagination (dataset is small).

---

## Phase 2 — Core slice viewer (frontend)

**Goal:** A working browser UI where you can pick a subject and scrub through axial slices, seeing the raw image, GT overlay, and all MC prediction samples in a horizontal grid — matching the layout of `plot_pred.py`.

### Frontend tasks

**`api/client.ts`**
- Typed wrappers: `fetchSubjects()`, `fetchSubject(id)`, `fetchSliceUrl(id, params)` (returns an object URL from a blob fetch), `fetchMetrics(id)`.

**`types/index.ts`**
- TypeScript interfaces for `SubjectSummary`, `SubjectDetail`, `SliceParams`, `PerSubjectMetrics`, `SampleMetric`.

**`hooks/useSubjects.ts`** — react-query `useQuery` over `fetchSubjects`.  
**`hooks/useSubject.ts`** — react-query `useQuery` over `fetchSubject(id)`.  
**`hooks/useSlice.ts`** — react-query `useQuery` that returns a blob URL for a given `SliceParams`; key includes all params so refetch is automatic on change.

**`components/SubjectSidebar.tsx`**
- Scrollable list. Each row shows: subject ID, scanner badge (GE/SIEMENS colored chip), Dice score (green/amber/red by value), stiffness indicator icon (greyed out when unavailable). Click selects the subject and sets it as `activeSubjectId` in App state.

**`components/SliceScrubber.tsx`**
- Axis toggle (Axial / Sagittal / Coronal).
- Numeric index slider; range auto-set to `[0, depth-1]` based on `SubjectDetail.prob_shape`.
- Overlay toggle: "GT Contour on / off".

**`components/SliceViewer.tsx`**
- Accepts a `sliceUrl: string | null` and renders the PNG inside a fixed-aspect container. Shows a skeleton placeholder while loading (`useSlice` status === 'loading'). Shows a grey "not available" placeholder on 404.

**`components/CandidateGrid.tsx`**
- Renders a horizontal row of `SliceViewer` panels:
  - Column 0: raw image (`volume=raw`)
  - Column 1: GT overlay (`volume=gt`)
  - Columns 2…K+1: `volume=sample_0` … `volume=sample_{K-1}`
- Column headers: "Input", "Ground Truth", "Sample 1" … "Sample K"
- All columns share the same `axis` and `index` from `SliceScrubber` state.

**`App.tsx`**
- Layout: `<SubjectSidebar>` fixed left, main area = `<CandidateGrid>` + `<SliceScrubber>`.
- Active subject ID in `useState`; passed down via props or context.

### Deliverables / acceptance criteria
- [ ] Selecting a subject in the sidebar populates the candidate grid.
- [ ] Moving the slice slider updates all column images simultaneously (no stale frames).
- [ ] Toggling "GT Contour" re-fetches slices with `overlay=gt` or `overlay=none`.
- [ ] Switching axis (Axial/Sagittal/Coronal) updates all columns.
- [ ] While images load, skeleton placeholders are shown (no layout shift).
- [ ] Subjects with stiffness unavailable show a greyed icon in sidebar; no errors in the main panel.

### Explicitly out of scope
- No metrics chart. No candidate selection. No stiffness panel. No export.

---

## Phase 3 — Metrics panel

**Goal:** Dice score, predictive entropy, and GED are displayed beside the slice viewer. A bar chart shows per-sample Dice so the user can visually identify the best candidate.

### Frontend tasks

**`hooks/useMetrics.ts`** — react-query over `fetchMetrics(id)`.

**`components/MetricsPanel.tsx`**
- Top row: three stat cards — "Mean Dice", "Mean Entropy", "GED" — each showing the value from `meta.json` with a subtle color scale (green ≥ 0.8, amber 0.6–0.8, red < 0.6 for Dice).
- Bottom: recharts `BarChart` with one bar per MC sample. X-axis: "Sample 0" … "Sample K-1". Y-axis: Dice [0, 1]. Bar color reflects value (same green/amber/red scale). Hovering a bar shows a tooltip with entropy and std.
- Layout: `MetricsPanel` sits below `CandidateGrid`, full width.

### Deliverables / acceptance criteria
- [ ] Metrics cards update when a new subject is selected.
- [ ] Bar chart shows one bar per sample with correct Dice values.
- [ ] Hovering a bar shows tooltip with sample index, Dice, entropy, std.
- [ ] Panel shows a loading skeleton while `useMetrics` is pending.
- [ ] Panel shows a "metrics unavailable" message (no GT at inference time) gracefully without crashing.

### Explicitly out of scope
- No export. No stiffness. No clicking bars to select candidates (that is Phase 4).

---

## Phase 4 — Candidate selection and pseudo-GT export

**Goal:** The user can click a sample column (or its bar in the chart) to mark it as the chosen pseudo-GT candidate, then confirm export with a single button click.

### Frontend tasks

**`components/CandidateSelector.tsx`**
- Wraps each sample column header in `CandidateGrid`. A selected column gets a visible highlight ring (e.g. amber border, "Selected" badge).
- Clicking a column header or its metrics bar sets `selectedSampleIndex` in component state.
- Renders an "Export as Pseudo-GT" button that is enabled only when a sample is selected. Button shows a spinner while the POST is in flight.
- On success: shows a green toast "Pseudo-GT written to `<path>`". On error: red toast with the error message.

**`api/client.ts`** — add `postSetPseudoGt(id, sampleIndex)` fetch wrapper.

**`components/StatusBar.tsx`**
- Minimal toast system (slide-in from bottom-right, auto-dismiss in 4 s). Used by `CandidateSelector` and later by the stiffness panel.

### Deliverables / acceptance criteria
- [ ] Clicking a sample column highlights it and enables the export button.
- [ ] Clicking the bar in `MetricsPanel` for sample K also highlights column K in the grid.
- [ ] Clicking "Export as Pseudo-GT" calls `POST /api/subjects/{id}/set-pseudo-gt` with the correct `sample_index`.
- [ ] On success, `pseudo_mask.mat` exists at `<workspace_root>/<split>/<id>/pseudo_mask.mat`.
- [ ] Toast confirms the written path.
- [ ] Selecting a different subject resets `selectedSampleIndex` to null.

### Explicitly out of scope
- No manifest.json update (just the .mat write). No stiffness.

---

## Phase 5 — Stiffness integration

**Goal:** When `NLI_output.mat` is present for a subject, a stiffness panel appears to the right of the prediction grid, synchronized to the same slice index.

### Backend tasks
- `slices.py` router: the existing `volume=stiffness` arm already returns 404 when absent. Verify it returns a valid PNG when `NLI_output.mat` exists (stiffness field is a float volume — use a `hot` / `jet` colormap instead of `gray`; add a small colorbar annotation to the PNG via Matplotlib).

### Frontend tasks

**`components/StiffnessPanel.tsx`**
- Conditionally rendered: only mounts when `SubjectDetail.stiffness_available === true`.
- Renders a single `SliceViewer` column (`volume=stiffness`, same `axis` and `index` as the candidate grid).
- Column header: "Stiffness (NLI)" with an info tooltip: "Displays when NLI_output.mat is present for this subject."
- When `stiffness_available` is false, renders a placeholder card: "Stiffness not yet available for this subject."

**`App.tsx`**
- Insert `<StiffnessPanel>` to the right of `<CandidateGrid>` in the main layout. Because it's conditional, `CandidateGrid` should naturally shrink/grow (flex layout).

### Deliverables / acceptance criteria
- [ ] For a subject with `NLI_output.mat`, the stiffness panel renders alongside the prediction columns.
- [ ] Scrubbing the slice index updates both the prediction columns and the stiffness slice synchronously.
- [ ] For a subject without `NLI_output.mat`, a styled placeholder card appears — no 500, no blank space, no console errors.
- [ ] Switching from a subject with stiffness to one without (and back) works without a page refresh.

### Explicitly out of scope
- No quantitative correlation between stiffness and segmentation. No stiffness-based candidate ranking.

---

## Phase 6 — Hardening and documentation

**Goal:** The app is robust for daily research use: no blank screens on bad data, loading states everywhere, and a developer can set it up from scratch following the README alone.

### Frontend tasks
- Add React error boundaries around `CandidateGrid`, `MetricsPanel`, and `StiffnessPanel` so one panel crashing does not take down the page.
- Handle the case where `OUTPUT_DIR` is valid but empty — sidebar shows "No inference results found. Run inference first." with the CLI command inline.
- Keyboard navigation: arrow keys scrub the slice index when the slider is focused.
- `<title>` updates to reflect the active subject ID.
- Confirm the Vite prod build is clean (`npm run build` zero errors, zero TypeScript errors).

### Backend tasks
- Add a startup check: on launch, log clearly whether `OUTPUT_DIR`, `MANIFEST_PATH`, and `WORKSPACE_ROOT` exist. If `MANIFEST_PATH` is missing, the server still starts but all manifest-dependent calls return a clear JSON error (not a 500 traceback).
- Add `GET /api/subjects/{id}/slice` cache headers (`Cache-Control: max-age=300`) so the browser doesn't re-fetch static slices every render.

### `gui/README.md` (final, complete version)

Must contain exactly these sections (populate during this phase):

1. **Prerequisites** — Python ≥ 3.10, Node.js ≥ 18, an activated conda/venv environment with vibes_pipe installed.
2. **Install** — `pip install -r gui/backend/requirements.txt` then `cd gui/frontend && npm install`.
3. **Configure** — copy `gui/.env.example` to `gui/.env`, fill in:
   ```
   OUTPUT_DIR=/path/to/output          # InferenceEngine save_dir
   MANIFEST_PATH=/path/to/manifest.json
   WORKSPACE_ROOT=/path/to/workspace_root
   PORT=8000
   ```
4. **Run (development)** — `./gui/run_dev.sh` → backend at `localhost:8000`, UI at `localhost:5173`.
5. **Run (production / single port)** — `./gui/run_prod.sh` → everything at `localhost:8000`.
6. **Access** — open `http://localhost:5173` (dev) or `http://localhost:8000` (prod) in any browser.
7. **Using the viewer** — step-by-step walkthrough: pick subject → scrub slices → inspect metrics → click candidate → export.
8. **Troubleshooting** — common errors and fixes (missing env vars, port already in use, missing NLI output).

### Deliverables / acceptance criteria
- [ ] `./gui/run_dev.sh` on a clean checkout with a valid `.env` reaches a functional UI within 30 seconds.
- [ ] Navigating to a subject with a corrupted `.npy` file shows an error card, not a white screen.
- [ ] `npm run build` exits 0 with no TypeScript errors.
- [ ] README is self-contained: a labmate who has never seen this repo can get the GUI running by following it alone.
- [ ] All Phase 0–5 acceptance criteria still pass.

---

## Implementation order and estimated scope

| Phase | Core work | Est. new files |
|-------|-----------|----------------|
| 0 | Scaffold, scripts, README skeleton | ~8 |
| 1 | Backend services + all 5 API endpoints | ~8 |
| 2 | Subject sidebar, slice viewer, candidate grid | ~8 |
| 3 | Metrics cards + bar chart | ~3 |
| 4 | Candidate selection + export flow | ~3 |
| 5 | Stiffness panel (backend colormap + frontend component) | ~3 |
| 6 | Error boundaries, cache headers, final README | ~2 |

Each phase is independently reviewable. A phase is **complete** only when all its acceptance criteria are checked off. Do not start the next phase until the current one passes.

---

## Key design constraints (do not deviate without discussion)

1. **Slice images are rendered server-side.** The backend returns PNG bytes; the frontend only displays `<img src={blobUrl}>`. No numpy → canvas drawing in JS.
2. **Use existing Python viz code.** `slice_renderer.py` must import and reuse `src/vibes_pipe/viz/slices.get_slice` and `_normalize_image` rather than re-implementing normalization.
3. **Pseudo-GT write path must match `ManifestDataset` convention.** The file is written to `<workspace_root>/<split>/<id>/pseudo_mask.mat` so it is automatically picked up when `label_mode="prefer_pseudo"` or `"pseudo"`.
4. **No model loading in the backend.** The GUI is a viewer for already-saved inference output. It must never call PyTorch or run inference.
5. **React Query for all server state.** No raw `useEffect` + `fetch`. This ensures consistent loading/error states and automatic deduplication of requests.
6. **Tailwind only — no component library.** Keeps the bundle small and avoids version-lock. Recharts is the one exception (charts only).
