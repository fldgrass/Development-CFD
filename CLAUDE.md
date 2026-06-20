# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit web GUI that automates OpenFOAM CFD simulations for aquaculture fish-cage drag/lift coefficient analysis. The system builds OpenFOAM cases from STL files, runs them with MPI parallelization, and extracts Cd/Cl results into CSV files compatible with a downstream C++ mass-spring model.

Target environment: Ubuntu 22.04, NVIDIA RTX 3090 ×2, 16+ CPU cores, OpenFOAM v2312.

## Running the app

```bash
# Activate the venv first
source .venv/bin/activate

# Launch the Streamlit dashboard
streamlit run app.py --server.port 8501
# → open http://localhost:8501
```

Shell-script alternatives (no GUI):
```bash
bash scripts/run_unit_cell.sh [speed_m/s] [angle_deg] [cell_size_mm] [n_cores]
bash scripts/run_full_structure.sh [speed_m/s] [angle_deg] [diameter_m] [depth_m] [n_cores]
bash scripts/run_batch.sh unit_cell "0.5 1.0 1.5 2.0" "0 15 30 45" 16
```

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key packages: `streamlit`, `pyvista[all]`, `numpy`, `pandas`, `matplotlib`, `plotly`. PyVista is optional — the visualizer falls back to matplotlib 2D plots if unavailable.

## Architecture

```
app.py  (Streamlit UI)
  ├── modules/cfd_manager.py   ← all CFD backend logic
  │     ├── UnitCellCaseBuilder     builds single-cell periodic BC cases
  │     ├── FullStructureCaseBuilder builds whole-cage open-domain cases
  │     ├── OpenFOAMRunner          runs blockMesh → snappyHexMesh → simpleFoam → reconstructPar
  │     ├── ResultExtractor         parses postProcessing/forceCoeffs → CSV
  │     └── BatchAnalysisManager    loops over speed × angle combos
  └── modules/visualizer.py    ← PyVista/matplotlib rendering
        ├── OpenFOAMResultReader    reads time steps, residual logs, sampled data
        ├── CFDVisualizer           renders U/p/k/omega fields, residual plots, Cd/Cl charts
        └── AutoRefreshVisualizer   background thread for live plot refresh
```

### Case-building pattern

Both builder classes follow the same pattern:
1. `shutil.copytree(openfoam/<mode>/, results/<mode>/<case_name>/)` — clone the OpenFOAM template
2. Copy the uploaded STL to `constant/triSurface/`
3. Call `replace_in_file()` to patch `0/U`, `0/k`, `0/omega`, `system/blockMeshDict`, `system/controlDict`, `system/decomposeParDict` with computed values

The actual string literals to replace are hard-coded in each `_patch_*` method. If you change the OpenFOAM template files, update the corresponding `replace_in_file()` calls in `cfd_manager.py`.

### OpenFOAM template layout

```
openfoam/
├── unit_cell/      ← periodic (cyclic) BC template: xMin↔xMax, yMin↔yMax
│   ├── 0/          U, p, k, omega, nut
│   ├── constant/   transportProperties, turbulenceProperties, triSurface/
│   └── system/     blockMeshDict, snappyHexMeshDict, controlDict, fvSchemes,
│                   fvSolution, decomposeParDict
└── full_structure/ ← inlet/outlet/wall BC template
    ├── 0/
    ├── constant/
    └── system/     (same set + velocitySampling in controlDict)
```

### Result layout

```
results/
├── unit_cell/
│   ├── unit_cell_U1.00_A0.0_HHMMSS/   ← full OpenFOAM case dir
│   └── force_coeffs_unit_cell.csv      ← appended after each run
└── full_structure/
    └── ...
```

`ResultExtractor.save_csv()` appends one row per run to the CSV with columns: `speed_m_s, angle_deg, Cd, Cl, Cm, Fx_N, Fy_N, Fz_N, rho_kg_m3, case_name, timestamp`.

### Threading model

Analysis runs in a daemon `threading.Thread` launched from `app.py`. `add_script_run_ctx(thread)` is called before starting the thread so the background thread can write to `st.session_state`. Without this call, `ss.*` assignments in the thread silently fail.

### OpenFOAM discovery

`OpenFOAMRunner.find_openfoam_bashrc()` checks a list of candidate paths in order (openfoam10 → openfoam9 → openfoam2312 → openfoam2206 → user home). All subprocess calls are wrapped with `bash -c "source <bashrc> && <cmd>"` so the OpenFOAM environment is always loaded correctly.

## Key computed values

- Velocity vector from speed + AoA: `compute_velocity_vector(speed, angle_deg)` → `(Ux, 0.0, Uz)`
- Turbulence IC: `compute_turbulence_params(speed, intensity, length_scale)` → `{k, omega}` using k-ω SST formulae

## GPU acceleration

AmgX GPU solver is optional. If `$AMGX_DIR` is set and AmgX4Foam is built, the `fvSolution` in the template can be switched to `amgxSolver` for the pressure equation. Without it, the GAMG CPU solver in the templates is used automatically.
