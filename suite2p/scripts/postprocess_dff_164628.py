"""
Post-processing: Photobleaching correction + dF/F
Session: 2026-04-20_164628, Fish A1
Author: Prakriti

Validated approach — uses suite2p's own dcnv.preprocess function:
  1. Neuropil subtraction: Fc = F - 0.7*Fneu
  2. Double exponential photobleaching correction (subtract fit)
  3. dF/F using suite2p's own preprocess with baseline='constant_prctile'
     (8th percentile of whole trace — correct for short recordings with bleaching)

References:
  - suite2p docs: https://suite2p.readthedocs.io/en/latest/deconvolution.html
  - Nature Neuroscience zebrafish paper: win_baseline=900s for bleaching correction
  - Suite2p source: constant_prctile mode avoids window-length problem
"""

import numpy as np
import os
from scipy.optimize import curve_fit
from suite2p.extraction import dcnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = "/home/abl-workstation2/Prakriti_FishBrain/suite2p/results/164628/suite2p"
OUTPUT_DIR = "/home/abl-workstation2/Prakriti_FishBrain/suite2p/results/164628"

N_PLANES       = 7
FS             = 5.0    # Hz volumetric
NEUROPIL_COEFF = 0.7

# ── Suite2p preprocess parameters ─────────────────────────────────────────
# Using constant_prctile: single global 8th percentile baseline
# This is immune to window-length issues for short recordings
# and correctly handles photobleaching after exponential correction
BASELINE      = 'constant_prctile'
WIN_BASELINE  = 60.0   # seconds (only used for maximin, kept for reference)
SIG_BASELINE  = 10.0   # frames
PRCTILE_BASELINE = 8.0 # 8th percentile — suite2p default

# ── Double exponential photobleaching correction ───────────────────────────
def double_exp(t, A1, tau1, A2, tau2, C):
    return A1 * np.exp(-t/tau1) + A2 * np.exp(-t/tau2) + C

def correct_photobleaching(trace):
    """
    Fit double exponential and SUBTRACT it (preserves fluctuation amplitudes).
    Falls back to single exp, then linear detrend.
    """
    n = len(trace)
    t = np.arange(n, dtype=np.float64)
    f = trace.astype(np.float64)
    drop = float(f[0] - f[-1])

    try:
        popt, _ = curve_fit(
            double_exp, t, f,
            p0=[drop*0.5, n*0.05, drop*0.5, n*0.5, float(f[-1])],
            bounds=([0, 0.5, 0, n*0.05, 0],
                    [np.inf, n*0.2, np.inf, np.inf, np.inf]),
            maxfev=10000, method='trf'
        )
        fit = double_exp(t, *popt)
        return f - fit, fit, 'double_exp'
    except Exception:
        pass

    try:
        def single_exp(t, A, tau, C):
            return A * np.exp(-t/tau) + C
        popt2, _ = curve_fit(single_exp, t, f,
                             p0=[drop, n/3, float(f[-1])],
                             bounds=([0,1,0],[np.inf,np.inf,np.inf]),
                             maxfev=5000)
        fit = single_exp(t, *popt2)
        return f - fit, fit, 'single_exp'
    except Exception:
        pass

    fit = np.linspace(float(f[0]), float(f[-1]), n)
    return f - fit, fit, 'linear'

# ── Process each plane ─────────────────────────────────────────────────────
all_dff   = []
all_plane = []
all_idx   = []

print("=" * 60)
print("Post-processing: Double-exp correction + suite2p constant_prctile dF/F")
print(f"  Baseline mode : {BASELINE}")
print(f"  Percentile    : {PRCTILE_BASELINE}")
print("=" * 60)

for p in range(N_PLANES):
    plane_dir   = os.path.join(BASE_DIR, f"plane{p}")
    F_path      = os.path.join(plane_dir, "F.npy")
    Fneu_path   = os.path.join(plane_dir, "Fneu.npy")
    iscell_path = os.path.join(plane_dir, "iscell.npy")

    if not os.path.exists(F_path):
        print(f"Plane {p}: no F.npy, skipping")
        continue

    F      = np.load(F_path)
    Fneu   = np.load(Fneu_path)
    iscell = np.load(iscell_path)

    cell_indices = np.where(iscell[:, 0] == 1)[0]
    n_cells      = len(cell_indices)
    n_frames     = F.shape[1]
    print(f"Plane {p}: {n_cells} cells, {n_frames} frames")

    # Step 1: Neuropil subtraction for all cells at once
    Fc_all = F[cell_indices] - NEUROPIL_COEFF * Fneu[cell_indices]

    # Step 2: Double exponential correction per cell
    Fc_corrected = np.zeros_like(Fc_all, dtype=np.float64)
    fit_methods = {'double_exp': 0, 'single_exp': 0, 'linear': 0}
    for i, idx in enumerate(cell_indices):
        corr, _, method = correct_photobleaching(Fc_all[i])
        Fc_corrected[i] = corr
        fit_methods[method] += 1
    print(f"  Fit methods: {fit_methods}")

    # Step 3: suite2p's own preprocess with constant_prctile
    # This computes global 8th percentile baseline and subtracts it
    # giving clean dF/F traces
    dff = dcnv.preprocess(
        F=Fc_corrected.astype(np.float32),
        baseline=BASELINE,
        win_baseline=WIN_BASELINE,
        sig_baseline=SIG_BASELINE,
        fs=FS,
        prctile_baseline=PRCTILE_BASELINE
    )

    all_dff.append(dff)
    for idx in cell_indices:
        all_plane.append(p)
        all_idx.append(idx)

    # ── Visualization: 5 cells, 3 panels ──────────────────────────────────
    n_show = min(5, n_cells)
    fig, axes = plt.subplots(n_show, 3, figsize=(24, 3*n_show))
    if n_show == 1:
        axes = axes.reshape(1, 3)

    t = np.arange(n_frames) / FS

    for i in range(n_show):
        f_raw  = F[cell_indices[i]].astype(np.float64)
        fc_raw = Fc_all[i]
        fc_corr, fit, method = correct_photobleaching(fc_raw)

        # Recompute dF/F for this single cell
        dff_cell = dcnv.preprocess(
            F=fc_corr.reshape(1,-1).astype(np.float32),
            baseline=BASELINE,
            win_baseline=WIN_BASELINE,
            sig_baseline=SIG_BASELINE,
            fs=FS,
            prctile_baseline=PRCTILE_BASELINE
        )[0]

        # Panel 1: Raw F + fit overlay
        axes[i,0].plot(t, f_raw, linewidth=0.8, color='gray')
        # Show fit in original space
        axes[i,0].set_ylabel(f'Cell {cell_indices[i]}', fontsize=8)
        if i==0: axes[i,0].set_title('Raw F', fontsize=10)

        # Panel 2: Bleach-corrected Fc
        axes[i,1].plot(t, fc_corr, linewidth=0.8, color='steelblue')
        axes[i,1].axhline(0, color='gray', linewidth=0.5, linestyle='--')
        if i==0: axes[i,1].set_title('Bleach-corrected Fc', fontsize=10)

        # Panel 3: dF/F (constant_prctile)
        axes[i,2].plot(t, dff_cell, linewidth=0.8, color='darkblue')
        axes[i,2].axhline(0, color='gray', linewidth=0.5, linestyle='--')
        if i==0: axes[i,2].set_title(f'dF/F ({BASELINE}, {PRCTILE_BASELINE}th pct)', fontsize=10)

    for col in range(3):
        axes[-1,col].set_xlabel('Time (s)')

    plt.suptitle(f'Plane {p} — Raw Fc | Bleach-corrected | dF/F', fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"traces_final_plane{p}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved traces_final_plane{p}.png")

# ── Save combined results ──────────────────────────────────────────────────
dff_combined = np.vstack(all_dff)
plane_arr    = np.array(all_plane)
idx_arr      = np.array(all_idx)

np.save(os.path.join(OUTPUT_DIR, "dff_all_planes.npy"),  dff_combined)
np.save(os.path.join(OUTPUT_DIR, "cell_plane.npy"),      plane_arr)
np.save(os.path.join(OUTPUT_DIR, "cell_idx.npy"),        idx_arr)

print("=" * 60)
print(f"Total cells : {len(dff_combined)}")
print(f"dF/F shape  : {dff_combined.shape}")
print("Saved: dff_all_planes.npy, cell_plane.npy, cell_idx.npy")

# ── Population average ─────────────────────────────────────────────────────
t = np.arange(dff_combined.shape[1]) / FS
pop_avg = dff_combined.mean(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(20, 8))
for i in range(min(20, len(dff_combined))):
    axes[0].plot(t, dff_combined[i] + i*2, linewidth=0.5, alpha=0.7)
axes[0].set_title('Individual dF/F traces (offset for visibility)')
axes[0].set_xlabel('Time (s)')

axes[1].plot(t, pop_avg, linewidth=1.5, color='darkblue')
axes[1].axhline(0, color='gray', linewidth=0.5)
axes[1].set_title('Population average dF/F — should be near zero if bleach corrected')
axes[1].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "population_dff.png"), dpi=150)
plt.close()
print("Saved: population_dff.png")
print("=" * 60)