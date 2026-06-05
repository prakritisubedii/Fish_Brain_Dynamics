"""
Post-processing: Photobleaching correction + dF/F computation
Session: 2026-04-20_164628, Fish A1
Author: Prakriti

Validated approach:
  1. Fit exponential decay per neuron: F(t) = A*exp(-t/tau) + C
  2. Divide by exponential fit to remove photobleaching
  3. Subtract neuropil (coefficient 0.7)
  4. Compute dF/F using short sliding window (20s) 8th percentile baseline
     appropriate for 140-second recording at 5 Hz

Saves:
  - dff_all_planes.npy : dF/F traces, shape (n_cells, n_frames)
  - cell_plane.npy     : which plane each cell belongs to
  - cell_idx.npy       : original suite2p ROI index per cell
  - traces_corrected_plane{p}.png : visualization per plane
"""

import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.ndimage import minimum_filter1d, maximum_filter1d, gaussian_filter1d
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

# dF/F parameters — tuned for 140-second recording at 5 Hz
# Short window: 20s = 100 frames — avoids removing real neural signal
WIN_BASELINE_SEC = 20.0
PRCTILE_BASELINE = 8.0
SIG_BASELINE     = 3.0   # frames, light smoothing before min filter

# ── Exponential fit ────────────────────────────────────────────────────────
def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C

def correct_photobleaching(trace):
    """
    Fit single exponential to trace and divide it out.
    Returns bleaching-corrected trace normalized to original mean.
    Falls back to linear detrend if fit fails.
    """
    n  = len(trace)
    t  = np.arange(n, dtype=np.float64)
    A0   = max(trace[0] - trace[-1], 1.0)
    tau0 = n / 2.0
    C0   = float(trace[-1])

    try:
        popt, _ = curve_fit(
            exp_decay, t, trace,
            p0=[A0, tau0, C0],
            bounds=([0, 1, 0], [np.inf, np.inf, np.inf]),
            maxfev=5000
        )
        fit = exp_decay(t, *popt)
        fit = np.maximum(fit, 1.0)
        # Divide by fit, rescale to preserve mean
        corrected = trace / fit * fit.mean()
    except Exception:
        # Linear detrend fallback
        trend     = np.linspace(float(trace[0]), float(trace[-1]), n)
        corrected = trace - trend + float(trace.mean())

    return corrected

def compute_dff(trace, fs, win_sec, prctile, sig_frames):
    """
    dF/F with short sliding window percentile baseline.
    Uses maximin: Gaussian smooth -> running min -> running max.
    """
    win_frames = max(int(win_sec * fs), 3)
    smoothed   = gaussian_filter1d(trace.astype(np.float64), sig_frames)
    baseline   = minimum_filter1d(smoothed, win_frames)
    baseline   = maximum_filter1d(baseline, win_frames)
    baseline   = np.maximum(baseline, 1.0)
    return (trace - baseline) / baseline

# ── Process each plane ─────────────────────────────────────────────────────
all_dff   = []
all_plane = []
all_idx   = []

print("=" * 60)
print("Post-processing: Photobleaching correction + dF/F")
print(f"  Sliding window: {WIN_BASELINE_SEC}s | Percentile: {PRCTILE_BASELINE}")
print("=" * 60)

for p in range(N_PLANES):
    plane_dir   = os.path.join(BASE_DIR, f"plane{p}")
    F_path      = os.path.join(plane_dir, "F.npy")
    Fneu_path   = os.path.join(plane_dir, "Fneu.npy")
    iscell_path = os.path.join(plane_dir, "iscell.npy")

    if not os.path.exists(F_path):
        print(f"Plane {p}: no F.npy found, skipping")
        continue

    F      = np.load(F_path)
    Fneu   = np.load(Fneu_path)
    iscell = np.load(iscell_path)

    cell_indices = np.where(iscell[:, 0] == 1)[0]
    n_cells      = len(cell_indices)
    print(f"Plane {p}: {n_cells} cells, {F.shape[1]} frames")

    dff_plane = []

    for idx in cell_indices:
        f    = F[idx].astype(np.float64)
        fneu = Fneu[idx].astype(np.float64)

        # Step 1: Correct photobleaching on F and Fneu separately
        f_corr    = correct_photobleaching(f)
        fneu_corr = correct_photobleaching(fneu)

        # Step 2: Neuropil subtraction
        fc = f_corr - NEUROPIL_COEFF * fneu_corr
        fc = np.maximum(fc, 1.0)

        # Step 3: dF/F with short sliding window
        dff = compute_dff(fc, FS, WIN_BASELINE_SEC, PRCTILE_BASELINE, SIG_BASELINE)

        dff_plane.append(dff)
        all_plane.append(p)
        all_idx.append(idx)

    dff_plane = np.array(dff_plane)
    all_dff.append(dff_plane)

    # ── Visualization: show raw F, corrected F, and dF/F side by side ─────
    n_show = min(5, n_cells)
    fig, axes = plt.subplots(n_show, 3, figsize=(24, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, 3)

    t = np.arange(F.shape[1]) / FS

    for i in range(n_show):
        idx    = cell_indices[i]
        f_raw  = F[idx].astype(np.float64)
        f_corr = correct_photobleaching(f_raw)
        fneu_c = correct_photobleaching(Fneu[idx].astype(np.float64))
        fc     = np.maximum(f_corr - NEUROPIL_COEFF * fneu_c, 1.0)
        dff    = compute_dff(fc, FS, WIN_BASELINE_SEC, PRCTILE_BASELINE, SIG_BASELINE)

        # Raw F
        axes[i, 0].plot(t, f_raw, linewidth=0.8, color='gray')
        axes[i, 0].set_ylabel(f'Cell {idx}', fontsize=8)
        if i == 0:
            axes[i, 0].set_title('Raw F', fontsize=10)

        # Corrected F (bleaching removed)
        axes[i, 1].plot(t, f_corr, linewidth=0.8, color='steelblue')
        if i == 0:
            axes[i, 1].set_title('Bleach-corrected F', fontsize=10)

        # dF/F
        axes[i, 2].plot(t, dff, linewidth=0.8, color='darkblue')
        axes[i, 2].axhline(0, color='gray', linewidth=0.5, linestyle='--')
        axes[i, 2].set_ylim(-0.5, 3.0)
        if i == 0:
            axes[i, 2].set_title('dF/F', fontsize=10)

    for col in range(3):
        axes[-1, col].set_xlabel('Time (s)')

    plt.suptitle(f'Plane {p} — Raw | Bleach-corrected | dF/F', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"traces_corrected_plane{p}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved traces_corrected_plane{p}.png")

# ── Save combined results ──────────────────────────────────────────────────
dff_combined = np.vstack(all_dff)
plane_arr    = np.array(all_plane)
idx_arr      = np.array(all_idx)

np.save(os.path.join(OUTPUT_DIR, "dff_all_planes.npy"), dff_combined)
np.save(os.path.join(OUTPUT_DIR, "cell_plane.npy"), plane_arr)
np.save(os.path.join(OUTPUT_DIR, "cell_idx.npy"), idx_arr)

print("=" * 60)
print(f"Total cells: {len(dff_combined)}")
print(f"dF/F shape : {dff_combined.shape}")
print(f"Saved: dff_all_planes.npy, cell_plane.npy, cell_idx.npy")
print("=" * 60)