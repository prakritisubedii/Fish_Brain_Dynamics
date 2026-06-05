"""
Post-processing: Photobleaching correction + dF/F
Session: 2026-04-20_164628, Fish A1
Author: Prakriti

Validated approach based on:
  - PLOS Comp Bio (TMAC): shared time constant, divide by fit
  - Nature (GCaMP8M paper): fit beginning and end segments only
  - suite2p docs: constant_prctile for final dF/F

Steps per plane:
  1. Neuropil subtraction: Fc = F - 0.7*Fneu
  2. Compute POPULATION mean Fc (average across all cells)
     - Fit double exponential to beginning + end segments of population mean
     - This gives a SHARED bleaching curve unaffected by individual cell activity
  3. DIVIDE each cell's Fc by the shared bleaching fit
     - Division (not subtraction) gives proper dF normalization
     - Result is Fc_norm ≈ 1.0 at start, fluctuates around 1.0
  4. dF/F = (Fc_norm - F0) / F0 using suite2p's constant_prctile (8th pct)
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
FS             = 5.0
NEUROPIL_COEFF = 0.7
PRCTILE_BASELINE = 8.0

# Fraction of trace to use for fitting (beginning and end segments)
# Avoids fitting the middle where neural activity is strongest
FIT_FRAC = 0.15  # use first 15% and last 15% of frames

# ── Double exponential on population mean ──────────────────────────────────
def double_exp(t, A1, tau1, A2, tau2, C):
    return A1 * np.exp(-t/tau1) + A2 * np.exp(-t/tau2) + C

def fit_population_bleaching(pop_mean):
    """
    Fit double exponential to beginning + end segments of population mean.
    Using beginning and end avoids fitting neural activity in the middle.
    Returns the fitted bleaching curve for the full trace.
    """
    n = len(pop_mean)
    t = np.arange(n, dtype=np.float64)

    # Use only beginning and end segments for fitting
    n_seg = max(int(n * FIT_FRAC), 5)
    idx_fit = np.concatenate([np.arange(n_seg), np.arange(n - n_seg, n)])
    t_fit   = t[idx_fit]
    f_fit   = pop_mean[idx_fit]

    drop = float(pop_mean[0] - pop_mean[-1])
    C0   = float(pop_mean[-1])

    try:
        popt, _ = curve_fit(
            double_exp, t_fit, f_fit,
            p0=[drop*0.4, n*0.05, drop*0.6, n*0.4, C0],
            bounds=([0, 0.5, 0, n*0.05, 0],
                    [np.inf, n*0.25, np.inf, np.inf, np.inf]),
            maxfev=20000, method='trf'
        )
        fit_full = double_exp(t, *popt)
        print(f"  Double-exp fit: A1={popt[0]:.1f} tau1={popt[1]:.1f}f "
              f"A2={popt[2]:.1f} tau2={popt[3]:.1f}f C={popt[4]:.1f}")
        return fit_full, 'double_exp'
    except Exception as e:
        print(f"  Double-exp failed ({e}), trying single-exp...")

    try:
        def single_exp(t, A, tau, C):
            return A * np.exp(-t/tau) + C
        popt2, _ = curve_fit(
            single_exp, t_fit, f_fit,
            p0=[drop, n/3, C0],
            bounds=([0, 1, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000
        )
        fit_full = single_exp(t, *popt2)
        print(f"  Single-exp fit: A={popt2[0]:.1f} tau={popt2[1]:.1f}f C={popt2[2]:.1f}")
        return fit_full, 'single_exp'
    except Exception as e:
        print(f"  Single-exp failed ({e}), using linear...")
        fit_full = np.linspace(float(pop_mean[0]), float(pop_mean[-1]), n)
        return fit_full, 'linear'

# ── Process each plane ─────────────────────────────────────────────────────
all_dff   = []
all_plane = []
all_idx   = []

print("=" * 60)
print("Post-processing: Shared bleaching fit + divide + constant_prctile dF/F")
print(f"  Fit segments  : first+last {FIT_FRAC*100:.0f}% of frames")
print(f"  Correction    : divide by shared population fit")
print(f"  dF/F baseline : constant {PRCTILE_BASELINE}th percentile")
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
    print(f"\nPlane {p}: {n_cells} cells, {n_frames} frames")

    # Step 1: Neuropil subtraction for all cells
    Fc_all = (F[cell_indices] - NEUROPIL_COEFF * Fneu[cell_indices]).astype(np.float64)

    # Step 2: Fit bleaching to population mean (beginning + end segments)
    pop_mean = Fc_all.mean(axis=0)
    bleach_fit, fit_method = fit_population_bleaching(pop_mean)

    # Clamp fit to be >= 1 to avoid division by near-zero
    bleach_fit = np.maximum(bleach_fit, 1.0)

    # Step 3: Divide each cell by the shared bleaching fit
    # Result: Fc_norm fluctuates around 1.0, bleaching removed
    Fc_norm = Fc_all / bleach_fit[np.newaxis, :]

    # Step 4: dF/F using suite2p's constant_prctile
    # constant_prctile subtracts the 8th percentile baseline
    # We then divide by it to get proper fractional dF/F
    F0 = np.percentile(Fc_norm, PRCTILE_BASELINE, axis=1, keepdims=True)
    F0 = np.maximum(F0, 1e-6)
    dff = (Fc_norm - F0) / F0

    all_dff.append(dff.astype(np.float32))
    for idx in cell_indices:
        all_plane.append(p)
        all_idx.append(idx)

    print(f"  dF/F stats: mean={dff.mean():.3f}, std={dff.std():.3f}, "
          f"min={dff.min():.2f}, max={dff.max():.2f}")

    # ── Visualization ──────────────────────────────────────────────────────
    n_show = min(5, n_cells)
    fig, axes = plt.subplots(n_show, 3, figsize=(24, 3*n_show))
    if n_show == 1:
        axes = axes.reshape(1, 3)

    t = np.arange(n_frames) / FS

    for i in range(n_show):
        fc   = Fc_all[i]
        fnorm = Fc_norm[i]
        dff_i = dff[i]

        # Panel 1: Raw Fc + population bleach fit overlay
        axes[i,0].plot(t, fc, linewidth=0.8, color='gray', label='Fc')
        axes[i,0].plot(t, bleach_fit * (fc.mean() / bleach_fit.mean()),
                       linewidth=2, color='red', alpha=0.8, label='Bleach fit')
        axes[i,0].set_ylabel(f'Cell {cell_indices[i]}', fontsize=8)
        if i==0:
            axes[i,0].set_title('Raw Fc + shared bleach fit', fontsize=10)
            axes[i,0].legend(fontsize=7)

        # Panel 2: Bleach-corrected (divided)
        axes[i,1].plot(t, fnorm, linewidth=0.8, color='steelblue')
        axes[i,1].axhline(1.0, color='gray', linewidth=0.5, linestyle='--')
        if i==0:
            axes[i,1].set_title('Bleach-corrected (÷ fit)', fontsize=10)

        # Panel 3: dF/F
        axes[i,2].plot(t, dff_i, linewidth=0.8, color='darkblue')
        axes[i,2].axhline(0, color='gray', linewidth=0.5, linestyle='--')
        axes[i,2].set_ylim(-1, 3)
        if i==0:
            axes[i,2].set_title(f'dF/F (8th pct baseline)', fontsize=10)

    for col in range(3):
        axes[-1,col].set_xlabel('Time (s)')

    # Also show population mean + fit in top of panel 1
    plt.suptitle(f'Plane {p} — Raw Fc | Bleach-corrected | dF/F\n'
                 f'Bleach fit method: {fit_method}', fontsize=11)
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

print("\n" + "=" * 60)
print(f"Total cells : {len(dff_combined)}")
print(f"dF/F shape  : {dff_combined.shape}")
print(f"dF/F range  : {dff_combined.min():.3f} to {dff_combined.max():.3f}")
print("Saved: dff_all_planes.npy, cell_plane.npy, cell_idx.npy")

# ── Population average ─────────────────────────────────────────────────────
t = np.arange(dff_combined.shape[1]) / FS
pop_avg = dff_combined.mean(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(20, 8))
for i in range(min(20, len(dff_combined))):
    axes[0].plot(t, dff_combined[i] + i*0.5,
                 linewidth=0.5, alpha=0.7)
axes[0].set_title('Individual dF/F traces (offset for visibility)')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('dF/F')

axes[1].plot(t, pop_avg, linewidth=1.5, color='darkblue')
axes[1].axhline(0, color='gray', linewidth=0.5)
axes[1].set_title('Population average dF/F — flat = bleach corrected ✓')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('dF/F')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "population_dff.png"), dpi=150)
plt.close()
print("Saved: population_dff.png")
print("=" * 60)