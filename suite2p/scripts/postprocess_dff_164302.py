"""
Post-processing: Photobleaching correction + dF/F
Session: 2026-04-20_164628, Fish A1
Author: Prakriti

Validated approach (Nature Methods, PLOS Comp Bio):
  1. Fit DOUBLE exponential per neuron: F(t) = A1*exp(-t/tau1) + A2*exp(-t/tau2) + C
     - Fast component (tau1): captures initial rapid bleaching (~first 10s)
     - Slow component (tau2): captures long-term bleaching
  2. SUBTRACT the fit (not divide) — preserves fluctuation amplitudes
     correctly when bleaching dominates baseline
  3. Subtract neuropil (coefficient 0.7)
  4. Z-score the corrected traces — valid for spontaneous activity analysis,
     avoids sliding window baseline problem for short recordings

Three-panel visualization per plane:
  Raw F | Double-exp corrected F | Z-scored dF/F
"""

import numpy as np
import os
from scipy.optimize import curve_fit
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

# ── Double exponential fit ─────────────────────────────────────────────────
def double_exp(t, A1, tau1, A2, tau2, C):
    """
    Double exponential decay:
    F(t) = A1*exp(-t/tau1) + A2*exp(-t/tau2) + C
    Fast component (tau1) + slow component (tau2) + constant offset
    """
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C

def correct_photobleaching_double_exp(trace):
    """
    Fit double exponential to trace and SUBTRACT it.
    Subtraction is correct when bleaching dominates baseline
    (preserves fluctuation amplitudes without noise amplification).
    Returns corrected trace centered at zero mean.
    Falls back to single exponential, then linear detrend if fits fail.
    """
    n = len(trace)
    t = np.arange(n, dtype=np.float64)
    
    trace_f = trace.astype(np.float64)
    total_drop = float(trace_f[0] - trace_f[-1])
    
    # Initial guesses for double exponential
    A1_0   = total_drop * 0.5        # fast component amplitude
    tau1_0 = n * 0.05                # fast time constant (~5% of recording)
    A2_0   = total_drop * 0.5        # slow component amplitude  
    tau2_0 = n * 0.5                 # slow time constant (~50% of recording)
    C0     = float(trace_f[-1])      # asymptotic offset
    
    try:
        popt, _ = curve_fit(
            double_exp, t, trace_f,
            p0=[A1_0, tau1_0, A2_0, tau2_0, C0],
            bounds=([0, 0.5, 0, n*0.05, 0],
                    [np.inf, n*0.2, np.inf, np.inf, np.inf]),
            maxfev=10000,
            method='trf'
        )
        fit = double_exp(t, *popt)
        # Subtract bleaching, preserve fluctuations around zero
        corrected = trace_f - fit
        return corrected, fit, 'double_exp'
        
    except Exception:
        # Fallback: single exponential
        try:
            from scipy.optimize import curve_fit as cf
            def single_exp(t, A, tau, C):
                return A * np.exp(-t/tau) + C
            popt2, _ = cf(single_exp, t, trace_f,
                         p0=[total_drop, n/3, float(trace_f[-1])],
                         bounds=([0, 1, 0], [np.inf, np.inf, np.inf]),
                         maxfev=5000)
            fit = single_exp(t, *popt2)
            corrected = trace_f - fit
            return corrected, fit, 'single_exp'
        except Exception:
            # Final fallback: linear detrend
            fit = np.linspace(float(trace_f[0]), float(trace_f[-1]), n)
            corrected = trace_f - fit
            return corrected, fit, 'linear'

def zscore(trace):
    """Z-score: subtract mean, divide by std."""
    mu  = trace.mean()
    std = trace.std()
    if std < 1e-6:
        return np.zeros_like(trace)
    return (trace - mu) / std

# ── Process each plane ─────────────────────────────────────────────────────
all_corrected = []
all_zscored   = []
all_plane     = []
all_idx       = []

print("=" * 60)
print("Post-processing: Double-exp photobleaching correction + z-score")
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

    corrected_plane = []
    zscored_plane   = []
    fit_methods     = {'double_exp': 0, 'single_exp': 0, 'linear': 0}

    for idx in cell_indices:
        f    = F[idx].astype(np.float64)
        fneu = Fneu[idx].astype(np.float64)

        # Correct bleaching on F and Fneu separately
        f_corr,    _, method = correct_photobleaching_double_exp(f)
        fneu_corr, _, _      = correct_photobleaching_double_exp(fneu)
        fit_methods[method] += 1

        # Neuropil subtraction
        fc = f_corr - NEUROPIL_COEFF * fneu_corr

        # Z-score
        fz = zscore(fc)

        corrected_plane.append(fc)
        zscored_plane.append(fz)
        all_plane.append(p)
        all_idx.append(idx)

    print(f"  Fit methods: {fit_methods}")

    corrected_plane = np.array(corrected_plane)
    zscored_plane   = np.array(zscored_plane)
    all_corrected.append(corrected_plane)
    all_zscored.append(zscored_plane)

    # ── Visualization: 5 cells, 3 panels each ─────────────────────────────
    n_show = min(5, n_cells)
    fig, axes = plt.subplots(n_show, 3, figsize=(24, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, 3)

    t = np.arange(F.shape[1]) / FS

    for i in range(n_show):
        idx    = cell_indices[i]
        f_raw  = F[idx].astype(np.float64)
        fneu_r = Fneu[idx].astype(np.float64)

        f_corr, fit, method = correct_photobleaching_double_exp(f_raw)
        fneu_c, _,  _       = correct_photobleaching_double_exp(fneu_r)
        fc = f_corr - NEUROPIL_COEFF * fneu_c
        fz = zscore(fc)

        # Panel 1: Raw F + exponential fit overlay
        axes[i, 0].plot(t, f_raw, linewidth=0.8, color='gray', label='Raw F')
        axes[i, 0].plot(t, fit + (f_raw - f_corr).mean() + f_corr.mean(),
                        linewidth=1.5, color='red', alpha=0.8, label=f'Fit ({method})')
        axes[i, 0].set_ylabel(f'Cell {idx}', fontsize=8)
        if i == 0:
            axes[i, 0].set_title('Raw F + bleach fit', fontsize=10)
            axes[i, 0].legend(fontsize=7)

        # Panel 2: Bleach-corrected F
        axes[i, 1].plot(t, fc, linewidth=0.8, color='steelblue')
        axes[i, 1].axhline(0, color='gray', linewidth=0.5, linestyle='--')
        if i == 0:
            axes[i, 1].set_title('Bleach-corrected F (subtraction)', fontsize=10)

        # Panel 3: Z-scored
        axes[i, 2].plot(t, fz, linewidth=0.8, color='darkblue')
        axes[i, 2].axhline(0, color='gray', linewidth=0.5, linestyle='--')
        axes[i, 2].set_ylim(-4, 4)
        if i == 0:
            axes[i, 2].set_title('Z-scored (neural activity)', fontsize=10)

    for col in range(3):
        axes[-1, col].set_xlabel('Time (s)')

    plt.suptitle(f'Plane {p} — Raw | Double-exp corrected | Z-scored', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"traces_doubleexp_plane{p}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved traces_doubleexp_plane{p}.png")

# ── Save combined results ──────────────────────────────────────────────────
zscored_combined   = np.vstack(all_zscored)
corrected_combined = np.vstack(all_corrected)
plane_arr          = np.array(all_plane)
idx_arr            = np.array(all_idx)

np.save(os.path.join(OUTPUT_DIR, "zscored_all_planes.npy"),   zscored_combined)
np.save(os.path.join(OUTPUT_DIR, "corrected_all_planes.npy"), corrected_combined)
np.save(os.path.join(OUTPUT_DIR, "cell_plane.npy"),           plane_arr)
np.save(os.path.join(OUTPUT_DIR, "cell_idx.npy"),             idx_arr)

print("=" * 60)
print(f"Total cells: {len(zscored_combined)}")
print(f"Z-scored shape: {zscored_combined.shape}")
print(f"Saved: zscored_all_planes.npy, corrected_all_planes.npy")
print(f"Check traces_doubleexp_plane*.png")
print("=" * 60)

# ── Population average z-scored ────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t = np.arange(zscored_combined.shape[1]) / FS
pop_avg = zscored_combined.mean(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(20, 8))
for i in range(min(20, len(zscored_combined))):
    axes[0].plot(t, zscored_combined[i] + i*3,
                 linewidth=0.5, alpha=0.7)
axes[0].set_title('Individual cell z-scored traces (offset for visibility)')
axes[0].set_xlabel('Time (s)')

axes[1].plot(t, pop_avg, linewidth=1.5, color='darkblue')
axes[1].axhline(0, color='gray', linewidth=0.5)
axes[1].set_title('Population average (all cells, z-scored) — should be flat if bleach corrected')
axes[1].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "population_zscored.png"), dpi=150)
plt.close()
print("Saved population_zscored.png")