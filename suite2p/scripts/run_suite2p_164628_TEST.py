"""
Suite2p TEST run script for zebrafish GCaMP8M data
Session: 2026-04-20_164628, Fish A1
Author: Prakriti

Uses first 700 frames (100 volumes x 7 planes) to verify 7-plane
pipeline works before full run.

Full run script: run_suite2p_164628.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import h5py
import numpy as np
import math
import time
import suite2p
import suite2p.io

# ── Paths ──────────────────────────────────────────────────────────────────
H5_PATH     = "/home/abl-workstation2/fishdynamics_data/20iv26/2026-04-20_164628/raw/stack_1-A1-GCaMP8M_channel_1_obj_bottom/Cam_long_00000.lux.h5"
OUTPUT_DIR  = "/home/abl-workstation2/Prakriti_FishBrain/suite2p/results/164628_TEST"
SCRATCH_DIR = "/home/abl-workstation2/Prakriti_FishBrain/suite2p/results/164628_TEST/scratch"

# ── Acquisition parameters ─────────────────────────────────────────────────
N_PLANES      = 7      # Z planes per volume
FS_VOLUME     = 5.0    # Hz volumetric rate
TAU           = 1.5    # GCaMP8M decay time in seconds
N_VOLS_TEST   = 100    # number of volumes to use for test
N_FRAMES_TEST = N_VOLS_TEST * N_PLANES  # = 700 frames

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

# ── Step 1: Compute per-plane column mean from test frames ─────────────────
print("=" * 60)
print(f"TEST RUN — {N_VOLS_TEST} volumes x {N_PLANES} planes = {N_FRAMES_TEST} frames")
print("=" * 60)
print("Step 1: Computing per-plane column means for stripe removal...")
t0 = time.time()

with h5py.File(H5_PATH, "r") as f:
    Ly = f["Data"].shape[1]
    Lx = f["Data"].shape[2]
    frames = f["Data"][:N_FRAMES_TEST].astype(np.float32)

col_means = np.zeros((N_PLANES, Ly, Lx), dtype=np.float32)
for p in range(N_PLANES):
    col_means[p] = frames[p::N_PLANES].mean(axis=0)

del frames
print(f"Per-plane column means computed. Time: {time.time()-t0:.1f}s")

# Sanity check — print mean brightness per plane to confirm 7-plane structure
print("Per-plane mean brightness (should vary across planes):")
with h5py.File(H5_PATH, "r") as f:
    for p in range(N_PLANES):
        plane_mean = f["Data"][p:N_FRAMES_TEST:N_PLANES].astype(np.float32).mean()
        print(f"  Plane {p}: {plane_mean:.1f}")

# ── Step 2: Monkey-patch suite2p's h5py_to_binary ─────────────────────────
def h5py_to_binary_with_stripe_removal(dbs, settings, reg_file, reg_file_chan2):
    nplanes   = dbs[0]["nplanes"]
    nchannels = dbs[0]["nchannels"]
    h5list    = dbs[0]["file_list"]
    keys      = dbs[0]["h5py_key"]
    if isinstance(keys, str):
        keys = [keys]

    iall = 0
    for j in range(nplanes):
        dbs[j]["nframes_per_folder"] = np.zeros(len(h5list), np.int32)

    for ih5, h5 in enumerate(h5list):
        with h5py.File(h5, "r") as f:
            for key in keys:
                nframes_all = min(f[key].shape[0], N_FRAMES_TEST)  # TEST: limit frames
                nbatch      = nplanes * max(1, dbs[0]["batch_size"] // nplanes)
                nbatch      = min(nbatch, nframes_all)
                nfunc       = dbs[0]["functional_chan"] - 1 if nchannels > 1 else 0
                ik = 0

                while True:
                    irange = np.arange(ik, min(ik + nbatch, nframes_all), 1)
                    if irange.size == 0:
                        break

                    im            = f[key][irange, :, :].astype(np.float32)
                    nframes_batch = im.shape[0]

                    # Per-plane stripe removal
                    for p in range(nplanes):
                        plane_indices = np.arange(p, nframes_batch, nplanes)
                        if len(plane_indices) > 0:
                            im[plane_indices] -= col_means[p][np.newaxis, :, :]

                    im = np.clip(im, 0, None)
                    im = (im / 2).astype(np.int16)

                    for j in range(nplanes):
                        if iall == 0:
                            dbs[j]["meanImg"] = np.zeros(
                                (im.shape[1], im.shape[2]), np.float32)
                            if nchannels > 1:
                                dbs[j]["meanImg_chan2"] = np.zeros(
                                    (im.shape[1], im.shape[2]), np.float32)
                            dbs[j]["nframes"] = 0

                        plane_indices = np.arange(j, nframes_batch, nplanes)
                        if len(plane_indices) > 0:
                            im2write = im[plane_indices].astype(np.int16)
                            reg_file[j].write(bytearray(im2write))
                            dbs[j]["meanImg"] += im2write.astype(np.float32).sum(axis=0)
                            dbs[j]["nframes"] += im2write.shape[0]
                            dbs[j]["nframes_per_folder"][ih5] += im2write.shape[0]

                    ik   += nframes_batch
                    iall += nframes_batch

    do_registration = settings["run"]["do_registration"]
    for db in dbs:
        db["Ly"] = im2write.shape[1]
        db["Lx"] = im2write.shape[2]
        if not do_registration:
            db["yrange"] = np.array([0, db["Ly"]])
            db["xrange"] = np.array([0, db["Lx"]])
        db["meanImg"] /= db["nframes"]
        if nchannels > 1:
            db["meanImg_chan2"] /= db["nframes"]
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)

    return dbs

suite2p.io.h5py_to_binary = h5py_to_binary_with_stripe_removal
print("Step 2: Stripe removal patch applied.")

# ── Step 3: Suite2p settings ───────────────────────────────────────────────
settings = suite2p.default_settings()

settings['fs']           = FS_VOLUME
settings['diameter']     = [20, 20]
settings['tau']          = TAU
settings['torch_device'] = 'cuda:0'

settings['run']['do_registration']  = 1
settings['run']['do_detection']     = True
settings['run']['do_deconvolution'] = True

settings['registration']['nimg_init']   = 50    # reduced for test (only 100 vols)
settings['registration']['batch_size']  = 100
settings['registration']['maxregshift'] = 0.1
settings['registration']['block_size']  = [128, 128]

settings['detection']['algorithm']         = 'sparsery'
settings['detection']['threshold_scaling'] = 1.0
settings['detection']['max_overlap']       = 0.75
settings['detection']['highpass_time']     = 100
settings['detection']['soma_crop']         = True

settings['classification']['use_builtin_classifier'] = True

# ── Step 4: Run suite2p ────────────────────────────────────────────────────
db = {
    'data_path':           [os.path.dirname(H5_PATH)],
    'file_list':           [H5_PATH],
    'look_one_level_down': False,
    'input_format':        'h5',
    'h5py_key':            'Data',
    'nplanes':             N_PLANES,
    'nchannels':           1,
    'save_path0':          OUTPUT_DIR,
    'fast_disk':           SCRATCH_DIR,
    'nframes':             N_FRAMES_TEST // N_PLANES,
}

print("=" * 60)
print("Step 3: Starting suite2p TEST run...")
print(f"  Volumes      : {N_VOLS_TEST} (of 700 total)")
print(f"  Planes       : {N_PLANES}")
print(f"  Frames       : {N_FRAMES_TEST}")
print(f"  fs (volume)  : {settings['fs']} Hz")
print(f"  Diameter     : {settings['diameter']} px")
print(f"  Device       : {settings['torch_device']}")
print("=" * 60)
t1 = time.time()

output_ops = suite2p.run_s2p(db=db, settings=settings)

total = time.time() - t1
print("=" * 60)
print(f"TEST RUN complete! Time: {total/60:.1f} minutes")
print(f"Results saved to: {OUTPUT_DIR}")

# ── Step 5: Summary across all planes ─────────────────────────────────────
print("\nResults per plane:")
total_rois  = 0
total_cells = 0
for p in range(N_PLANES):
    plane_dir   = os.path.join(OUTPUT_DIR, "suite2p", f"plane{p}")
    stat_path   = os.path.join(plane_dir, "stat.npy")
    iscell_path = os.path.join(plane_dir, "iscell.npy")
    if os.path.exists(stat_path) and os.path.exists(iscell_path):
        stat    = np.load(stat_path, allow_pickle=True)
        iscell  = np.load(iscell_path)
        n_rois  = len(stat)
        n_cells = int(iscell[:, 0].sum())
        total_rois  += n_rois
        total_cells += n_cells
        print(f"  Plane {p}: {n_rois} ROIs, {n_cells} cells")
    else:
        print(f"  Plane {p}: results not found — check errors above")

print(f"\nTotal across all planes: {total_rois} ROIs, {total_cells} cells")

# ── Step 6: Save visualization for all 7 planes ───────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for p in range(N_PLANES):
        plane_dir   = os.path.join(OUTPUT_DIR, "suite2p", f"plane{p}")
        ops_path    = os.path.join(plane_dir, "ops.npy")
        stat_path   = os.path.join(plane_dir, "stat.npy")
        iscell_path = os.path.join(plane_dir, "iscell.npy")

        if os.path.exists(ops_path):
            ops     = np.load(ops_path, allow_pickle=True).item()
            stat    = np.load(stat_path, allow_pickle=True)
            iscell  = np.load(iscell_path)
            n_cells = int(iscell[:, 0].sum())

            axes[p].imshow(ops['meanImg'], cmap='gray',
                           vmin=np.percentile(ops['meanImg'], 1),
                           vmax=np.percentile(ops['meanImg'], 99))
            for i, s in enumerate(stat):
                if iscell[i, 0] == 1:
                    axes[p].plot(s['med'][1], s['med'][0], 'r.', markersize=1.5)
            axes[p].set_title(f"Plane {p} — {n_cells} cells")
            axes[p].axis('off')

    axes[7].axis('off')
    plt.suptitle("TEST — suite2p 7-plane results, session 164628", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "cell_locations_all_planes_TEST.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nVisualization saved to: {plot_path}")
    print("Open this image to verify cells appear in correct brain regions across all 7 planes.")
except Exception as e:
    print(f"\nVisualization failed (non-critical): {e}")

print("\nIf all 7 planes show results and no errors above, launch the full run.")