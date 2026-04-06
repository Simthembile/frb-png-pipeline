"""
pipeline.py
===========
FRB Dispersion Measure x Timing: Primordial Non-Gaussianity Fisher Pipeline
============================================================================

Title: Constraining Primordial Non-Gaussianity with Fast Radio Burst
       Dispersion Measure and Timing Cross-Spectra

This pipeline runs the full analysis in four sequential stages:

  Stage 1 — Cosmology setup      (CAMB P_mm, chi(z) interpolators)
  Stage 2 — Fisher forecasts      (single-bin: fiducial + sweeps)
  Stage 3 — Tomographic Fisher    (n_tomo=4 redshift bins)
  Stage 4 — Figures               (5 publication-quality plots)

Usage
-----
    python pipeline.py [--skip-plots] [--outdir <dir>]

Outputs
-------
    results.pkl         All Fisher results
    fig1_spectra.png    C^DD, C^tt, C^Dt vs ell
    fig2_sigma_be.png   sigma(b0e) vs N_FRB
    fig3_sigma_fNL_bar.png  Bar chart of sigma(fNL) cases
    fig4_sigma_fNL_vs_N.png sigma(fNL) vs N_FRB
    fig5_lmin_vs_N.png  Minimum baseline vs N_FRB
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="FRB DM x Timing — Primordial Non-Gaussianity Fisher Pipeline"
    )
    p.add_argument(
        '--skip-plots', action='store_true',
        help='Skip figure generation (Stage 4)'
    )
    p.add_argument(
        '--outdir', default='.', metavar='DIR',
        help='Output directory for results.pkl and figures (default: current dir)'
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def banner(text):
    line = "=" * 65
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")

def section(text):
    print(f"\n  {'─'*55}")
    print(f"  {text}")
    print(f"  {'─'*55}")

def done(t0, label=""):
    dt = time.time() - t0
    tag = f" [{label}]" if label else ""
    print(f"  ✓ Done in {dt:.1f}s{tag}")

# ---------------------------------------------------------------------------
# Stage 1: Cosmology
# ---------------------------------------------------------------------------

def stage1_cosmology():
    section("Stage 1 — Cosmology setup  (CAMB P_mm, chi-z interpolators)")
    t0 = time.time()
    from cosmology import build_chi_z, build_Pmm
    chi_of_z, z_of_chi = build_chi_z()
    Pmm, k_mpc         = build_Pmm()
    done(t0, "CAMB + interpolators")
    print(f"    chi(z=1) = {chi_of_z(1.0):.1f} Mpc  |  "
          f"P_mm(k=0.1, z=0) = {float(Pmm(0.1, 0.0)):.3e} Mpc³")
    return chi_of_z, z_of_chi, Pmm, k_mpc

# ---------------------------------------------------------------------------
# Stage 2: Single-bin Fisher (fiducial + sweeps)
# ---------------------------------------------------------------------------

def stage2_fisher(Pmm, z_of_chi):
    section("Stage 2 — Single-bin Fisher forecasts")
    import fisher

    results  = {}
    surveys  = [(3.5, 'shallow'), (2.0, 'deep')]
    baselines = [100., 500.]
    N_arr    = np.logspace(3, 5, 18)
    l_arr    = np.logspace(0.3, 3.0, 16)      # ~2–1000 AU

    # --- Fiducial ---
    print("\n  [2a] Fiducial  N=1e4")
    for alpha, survey in surveys:
        for l_AU in baselines:
            key = f"{survey}_l{int(l_AU)}"
            t0  = time.time()
            r   = fisher.run_cases(alpha, Pmm, z_of_chi, l_AU=l_AU, N_FRB=1e4)
            results[key] = r
            print(f"    {key:<18}  "
                  f"sA={r['sA']:7.1f}  sB={r['sB']:7.1f}  sC={r['sC']:7.1f}  "
                  f"({time.time()-t0:.1f}s)")

    # --- N_FRB sweeps ---
    print("\n  [2b] N_FRB sweeps")
    for alpha, survey in surveys:
        for l_AU in baselines:
            key = f"sweep_N_{survey}_l{int(l_AU)}"
            t0  = time.time()
            r   = fisher.sweep_NFRB(alpha, Pmm, z_of_chi, l_AU=l_AU, N_arr=N_arr)
            results[key] = r
            print(f"    {key:<28}  ({time.time()-t0:.1f}s)")

    # --- Baseline sweeps ---
    print("\n  [2c] Baseline sweeps")
    for alpha, survey in surveys:
        key = f"sweep_l_{survey}"
        t0  = time.time()
        r   = fisher.sweep_baseline(alpha, Pmm, z_of_chi, N_FRB=1e4, l_arr=l_arr)
        results[key] = r
        print(f"    {key:<28}  ({time.time()-t0:.1f}s)")

    return results

# ---------------------------------------------------------------------------
# Stage 3: Tomographic Fisher
# ---------------------------------------------------------------------------

def stage3_tomo(Pmm, z_of_chi, results):
    section("Stage 3 — Tomographic Fisher  (n_tomo = 4)")
    import tomo

    surveys = [(3.5, 'shallow'), (2.0, 'deep')]
    for alpha, survey in surveys:
        key = f"tomo_{survey}"
        t0  = time.time()
        r   = tomo.run_tomo_cases(alpha, Pmm, z_of_chi,
                                   n_tomo=4, l_AU=100., N_FRB=1e4)
        results[key] = r
        print(f"    {key:<18}  "
              f"sA={r['sA']:7.1f}  sB={r['sB']:7.1f}  sC={r['sC']:7.1f}  "
              f"sb_B={r['sb_B']:.4f}  sb_C={r['sb_C']:.4f}  "
              f"({time.time()-t0:.1f}s)")

    return results

# ---------------------------------------------------------------------------
# Stage 4: Figures
# ---------------------------------------------------------------------------

def stage4_figures(outdir):
    section("Stage 4 — Generating publication figures")
    import plot_all   # plot_all.py reads results.pkl and saves figures
    print(f"    Figures written to: {os.path.abspath(outdir)}/")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results):
    banner("Summary: sigma(fNL) at N=1e4, l=100 AU")
    header = f"  {'Survey':<16} {'sA':>8} {'sB':>8} {'sC':>8} {'sB/sC':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, key in [('shallow',      'shallow_l100'),
                        ('deep',         'deep_l100'),
                        ('shallow (tomo)','tomo_shallow'),
                        ('deep (tomo)',   'tomo_deep')]:
        r = results[key]
        sA = r['sA']
        sB = r['sB']
        sC = r['sC']
        ratio = sB / sC
        print(f"  {label:<16} {sA:>8.1f} {sB:>8.1f} {sC:>8.1f} {ratio:>7.2f}x")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    t_total = time.time()
    banner("FRB DM × Timing — Primordial Non-Gaussianity Fisher Pipeline")
    print(f"  Output directory : {os.path.abspath(outdir)}")
    print(f"  Skip plots       : {args.skip_plots}")

    # Add pipeline directory to path so modules resolve correctly
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # --- Stage 1 ---
    chi_of_z, z_of_chi, Pmm, k_mpc = stage1_cosmology()

    # --- Stage 2 ---
    results = stage2_fisher(Pmm, z_of_chi)

    # --- Stage 3 ---
    results = stage3_tomo(Pmm, z_of_chi, results)

    # --- Save results ---
    pkl_path = os.path.join(outdir, 'results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  Results saved → {pkl_path}")

    # --- Stage 4 ---
    if not args.skip_plots:
        stage4_figures(outdir)
    else:
        print("\n  [Stage 4 skipped — use plot_all.py to generate figures]")

    # --- Summary ---
    print_summary(results)

    banner(f"Pipeline complete  ({time.time()-t_total:.0f}s total)")


if __name__ == '__main__':
    main()
