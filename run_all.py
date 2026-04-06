"""
run_all.py
==========
Master computation script.  Runs all Fisher analyses and saves results
to  results.pkl  for use by plot_all.py.

Estimated runtime: 5-10 minutes on a modern laptop.

Output
------
results.pkl  -- dict containing all results:

  Single-bin results (fisher.py):
    'shallow_l100', 'shallow_l500'   -- run_cases at alpha=3.5
    'deep_l100',    'deep_l500'      -- run_cases at alpha=2.0
    'sweep_N_shallow_l100', etc.     -- N_FRB sweeps
    'sweep_l_shallow', 'sweep_l_deep'-- baseline sweeps

  Tomographic results (tomo.py), n_tomo=4:
    'tomo_shallow', 'tomo_deep'      -- run_tomo_cases

Run as:
    python run_all.py
"""

import time, pickle
import numpy as np
from cosmology import build_chi_z, build_Pmm
import fisher
import tomo

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("=" * 65)
print("  FRB DM x Timing  –  full computation")
print("=" * 65)

t_start = time.time()
print("\nBuilding background and P_mm (CAMB, ~20s) ...")
chi_of_z, z_of_chi = build_chi_z()
Pmm, k_mpc         = build_Pmm()
print(f"  Done in {time.time()-t_start:.1f}s")

results = {}

# ---------------------------------------------------------------------------
# Single-bin Fisher: fiducial parameter point
# ---------------------------------------------------------------------------
surveys   = [(3.5, 'shallow'), (2.0, 'deep')]
baselines = [100., 500.]

for alpha, survey in surveys:
    for l_AU in baselines:
        key = f"{survey}_l{int(l_AU)}"
        print(f"\n{'─'*55}")
        print(f"  Fiducial: {survey}, l={l_AU:.0f} AU, N=1e4")
        r = fisher.run_cases(alpha, Pmm, z_of_chi,
                              l_AU=l_AU, N_FRB=1e4)
        results[key] = r
        print(f"  sA={r['sA']:.2f}  sB={r['sB']:.2f}  sC={r['sC']:.2f}")
        print(f"  sb_no_cross={r['sb_no_cross']:.4f}  "
              f"sb_with_cross={r['sb_with_cross']:.4f}  "
              f"sb_ratio={r['sb_ratio']:.4f}")

# ---------------------------------------------------------------------------
# N_FRB sweeps
# ---------------------------------------------------------------------------
N_arr = np.logspace(3, 5, 18)

for alpha, survey in surveys:
    for l_AU in baselines:
        key = f"sweep_N_{survey}_l{int(l_AU)}"
        print(f"\n{'─'*55}")
        print(f"  N sweep: {survey}, l={l_AU:.0f} AU")
        r = fisher.sweep_NFRB(alpha, Pmm, z_of_chi,
                               l_AU=l_AU, N_arr=N_arr)
        results[key] = r

# ---------------------------------------------------------------------------
# Baseline sweep
# ---------------------------------------------------------------------------
l_arr = np.logspace(0.3, 3.0, 16)   # ~2 to 1000 AU

for alpha, survey in surveys:
    key = f"sweep_l_{survey}"
    print(f"\n{'─'*55}")
    print(f"  Baseline sweep: {survey}")
    r = fisher.sweep_baseline(alpha, Pmm, z_of_chi,
                               N_FRB=1e4, l_arr=l_arr)
    results[key] = r

# ---------------------------------------------------------------------------
# Tomographic Fisher (n_tomo=4)
# ---------------------------------------------------------------------------
for alpha, survey in surveys:
    key = f"tomo_{survey}"
    print(f"\n{'─'*55}")
    print(f"  Tomographic (n_tomo=4): {survey}, l=100 AU, N=1e4")
    r = tomo.run_tomo_cases(alpha, Pmm, z_of_chi,
                             n_tomo=4, l_AU=100., N_FRB=1e4)
    results[key] = r
    print(f"  Tomo sA={r['sA']:.2f}  sB={r['sB']:.2f}  sC={r['sC']:.2f}")
    print(f"  Tomo sb_B={r['sb_B']:.4f}  sb_C={r['sb_C']:.4f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
outfile = '/home/sim/Documents/frb_pipeline/results.pkl'
with open(outfile, 'wb') as f:
    pickle.dump(results, f)

print(f"\n{'='*65}")
print(f"  All done in {time.time()-t_start:.0f}s")
print(f"  Results saved to {outfile}")
print(f"  Keys: {list(results.keys())}")

# Print summary table
print("\n  ┌────────────────────────────────────────────────────────┐")
print("  │  Summary: sigma(fNL) at N=1e4, l=100 AU              │")
print("  ├──────────────┬────────┬────────┬────────┬────────────┤")
print("  │  Survey      │   sA   │   sB   │   sC   │  sB/sC     │")
print("  ├──────────────┼────────┼────────┼────────┼────────────┤")
for survey in ['shallow','deep']:
    r = results[f'{survey}_l100']
    print(f"  │  {survey:<12} │ {r['sA']:6.1f} │ {r['sB']:6.1f} │ {r['sC']:6.1f} │  {r['sB']/r['sC']:6.2f}x    │")
print("  ├──────────────┼────────┼────────┼────────┼────────────┤")
for survey in ['shallow','deep']:
    r = results[f'tomo_{survey}']
    print(f"  │  {survey:<5}(tomo) │ {r['sA']:6.1f} │ {r['sB']:6.1f} │ {r['sC']:6.1f} │  {r['sB']/r['sC']:6.2f}x    │")
print("  └──────────────┴────────┴────────┴────────┴────────────┘")
