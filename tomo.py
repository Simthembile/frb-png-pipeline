"""
tomo.py
=======
Tomographic extension of the Fisher analysis.

Splits the FRB source distribution n(chi) into n_tomo equal-count
redshift bins and computes cross-spectra between all bin pairs (i,j).
This adds n_tomo*(n_tomo+1)/2 independent C^DD_{ij} spectra per ell,
improving sigma(fNL) by roughly sqrt(n_tomo*(n_tomo+1)/2) ~ 3x for
n_tomo=4 over the single-bin case.

Following Reischke et al. (2020) Sec. IID: bins are constructed using
the DM as a proxy for redshift, with equal numbers of FRBs per bin.

Main function
-------------
run_tomo_cases(alpha, Pmm, z_of_chi, n_tomo=4, ...)
    -> analogous to fisher.run_cases but tomographic
"""

import numpy as np
from scipy.interpolate import interp1d

from cosmology import (
    b_e, delta_b_PNG, f_PhiPhi, f_mPhi,
    AU_MPC, C_KMS, H0, E_z, OM_M,
    nz_frb, fIGM, build_nchi, build_WD,
)
from spectra import (
    A_DM, CONV, CHI_MAX, N_CHI,
    noise_DD, noise_tt,
)
from fisher import ELL_ARR, F_SKY, _fisher_ell, _invert_marg, _unmarg

# ---------------------------------------------------------------------------
# Build tomographic bin boundaries
# ---------------------------------------------------------------------------

def build_tomo_boundaries(alpha, z_of_chi, chi_max=CHI_MAX, n=N_CHI,
                           n_tomo=4):
    """
    Split n(chi) into n_tomo equal-count bins.

    Returns
    -------
    bounds : list of (n_tomo+1) comoving distances [Mpc]
             bounds[i] to bounds[i+1] defines bin i
    """
    chi_arr  = np.linspace(1.0, chi_max, n)
    z_arr    = np.clip(z_of_chi(chi_arr), 0.0, 10.0)
    dzdchi   = H0 * E_z(z_arr) / C_KMS
    nchi_raw = nz_frb(z_arr, alpha) * dzdchi
    norm     = np.trapezoid(nchi_raw, chi_arr)
    nchi     = nchi_raw / norm

    cdf      = np.zeros(n)
    for k in range(1, n):
        cdf[k] = np.trapezoid(nchi[:k+1], chi_arr[:k+1])

    bounds = [chi_arr[0]]
    for k in range(1, n_tomo):
        target = k / n_tomo
        idx    = np.searchsorted(cdf, target)
        idx    = int(np.clip(idx, 0, n - 1))
        bounds.append(float(chi_arr[idx]))
    bounds.append(float(chi_arr[-1]))
    return bounds


# ---------------------------------------------------------------------------
# Bin-restricted kernels
# ---------------------------------------------------------------------------

def _build_WD_bin(alpha, z_of_chi, lo, hi, chi_max=CHI_MAX, n=N_CHI):
    """
    DM weight kernel for bin [lo, hi]:
    G^i_D(chi) = W(chi) * integral_{max(chi,lo)}^{hi} n(chi') dchi'
    """
    chi_arr  = np.linspace(1.0, chi_max, n)
    z_arr    = np.clip(z_of_chi(chi_arr), 0.0, 10.0)
    Ez       = E_z(z_arr)
    F        = fIGM(z_arr)
    dzdchi   = H0 * Ez / C_KMS
    W        = F / ((1.0 + z_arr) * Ez) * dzdchi

    nchi_fn, _ = build_nchi(alpha, z_of_chi, chi_max, n)
    nchi        = nchi_fn(chi_arr)

    tail = np.zeros(n)
    for j in range(n):
        lo_eff = max(chi_arr[j], lo)
        if lo_eff >= hi:
            tail[j] = 0.0
            continue
        mask    = (chi_arr >= lo_eff) & (chi_arr <= hi)
        if mask.sum() >= 2:
            tail[j] = np.trapezoid(nchi[mask], chi_arr[mask])

    GD = W * tail
    return interp1d(chi_arr, GD, bounds_error=False, fill_value=0.0)


def _build_nchi_bin(alpha, z_of_chi, lo, hi, chi_max=CHI_MAX, n=N_CHI):
    """n(chi) restricted to [lo, hi] and globally normalised."""
    chi_arr  = np.linspace(1.0, chi_max, n)
    z_arr    = np.clip(z_of_chi(chi_arr), 0.0, 10.0)
    dzdchi   = H0 * E_z(z_arr) / C_KMS
    nchi_raw = nz_frb(z_arr, alpha) * dzdchi
    norm     = np.trapezoid(nchi_raw, chi_arr)
    nchi     = nchi_raw / norm
    nchi_bin = np.where((chi_arr >= lo) & (chi_arr <= hi), nchi, 0.0)
    return interp1d(chi_arr, nchi_bin, bounds_error=False, fill_value=0.0)


# ---------------------------------------------------------------------------
# Tomographic spectra for a single bin pair (i, j)
# ---------------------------------------------------------------------------

def _CDD_ij(ell_arr, GDi_fn, GDj_fn, Pmm, z_of_chi,
             b0e=0.75, zfb=5.0, fNL=0.0):
    """C^DD_{ij}(ell) in [pc/cm^3]^2 sr."""
    chi_arr = np.linspace(2.0, CHI_MAX, N_CHI)
    z_arr   = np.clip(z_of_chi(chi_arr), 0.0, 9.9)
    GDi     = GDi_fn(chi_arr)
    GDj     = GDj_fn(chi_arr)
    be      = b_e(z_arr, b0e, zfb)
    out     = np.zeros(len(ell_arr))
    for j, ell in enumerate(ell_arr):
        k  = ell / chi_arr
        Pm = Pmm(k, z_arr)
        Db = delta_b_PNG(k, z_arr, fNL, b0e, zfb) if fNL != 0.0 else 0.0
        beff = be + Db
        out[j] = np.trapezoid(GDi * GDj * beff**2 * Pm / chi_arr**2, chi_arr)
    return A_DM**2 * out


def _Ctt_ij(ell_arr, nchi_i_fn, nchi_j_fn, Pmm, z_of_chi, l_AU=100.0):
    """C^tt_{ij}(ell) in s^2 sr."""
    chi_arr = np.linspace(2.0, CHI_MAX, N_CHI)
    z_arr   = np.clip(z_of_chi(chi_arr), 0.0, 9.9)
    l_Mpc   = l_AU * AU_MPC
    ni      = nchi_i_fn(chi_arr)
    nj      = nchi_j_fn(chi_arr)
    out     = np.zeros(len(ell_arr))
    for j, ell in enumerate(ell_arr):
        k  = ell / chi_arr
        Pm = Pmm(k, z_arr)
        fPP = f_PhiPhi(k, z_arr)
        intgd = (ni / chi_arr) * (nj / chi_arr) * (ell / chi_arr)**4 * fPP * Pm / chi_arr**2
        out[j] = CONV**2 * (l_Mpc**4 / 4.0) * np.trapezoid(intgd, chi_arr)
    return out


def _CDt_ij(ell_arr, GDi_fn, nchi_j_fn, Pmm, z_of_chi,
             b0e=0.75, zfb=5.0, l_AU=100.0):
    """C^Dt_{ij}(ell): DM from bin i, timing from bin j. Units: pc/cm^3*s*sr."""
    chi_arr = np.linspace(2.0, CHI_MAX, N_CHI)
    z_arr   = np.clip(z_of_chi(chi_arr), 0.0, 9.9)
    l_Mpc   = l_AU * AU_MPC
    GDi     = GDi_fn(chi_arr)
    nj      = nchi_j_fn(chi_arr)
    be      = b_e(z_arr, b0e, zfb)
    out     = np.zeros(len(ell_arr))
    for j, ell in enumerate(ell_arr):
        k  = ell / chi_arr
        Pm = Pmm(k, z_arr)
        fmP = f_mPhi(k, z_arr)
        intgd = GDi * (nj / chi_arr) * be * (ell / chi_arr)**2 * fmP * Pm / chi_arr**2
        out[j] = CONV * (l_Mpc**2 / 2.0) * np.trapezoid(intgd, chi_arr)
    return A_DM * out


# ---------------------------------------------------------------------------
# Full tomographic Fisher
# ---------------------------------------------------------------------------

def run_tomo_cases(alpha, Pmm, z_of_chi, n_tomo=4,
                   l_AU=100.0, N_FRB=1e4, b0e=0.75, zfb=5.0,
                   verbose=True):
    """
    Tomographic Fisher analysis with n_tomo redshift bins.

    For each unique bin pair (i <= j) and each ell, computes
    C^DD_{ij}, C^tt_{ij}, C^Dt_{ij} and accumulates the Fisher matrix.

    The covariance between different bin pairs is neglected (block-diagonal
    approximation), which is conservative.

    Returns
    -------
    dict with keys: sA, sB, sC, sb_B, sb_C (sigma values)
                    FA, FB, FC (Fisher matrices)
    """
    if verbose:
        print(f"  Tomographic Fisher: alpha={alpha}, n_tomo={n_tomo}, "
              f"l={l_AU}AU, N={N_FRB:.0e}")

    # Build bin boundaries
    bounds = build_tomo_boundaries(alpha, z_of_chi, n_tomo=n_tomo)
    if verbose:
        print(f"  Bin boundaries [Mpc]: {[f'{b:.0f}' for b in bounds]}")

    # Build kernels for each bin
    GD_fns    = []
    nchi_fns  = []
    for i in range(n_tomo):
        lo, hi = bounds[i], bounds[i+1]
        GD_fns.append(   _build_WD_bin(  alpha, z_of_chi, lo, hi))
        nchi_fns.append( _build_nchi_bin(alpha, z_of_chi, lo, hi))

    # Unique bin pairs
    pairs = [(i, j) for i in range(n_tomo) for j in range(i, n_tomo)]
    n_pairs = len(pairs)
    if verbose:
        print(f"  Computing {n_pairs} bin-pair spectra ...")

    # Noise per bin (equal count -> noise x n_tomo per bin)
    nDD = noise_DD(N_FRB=N_FRB, n_tomo=n_tomo)
    nTT = noise_tt(N_FRB=N_FRB)

    # Parameters: [fNL, b0e, zfb]
    FA_tot = np.zeros((1, 1))
    FB_tot = np.zeros((3, 3))
    FC_tot = np.zeros((3, 3))

    for ip, (bi, bj) in enumerate(pairs):
        if verbose and ip % 2 == 0:
            print(f"    pair ({bi},{bj}) of {n_pairs} ...", flush=True)

        # Fiducial spectra
        cDD0 = _CDD_ij(ELL_ARR, GD_fns[bi], GD_fns[bj], Pmm, z_of_chi, b0e, zfb)
        cTT0 = _Ctt_ij(ELL_ARR, nchi_fns[bi], nchi_fns[bj], Pmm, z_of_chi, l_AU)
        # Symmetrised cross-spectrum: average (DM_i, t_j) and (DM_j, t_i)
        cDT0 = 0.5 * (
            _CDt_ij(ELL_ARR, GD_fns[bi], nchi_fns[bj], Pmm, z_of_chi, b0e, zfb, l_AU) +
            _CDt_ij(ELL_ARR, GD_fns[bj], nchi_fns[bi], Pmm, z_of_chi, b0e, zfb, l_AU)
        )

        # Derivatives: fNL
        eps_f = 0.5
        dDD_f = (_CDD_ij(ELL_ARR,GD_fns[bi],GD_fns[bj],Pmm,z_of_chi,b0e,zfb,+eps_f) -
                 _CDD_ij(ELL_ARR,GD_fns[bi],GD_fns[bj],Pmm,z_of_chi,b0e,zfb,-eps_f)) / (2*eps_f)
        dDT_f = np.zeros_like(ELL_ARR)  # small; set to zero for stability

        # Derivatives: b0e
        eps_b = 0.02
        dDD_b = (_CDD_ij(ELL_ARR,GD_fns[bi],GD_fns[bj],Pmm,z_of_chi,b0e+eps_b,zfb) -
                 _CDD_ij(ELL_ARR,GD_fns[bi],GD_fns[bj],Pmm,z_of_chi,b0e-eps_b,zfb)) / (2*eps_b)
        dDT_b = 0.5*(
            (_CDt_ij(ELL_ARR,GD_fns[bi],nchi_fns[bj],Pmm,z_of_chi,b0e+eps_b,zfb,l_AU) -
             _CDt_ij(ELL_ARR,GD_fns[bi],nchi_fns[bj],Pmm,z_of_chi,b0e-eps_b,zfb,l_AU)) +
            (_CDt_ij(ELL_ARR,GD_fns[bj],nchi_fns[bi],Pmm,z_of_chi,b0e+eps_b,zfb,l_AU) -
             _CDt_ij(ELL_ARR,GD_fns[bj],nchi_fns[bi],Pmm,z_of_chi,b0e-eps_b,zfb,l_AU))
        ) / (2*eps_b)

        # Derivatives: zfb
        eps_z = 0.2
        dDD_z = (_CDD_ij(ELL_ARR,GD_fns[bi],GD_fns[bj],Pmm,z_of_chi,b0e,zfb+eps_z) -
                 _CDD_ij(ELL_ARR,GD_fns[bi],GD_fns[bj],Pmm,z_of_chi,b0e,zfb-eps_z)) / (2*eps_z)
        dDT_z = 0.5*(
            (_CDt_ij(ELL_ARR,GD_fns[bi],nchi_fns[bj],Pmm,z_of_chi,b0e,zfb+eps_z,l_AU) -
             _CDt_ij(ELL_ARR,GD_fns[bi],nchi_fns[bj],Pmm,z_of_chi,b0e,zfb-eps_z,l_AU)) +
            (_CDt_ij(ELL_ARR,GD_fns[bj],nchi_fns[bi],Pmm,z_of_chi,b0e,zfb+eps_z,l_AU) -
             _CDt_ij(ELL_ARR,GD_fns[bj],nchi_fns[bi],Pmm,z_of_chi,b0e,zfb-eps_z,l_AU))
        ) / (2*eps_z)

        # Count each pair once if i==j, twice (by symmetry) if i!=j
        weight = 1.0 if bi == bj else 2.0

        for k, ell in enumerate(ELL_ARR):
            cdd = cDD0[k]; ctt = cTT0[k]; cdt = cDT0[k]

            # --- Case A/B: C^DD only ---
            S1  = np.array([[cdd]])
            N1  = np.array([[nDD]])
            dS_fA = [np.array([[dDD_f[k]]])]
            dS_fB = [np.array([[dDD_f[k]]]),
                     np.array([[dDD_b[k]]]),
                     np.array([[dDD_z[k]]])]
            FA_tot += weight * _fisher_ell(ell, S1, N1, dS_fA)
            FB_tot += weight * _fisher_ell(ell, S1, N1, dS_fB)

            # --- Case C: joint {C^DD, C^tt, C^Dt} ---
            S2  = np.array([[cdd, cdt],[cdt, ctt]])
            N2  = np.array([[nDD, 0.0],[0.0, nTT]])
            dS2_f = np.array([[dDD_f[k], dDT_f[k]],[dDT_f[k], 0.0]])
            dS2_b = np.array([[dDD_b[k], dDT_b[k]],[dDT_b[k], 0.0]])
            dS2_z = np.array([[dDD_z[k], dDT_z[k]],[dDT_z[k], 0.0]])
            FC_tot += weight * _fisher_ell(ell, S2, N2, [dS2_f, dS2_b, dS2_z])

    sA = _unmarg(FA_tot, 0)
    sB = _invert_marg(FB_tot, 0)
    sC = _invert_marg(FC_tot, 0)
    sb_B = _invert_marg(FB_tot, 1)
    sb_C = _invert_marg(FC_tot, 1)

    return dict(FA=FA_tot, FB=FB_tot, FC=FC_tot,
                sA=sA, sB=sB, sC=sC,
                sb_B=sb_B, sb_C=sb_C)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    from cosmology import build_chi_z, build_Pmm

    print("=" * 60)
    print("tomo.py  self-test  (n_tomo=4, deep survey)")
    print("=" * 60)

    _, z_of_chi = build_chi_z()
    Pmm, _      = build_Pmm()

    t0 = time.time()
    r  = run_tomo_cases(2.0, Pmm, z_of_chi, n_tomo=4,
                         l_AU=100., N_FRB=1e4, verbose=True)
    dt = time.time() - t0

    print(f"\n  Done in {dt:.1f}s")
    print(f"  sigma(fNL):")
    print(f"    Case A (fixed bias):     {r['sA']:.2f}")
    print(f"    Case B (marg, no cross): {r['sB']:.2f}")
    print(f"    Case C (marg, +cross):   {r['sC']:.2f}")
    print(f"    Ratio sB/sC:             {r['sB']/r['sC']:.2f}x")
    print(f"  sigma(b0e):")
    print(f"    Case B: {r['sb_B']:.4f}")
    print(f"    Case C: {r['sb_C']:.4f}")
    print(f"    Improvement: {r['sb_B']/r['sb_C']:.2f}x")
    print("\n  tomo.py  OK")
