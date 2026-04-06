"""
fisher.py
=========
Fisher matrix analysis for theta = {fNL, b0e, zfb} using the joint
data vector {C^DD(ell), C^tt(ell), C^Dt(ell)}.

Four analysis cases
-------------------
  A : C^DD only;  fNL alone (b0e, zfb fixed at fiducial)
      -> Reischke et al. 2020 benchmark
  B : C^DD only;  {fNL, b0e, zfb} marginalised
      -> what Reischke et al. would get if marginalising over bias
  C : {C^DD, C^tt, C^Dt};  {fNL, b0e, zfb} marginalised
      -> THE NEW RESULT: cross-spectrum self-calibrates b_e
  D : C^tt only;  fNL alone (timing is bias-free -> no fNL constraint)
      -> reference: timing alone cannot constrain local fNL

The Fisher matrix at each ell uses the full Gaussian covariance of the
2x2 signal matrix S = [[C^DD, C^Dt],[C^Dt, C^tt]] (for Case C), or the
1x1 submatrix for Cases A/B/D.

Sweeps
------
  sweep_NFRB()     : sigma(fNL) and sigma(b0e) vs N_FRB
  sweep_baseline() : sigma(b0e) vs interferometer baseline l_AU
"""

import numpy as np
from spectra import (
    C_DD, C_tt, C_Dt,
    dC_DD_dfNL, dC_DD_db0e, dC_DD_dzfb,
    dC_tt_dfNL, dC_tt_db0e, dC_tt_dzfb,
    dC_Dt_dfNL, dC_Dt_db0e, dC_Dt_dzfb,
    noise_DD, noise_tt, CONV, A_DM,
)

# ell range (Reischke et al. use ell_max=100 since PNG imprint vanishes above)
ELL_ARR = np.arange(2, 101, dtype=float)
F_SKY   = 0.9   # sky fraction (allowing for Galactic plane)

# Parameter ordering: theta = [fNL, b0e, zfb]
PARAMS = ['fNL', 'b0e', 'zfb']

# ---------------------------------------------------------------------------
# Core Fisher utility
# ---------------------------------------------------------------------------

def _fisher_ell(ell, S, N, dS_list):
    """
    Fisher matrix contribution from one multipole ell.

    F_{ab} = (2ell+1)/2 * f_sky * Tr[C^{-1} dS_a C^{-1} dS_b]

    where C = S + N is the total covariance.

    Parameters
    ----------
    ell     : float
    S       : (n,n) signal covariance matrix
    N       : (n,n) noise covariance matrix
    dS_list : list of (n,n) arrays, one per parameter

    Returns
    -------
    F : (npar, npar) Fisher contribution
    """
    C    = S + N
    # Regularise to avoid singular matrices when signal << noise
    reg  = max(np.abs(np.diag(C)).mean(), 1e-300) * 1e-12
    C   += np.eye(len(C)) * reg
    Cinv = np.linalg.inv(C)
    pref = (2.0 * ell + 1.0) / 2.0 * F_SKY
    npar = len(dS_list)
    F    = np.zeros((npar, npar))
    for a in range(npar):
        for b in range(a, npar):
            val    = np.trace(Cinv @ dS_list[a] @ Cinv @ dS_list[b])
            F[a,b] = pref * val
            F[b,a] = F[a,b]
    return F


def _invert_marg(F, param_idx=0):
    """
    Return sigma(param[param_idx]) after marginalising over all others.
    sigma = sqrt( (F^{-1})_{ii} )
    """
    try:
        Finv = np.linalg.inv(F + np.eye(len(F)) * 1e-300)
        v    = Finv[param_idx, param_idx]
        return np.sqrt(abs(v)) if v > 0 else np.inf
    except np.linalg.LinAlgError:
        return np.inf


def _unmarg(F, param_idx=0):
    """sigma(param[param_idx]) without marginalisation."""
    v = F[param_idx, param_idx]
    return 1.0 / np.sqrt(v) if v > 0 else np.inf

# ---------------------------------------------------------------------------
# Spectra cache (avoids recomputing for N_FRB sweeps)
# ---------------------------------------------------------------------------

def build_cache(alpha, Pmm, z_of_chi, b0e=0.75, zfb=5.0,
                l_AU=100.0, N_FRB=1e4, verbose=True):
    """
    Pre-compute all spectra and derivatives on ELL_ARR.
    Noise terms are stored separately so they can be swapped for sweeps.

    Returns
    -------
    cache : dict with keys
        cDD, cTT, cDT          -- spectra on ELL_ARR
        dDD_f, dDD_b, dDD_z   -- dC^DD / d{fNL, b0e, zfb}
        dDT_f, dDT_b, dDT_z   -- dC^Dt / d{fNL, b0e, zfb}
        nDD, nTT               -- noise at fiducial N_FRB
        meta                   -- dict of input parameters
    """
    kw = dict(alpha=alpha, Pmm=Pmm, z_of_chi=z_of_chi, b0e=b0e, zfb=zfb)
    if verbose:
        print(f"    Computing spectra (alpha={alpha}, l={l_AU}AU, "
              f"N={N_FRB:.0e}) ...")

    cDD = C_DD(ELL_ARR, **kw)
    cTT = C_tt(ELL_ARR, alpha=alpha, Pmm=Pmm, z_of_chi=z_of_chi, l_AU=l_AU)
    cDT = C_Dt(ELL_ARR, **kw, l_AU=l_AU)

    if verbose:
        print("    Computing derivatives ...")

    dDD_f = dC_DD_dfNL(ELL_ARR, **kw)
    dDD_b = dC_DD_db0e(ELL_ARR, **kw)
    dDD_z = dC_DD_dzfb(ELL_ARR, **kw)
    dDT_f = dC_Dt_dfNL(ELL_ARR, **kw, l_AU=l_AU)
    dDT_b = dC_Dt_db0e(ELL_ARR, **kw, l_AU=l_AU)
    dDT_z = dC_Dt_dzfb(ELL_ARR, **kw, l_AU=l_AU)

    nDD = noise_DD(N_FRB=N_FRB)
    nTT = noise_tt(N_FRB=N_FRB)

    return dict(
        cDD=cDD, cTT=cTT, cDT=cDT,
        dDD_f=dDD_f, dDD_b=dDD_b, dDD_z=dDD_z,
        dDT_f=dDT_f, dDT_b=dDT_b, dDT_z=dDT_z,
        nDD=nDD, nTT=nTT,
        meta=dict(alpha=alpha, b0e=b0e, zfb=zfb, l_AU=l_AU, N_FRB=N_FRB),
    )


def _update_noise(cache, N_FRB):
    """Return shallow copy of cache with noise updated for new N_FRB."""
    c = dict(cache)
    c['nDD'] = noise_DD(N_FRB=N_FRB)
    c['nTT'] = noise_tt(N_FRB=N_FRB)
    return c

# ---------------------------------------------------------------------------
# Fisher matrices for the four cases
# ---------------------------------------------------------------------------

def fisher_A(cache):
    """Case A: C^DD only, fNL alone (b0e/zfb fixed)."""
    F = 0.0
    nDD = cache['nDD']
    for i, ell in enumerate(ELL_ARR):
        S  = np.array([[cache['cDD'][i]]])
        N  = np.array([[nDD]])
        dS = [np.array([[cache['dDD_f'][i]]])]
        F += _fisher_ell(ell, S, N, dS)[0, 0]
    return np.array([[F]])


def fisher_B(cache):
    """Case B: C^DD only, {fNL, b0e, zfb} marginalised."""
    F   = np.zeros((3, 3))
    nDD = cache['nDD']
    for i, ell in enumerate(ELL_ARR):
        S   = np.array([[cache['cDD'][i]]])
        N   = np.array([[nDD]])
        dS  = [np.array([[cache['dDD_f'][i]]]),
               np.array([[cache['dDD_b'][i]]]),
               np.array([[cache['dDD_z'][i]]])]
        F  += _fisher_ell(ell, S, N, dS)
    return F


def fisher_C(cache):
    """
    Case C: joint {C^DD, C^tt, C^Dt}, {fNL, b0e, zfb} marginalised.
    THE NEW RESULT.

    Uses 2x2 signal covariance  S = [[C^DD, C^Dt],[C^Dt, C^tt]]
    and noise matrix            N = diag(N^DD, N^tt).
    """
    F   = np.zeros((3, 3))
    nDD = cache['nDD']
    nTT = cache['nTT']
    for i, ell in enumerate(ELL_ARR):
        cdd = cache['cDD'][i]
        ctt = cache['cTT'][i]
        cdt = cache['cDT'][i]
        S   = np.array([[cdd, cdt],
                        [cdt, ctt]])
        N   = np.array([[nDD, 0.0],
                        [0.0, nTT]])
        # Derivative matrices d S / d{fNL, b0e, zfb}
        dS_f = np.array([[cache['dDD_f'][i], cache['dDT_f'][i]],
                         [cache['dDT_f'][i], 0.0]])
        dS_b = np.array([[cache['dDD_b'][i], cache['dDT_b'][i]],
                         [cache['dDT_b'][i], 0.0]])
        dS_z = np.array([[cache['dDD_z'][i], cache['dDT_z'][i]],
                         [cache['dDT_z'][i], 0.0]])
        F += _fisher_ell(ell, S, N, [dS_f, dS_b, dS_z])
    return F


def fisher_D(cache):
    """Case D: C^tt only, fNL alone.
    Since C^tt has no fNL dependence (timing bypasses bias), this gives F=0
    and sigma(fNL) = inf.  Included for completeness."""
    return np.array([[0.0]])

# ---------------------------------------------------------------------------
# Convenience sigma functions
# ---------------------------------------------------------------------------

def sigma_fNL(F, marg=True):
    """
    sigma(fNL) from Fisher matrix F.
    marg=True  : marginalise over b0e, zfb  (use Finv[0,0])
    marg=False : treat b0e, zfb as fixed    (use 1/sqrt(F[0,0]))
    """
    if not marg or F.shape == (1, 1):
        return _unmarg(F, 0)
    return _invert_marg(F, 0)


def sigma_b0e(F):
    """sigma(b0e) marginalised over fNL and zfb."""
    return _invert_marg(F, 1)


def sigma_b0e_ratio(cache):
    """
    Standalone sigma(b0e) from the ratio estimator C^Dt / C^tt.

    Uses error propagation on b_hat_e ~ C^Dt / C^tt:
    Var(b_hat_e, ell) = [(C^DD+N^DD)(C^tt+N^tt) + (C^Dt)^2]
                        / [(2ell+1) f_sky (C^tt)^2]

    Total Fisher for b0e = sum_ell 1/Var
    """
    inv_var = 0.0
    nDD, nTT = cache['nDD'], cache['nTT']
    for i, ell in enumerate(ELL_ARR):
        cdd = cache['cDD'][i]
        ctt = cache['cTT'][i]
        cdt = cache['cDT'][i]
        if ctt <= 0:
            continue
        num = (cdd + nDD) * (ctt + nTT) + cdt**2
        den = (2.0 * ell + 1.0) * F_SKY * ctt**2
        if den > 0:
            inv_var += den / num
    return 1.0 / np.sqrt(inv_var) if inv_var > 0 else np.inf

# ---------------------------------------------------------------------------
# Run all four cases at one fiducial point
# ---------------------------------------------------------------------------

def run_cases(alpha, Pmm, z_of_chi, l_AU=100.0, N_FRB=1e4,
              b0e=0.75, zfb=5.0, verbose=True):
    """
    Compute all four cases and return results dict.

    Returns
    -------
    dict with keys:
        cache                  -- the full spectra/derivatives cache
        FA, FB, FC, FD         -- Fisher matrices
        sA, sB, sC, sD         -- sigma(fNL) for each case
        sb_no_cross            -- sigma(b0e) from Case B
        sb_with_cross          -- sigma(b0e) from Case C (Fisher)
        sb_ratio               -- sigma(b0e) from ratio estimator
    """
    cache = build_cache(alpha, Pmm, z_of_chi, b0e, zfb, l_AU, N_FRB, verbose)

    FA = fisher_A(cache)
    FB = fisher_B(cache)
    FC = fisher_C(cache)
    FD = fisher_D(cache)

    sA = sigma_fNL(FA, marg=False)
    sB = sigma_fNL(FB, marg=True)
    sC = sigma_fNL(FC, marg=True)
    sD = sigma_fNL(FD, marg=False)

    sb_B   = sigma_b0e(FB)
    sb_C   = sigma_b0e(FC)
    sb_rat = sigma_b0e_ratio(cache)

    return dict(
        cache=cache,
        FA=FA, FB=FB, FC=FC, FD=FD,
        sA=sA, sB=sB, sC=sC, sD=sD,
        sb_no_cross=sb_B,
        sb_with_cross=sb_C,
        sb_ratio=sb_rat,
    )

# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep_NFRB(alpha, Pmm, z_of_chi, l_AU=100.0, b0e=0.75, zfb=5.0,
               N_arr=None, verbose=True):
    """
    Sweep N_FRB and return sigma(fNL) for A,B,C and sigma(b0e).
    Spectra and derivatives are computed once; only noise changes.

    Returns
    -------
    dict with keys N_arr, sA, sB, sC, sD, sb_B, sb_C, sb_ratio
    """
    if N_arr is None:
        N_arr = np.logspace(3, 5, 20)

    if verbose:
        print(f"  sweep_NFRB: alpha={alpha}, l={l_AU}AU")
    # Build cache at reference N (noise will be overridden)
    cache0 = build_cache(alpha, Pmm, z_of_chi, b0e, zfb,
                          l_AU, N_FRB=1e4, verbose=verbose)

    out = dict(N_arr=N_arr,
               sA=[], sB=[], sC=[], sD=[],
               sb_B=[], sb_C=[], sb_ratio=[])

    for N in N_arr:
        c  = _update_noise(cache0, N)
        FA = fisher_A(c);  FB = fisher_B(c)
        FC = fisher_C(c);  FD = fisher_D(c)
        out['sA'].append(sigma_fNL(FA, False))
        out['sB'].append(sigma_fNL(FB, True))
        out['sC'].append(sigma_fNL(FC, True))
        out['sD'].append(sigma_fNL(FD, False))
        out['sb_B'].append(sigma_b0e(FB))
        out['sb_C'].append(sigma_b0e(FC))
        out['sb_ratio'].append(sigma_b0e_ratio(c))
        if verbose:
            print(f"    N={N:.1e}: sA={out['sA'][-1]:.1f}  "
                  f"sB={out['sB'][-1]:.1f}  sC={out['sC'][-1]:.1f}  "
                  f"sb_C={out['sb_C'][-1]:.4f}")

    for k in ['sA','sB','sC','sD','sb_B','sb_C','sb_ratio']:
        out[k] = np.array(out[k])
    return out


def sweep_baseline(alpha, Pmm, z_of_chi, N_FRB=1e4, b0e=0.75, zfb=5.0,
                   l_arr=None, verbose=True):
    """
    Sweep interferometer baseline l_AU and return sigma(b0e).
    Recomputes spectra at each baseline (C^tt and C^Dt depend on l_AU).

    Returns
    -------
    dict with keys l_arr, sB, sC, sb_B, sb_C, sb_ratio
    """
    if l_arr is None:
        l_arr = np.logspace(0.3, 3.0, 16)   # ~2 to 1000 AU

    if verbose:
        print(f"  sweep_baseline: alpha={alpha}, N={N_FRB:.0e}")

    out = dict(l_arr=l_arr, sB=[], sC=[], sb_B=[], sb_C=[], sb_ratio=[])

    for l_AU in l_arr:
        c  = build_cache(alpha, Pmm, z_of_chi, b0e, zfb,
                          l_AU, N_FRB, verbose=False)
        FB = fisher_B(c);  FC = fisher_C(c)
        out['sB'].append(sigma_fNL(FB, True))
        out['sC'].append(sigma_fNL(FC, True))
        out['sb_B'].append(sigma_b0e(FB))
        out['sb_C'].append(sigma_b0e(FC))
        out['sb_ratio'].append(sigma_b0e_ratio(c))
        if verbose:
            print(f"    l={l_AU:7.1f} AU: sC={out['sC'][-1]:.1f}  "
                  f"sb_C={out['sb_C'][-1]:.4f}  sb_ratio={out['sb_ratio'][-1]:.4f}")

    for k in ['sB','sC','sb_B','sb_C','sb_ratio']:
        out[k] = np.array(out[k])
    return out

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    from cosmology import build_chi_z, build_Pmm

    print("=" * 60)
    print("fisher.py  self-test")
    print("=" * 60)

    _, z_of_chi = build_chi_z()
    Pmm, _      = build_Pmm()

    for alpha, label in [(3.5, 'Shallow (α=3.5)'), (2.0, 'Deep (α=2.0)')]:
        print(f"\n{'─'*50}")
        print(f"  {label}  l=100 AU  N=1e4")
        t0 = time.time()
        r  = run_cases(alpha, Pmm, z_of_chi, l_AU=100., N_FRB=1e4)
        print(f"  Time: {time.time()-t0:.1f}s")
        print(f"  sigma(fNL):")
        print(f"    Case A (fixed bias):        {r['sA']:.2f}")
        print(f"    Case B (marg, no cross):    {r['sB']:.2f}")
        print(f"    Case C (marg, +cross):      {r['sC']:.2f}")
        print(f"    Ratio sB/sC:                {r['sB']/r['sC']:.2f}x")
        print(f"  sigma(b0e):")
        print(f"    Case B (no cross):          {r['sb_no_cross']:.4f}")
        print(f"    Case C (Fisher, +cross):    {r['sb_with_cross']:.4f}")
        print(f"    Ratio estimator:            {r['sb_ratio']:.4f}")
        print(f"    Improvement factor:         {r['sb_no_cross']/r['sb_with_cross']:.2f}x")

    print("\n  fisher.py  OK")
