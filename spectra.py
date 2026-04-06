"""
spectra.py
==========
Angular power spectra C^DD, C^tt, C^Dt in the Limber approximation,
with physically consistent units throughout.

Unit convention
---------------
  C^DD  [pc/cm^3]^2 sr  -- DM auto-spectrum (A_DM factor applied internally)
  C^tt  s^2 sr          -- Timing delay auto-spectrum
  C^Dt  pc/cm^3 * s * sr -- Cross-spectrum (one A_DM factor)
  N^DD  [pc/cm^3]^2 sr  -- DM shot noise  (sigma_host^2 / n_bar)
  N^tt  s^2 sr          -- Timing shot noise (delta_t^2 / n_bar)
  N^Dt  = 0             -- Cross shot noise (independent noise sources)

The DM amplitude factor A_DM ~ 1000 pc/cm^3 (Reischke et al. 2020).
All C^DD / dC^DD quantities are returned pre-multiplied by A_DM^2.
All C^Dt / dC^Dt quantities are returned pre-multiplied by A_DM.
C^tt is unchanged (no DM dependence).

This ensures the Fisher matrix covariance matrix is dimensionally consistent
and that SNR(C^DD) matches the Reischke et al. results.

Limber-approximation note
-------------------------
For C^tt and C^Dt, the Limber approximation is *exact* at the level of
the dominant contribution: Lu et al. (2025) show the timing signal is
dominated by k_perp = ell/chi modes (k_|| ~ 0), which is precisely the
Limber condition. Corrections are O((kD)^{-2}) < 10^{-6}.
"""

import numpy as np
from cosmology import (
    build_WD, build_nchi,
    b_e, delta_b_PNG, f_PhiPhi, f_mPhi,
    AU_MPC, C_KMS, H0
)

# ---------------------------------------------------------------------------
# Physical constants and normalisations
# ---------------------------------------------------------------------------

# DM amplitude (Reischke et al. 2020):  A ~ integral rho_b/m_p c/H0 F(z) ...
# Numerically A ~ 840-1000 pc/cm^3; we use 1000 matching Reischke et al.
A_DM = 1000.0    # pc/cm^3

# Shapiro delay conversion: delta_t [s] = CONV * Phi [dimensionless] * path [Mpc]
# CONV = 2 * (Mpc in km) / c [km/s] = 2 * 3.0857e19 / 2.998e5
MPC_KM = 3.085677581e19    # km/Mpc
CONV   = 2.0 * MPC_KM / C_KMS   # s / (Phi_dimless * Mpc) ~ 2.058e14

# Integration grid
CHI_MAX = 6000.0  # Mpc  (z ~ 4.7)
N_CHI   = 300

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _grid(chi_arr, z_arr, Pmm, ell):
    """Return k = ell/chi [Mpc^-1] and P_mm(k,z) on chi grid."""
    k   = ell / chi_arr
    Pm  = Pmm(k, z_arr)
    return k, Pm

# ---------------------------------------------------------------------------
# C^DD  --  DM auto-spectrum
# ---------------------------------------------------------------------------

def C_DD(ell_arr, alpha, Pmm, z_of_chi,
         b0e=0.75, zfb=5.0, fNL=0.0):
    """
    C^DD(ell) in [pc/cm^3]^2 sr.

    C^DD(ell) = A_DM^2 * integral_0^{chi_H} dchi/chi^2
                         [W_D(chi)]^2 [b_e(z) + Db_PNG(k,z)]^2 P_mm(ell/chi,z)

    Parameters
    ----------
    ell_arr  : 1-D array of multipoles
    alpha    : FRB redshift distribution parameter (3.5=shallow, 2.0=deep)
    Pmm      : callable P_mm(k, z) in Mpc^3
    z_of_chi : callable z(chi)
    b0e      : electron bias today
    zfb      : feedback redshift
    fNL      : local non-Gaussianity parameter

    Returns
    -------
    out : ndarray, shape (len(ell_arr),)  in [pc/cm^3]^2 sr
    """
    chi_arr = np.linspace(2.0, CHI_MAX, N_CHI)
    z_arr   = np.clip(z_of_chi(chi_arr), 0.0, 9.9)
    WD_fn, _, _ = build_WD(alpha, z_of_chi, CHI_MAX, N_CHI)
    WD      = WD_fn(chi_arr)
    be      = b_e(z_arr, b0e, zfb)
    out     = np.zeros(len(ell_arr))
    for j, ell in enumerate(ell_arr):
        k, Pm = _grid(chi_arr, z_arr, Pmm, ell)
        Db    = delta_b_PNG(k, z_arr, fNL, b0e, zfb) if fNL != 0.0 else 0.0
        beff  = be + Db
        out[j] = np.trapezoid(WD**2 * beff**2 * Pm / chi_arr**2, chi_arr)
    return A_DM**2 * out


def dC_DD_dfNL(ell_arr, alpha, Pmm, z_of_chi,
               b0e=0.75, zfb=5.0, eps=0.5):
    """dC^DD/dfNL via central finite difference."""
    cp = C_DD(ell_arr, alpha, Pmm, z_of_chi, b0e, zfb, fNL=+eps)
    cm = C_DD(ell_arr, alpha, Pmm, z_of_chi, b0e, zfb, fNL=-eps)
    return (cp - cm) / (2.0 * eps)


def dC_DD_db0e(ell_arr, alpha, Pmm, z_of_chi,
               b0e=0.75, zfb=5.0, eps=0.02):
    """dC^DD/db0e via central finite difference."""
    cp = C_DD(ell_arr, alpha, Pmm, z_of_chi, b0e + eps, zfb)
    cm = C_DD(ell_arr, alpha, Pmm, z_of_chi, b0e - eps, zfb)
    return (cp - cm) / (2.0 * eps)


def dC_DD_dzfb(ell_arr, alpha, Pmm, z_of_chi,
               b0e=0.75, zfb=5.0, eps=0.2):
    """dC^DD/dzfb via central finite difference."""
    cp = C_DD(ell_arr, alpha, Pmm, z_of_chi, b0e, zfb + eps)
    cm = C_DD(ell_arr, alpha, Pmm, z_of_chi, b0e, zfb - eps)
    return (cp - cm) / (2.0 * eps)

# ---------------------------------------------------------------------------
# C^tt  --  Timing auto-spectrum
# ---------------------------------------------------------------------------

def C_tt(ell_arr, alpha, Pmm, z_of_chi, l_AU=100.0):
    """
    C^tt(ell) in s^2 sr.

    Limber projection of the Lu et al. (2025) Shapiro timing quadrupole
    observable. The timing field Δt^(2)(n̂) is dominated by k_perp = ell/chi
    modes (Limber is exact for this observable):

    C^tt(ell) = CONV^2 * (l_Mpc^4 / 4)
                * integral dchi/chi^2  [n(chi)/chi]^2  (ell/chi)^4
                                       f_PhiPhi(ell/chi,z)  P_mm(ell/chi,z)

    CONV = 2 Mpc / c  converts dimensionless-Phi*Mpc to seconds.
    f_PhiPhi converts P_mm to P_PhiPhi via the Poisson equation.

    Parameters
    ----------
    l_AU : float  interferometer baseline in AU

    Returns
    -------
    out : ndarray  in s^2 sr
    """
    chi_arr  = np.linspace(2.0, CHI_MAX, N_CHI)
    z_arr    = np.clip(z_of_chi(chi_arr), 0.0, 9.9)
    l_Mpc    = l_AU * AU_MPC
    nchi_fn, _ = build_nchi(alpha, z_of_chi, CHI_MAX, N_CHI)
    nchi     = nchi_fn(chi_arr)
    out      = np.zeros(len(ell_arr))
    for j, ell in enumerate(ell_arr):
        k, Pm  = _grid(chi_arr, z_arr, Pmm, ell)
        fPP    = f_PhiPhi(k, z_arr)
        intgd  = (nchi / chi_arr)**2 * (ell / chi_arr)**4 * fPP * Pm / chi_arr**2
        out[j] = CONV**2 * (l_Mpc**4 / 4.0) * np.trapezoid(intgd, chi_arr)
    return out


# C^tt has no bias dependence -- all derivatives are identically zero
def dC_tt_dfNL(ell_arr, *args, **kwargs):
    return np.zeros(len(ell_arr))

def dC_tt_db0e(ell_arr, *args, **kwargs):
    return np.zeros(len(ell_arr))

def dC_tt_dzfb(ell_arr, *args, **kwargs):
    return np.zeros(len(ell_arr))

# ---------------------------------------------------------------------------
# C^Dt  --  DM x Timing cross-spectrum  [THE CENTRAL NEW RESULT]
# ---------------------------------------------------------------------------

def C_Dt(ell_arr, alpha, Pmm, z_of_chi,
         b0e=0.75, zfb=5.0, fNL=0.0, l_AU=100.0):
    """
    C^Dt(ell) in pc/cm^3 * s * sr.

    Derived by cross-correlating the DM field D(n̂) with the timing field
    Δt^(2)(n̂). Both are sourced by the same δ_m along the same sightline.
    Their cross-spectrum is negative (f_mPhi < 0: overdensities sit in
    potential wells) with |ρ| ~ 0.5–0.65.

    C^Dt(ell) = A_DM * CONV * (l_Mpc^2 / 2)
                * integral dchi/chi^2  W_D(chi) * n(chi)/chi * b_e(z)
                                       * (ell/chi)^2 * f_mPhi(ell/chi,z)
                                       * P_mm(ell/chi,z)

    Factors:
      A_DM  : DM amplitude (one power, from DM side)
      CONV  : Shapiro delay conversion (one power, from timing side)
      l_Mpc^2/2 : timing quadrupole baseline factor (one power of l^2)
      (ell/chi)^2 : k_perp^2 from the timing kernel
      f_mPhi : P_{m,Phi}/P_mm from Poisson (negative)

    Returns
    -------
    out : ndarray  in pc/cm^3 * s * sr  (NEGATIVE)
    """
    chi_arr  = np.linspace(2.0, CHI_MAX, N_CHI)
    z_arr    = np.clip(z_of_chi(chi_arr), 0.0, 9.9)
    l_Mpc    = l_AU * AU_MPC
    WD_fn, _, _ = build_WD(alpha, z_of_chi, CHI_MAX, N_CHI)
    WD       = WD_fn(chi_arr)
    be       = b_e(z_arr, b0e, zfb)
    if fNL != 0.0:
        # Use representative k ~ ell_eff / chi_eff for PNG in C^Dt
        # This is a small effect; the main PNG signal is in C^DD
        k_eff  = max(ell_arr.mean() / 2000.0, 1e-4)
        z_eff  = float(np.clip(z_of_chi(2000.0), 0.0, 5.0))
        be     = be + delta_b_PNG(k_eff, z_eff, fNL, b0e, zfb)
    nchi_fn, _ = build_nchi(alpha, z_of_chi, CHI_MAX, N_CHI)
    nchi     = nchi_fn(chi_arr)
    out      = np.zeros(len(ell_arr))
    for j, ell in enumerate(ell_arr):
        k, Pm  = _grid(chi_arr, z_arr, Pmm, ell)
        fmP    = f_mPhi(k, z_arr)
        intgd  = (WD * (nchi / chi_arr) * be *
                  (ell / chi_arr)**2 * fmP * Pm / chi_arr**2)
        out[j] = CONV * (l_Mpc**2 / 2.0) * np.trapezoid(intgd, chi_arr)
    return A_DM * out


def dC_Dt_dfNL(ell_arr, alpha, Pmm, z_of_chi,
               b0e=0.75, zfb=5.0, l_AU=100.0, eps=0.5):
    """dC^Dt/dfNL via central finite difference."""
    cp = C_Dt(ell_arr, alpha, Pmm, z_of_chi, b0e, zfb, +eps, l_AU)
    cm = C_Dt(ell_arr, alpha, Pmm, z_of_chi, b0e, zfb, -eps, l_AU)
    return (cp - cm) / (2.0 * eps)


def dC_Dt_db0e(ell_arr, alpha, Pmm, z_of_chi,
               b0e=0.75, zfb=5.0, l_AU=100.0, eps=0.02):
    """dC^Dt/db0e via central finite difference."""
    cp = C_Dt(ell_arr, alpha, Pmm, z_of_chi, b0e + eps, zfb, 0.0, l_AU)
    cm = C_Dt(ell_arr, alpha, Pmm, z_of_chi, b0e - eps, zfb, 0.0, l_AU)
    return (cp - cm) / (2.0 * eps)


def dC_Dt_dzfb(ell_arr, alpha, Pmm, z_of_chi,
               b0e=0.75, zfb=5.0, l_AU=100.0, eps=0.2):
    """dC^Dt/dzfb via central finite difference."""
    cp = C_Dt(ell_arr, alpha, Pmm, z_of_chi, b0e, zfb + eps, 0.0, l_AU)
    cm = C_Dt(ell_arr, alpha, Pmm, z_of_chi, b0e, zfb - eps, 0.0, l_AU)
    return (cp - cm) / (2.0 * eps)

# ---------------------------------------------------------------------------
# Shot noise terms
# ---------------------------------------------------------------------------

def noise_DD(N_FRB=1e4, sigma_host=50.0, n_tomo=1, f_sky=0.9):
    """
    DM shot noise in [pc/cm^3]^2 sr.
    N^DD = n_tomo * sigma_host^2 / n_bar
    n_bar = N_FRB / (4 pi f_sky)  [sr^-1]
    """
    n_bar = N_FRB / (4.0 * np.pi * f_sky)
    return n_tomo * sigma_host**2 / n_bar


def noise_tt(N_FRB=1e4, delta_t_ns=1.0, f_sky=0.9):
    """
    Timing shot noise in s^2 sr.
    N^tt = delta_t^2 / n_bar
    """
    n_bar = N_FRB / (4.0 * np.pi * f_sky)
    return (delta_t_ns * 1e-9)**2 / n_bar


def noise_Dt():
    """Cross shot noise is zero (DM and timing noise are independent)."""
    return 0.0

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    from cosmology import build_chi_z, build_Pmm

    print("=" * 65)
    print("spectra.py  self-test")
    print("=" * 65)
    _, z_of_chi = build_chi_z()
    Pmm, _      = build_Pmm()

    ell_arr = np.array([2., 5., 10., 20., 50., 100.])
    t0 = time.time()
    cDD = C_DD(ell_arr, 2.0, Pmm, z_of_chi)          # deep survey
    cTT = C_tt(ell_arr, 2.0, Pmm, z_of_chi, 100.)
    cDT = C_Dt(ell_arr, 2.0, Pmm, z_of_chi, 100.)
    dt  = time.time() - t0

    NDD = noise_DD(N_FRB=1e4)
    NTT = noise_tt(N_FRB=1e4)
    rho = cDT / np.sqrt(np.abs(cDD * cTT))

    print(f"  Computed in {dt:.1f}s  (CONV={CONV:.3e} s/Mpc, A_DM={A_DM})")
    print(f"  N^DD = {NDD:.3e} [pc/cm^3]^2 sr")
    print(f"  N^tt = {NTT:.3e} s^2 sr")
    print()
    hdr = f"  {'ell':>4} | {'C^DD':>11} | {'SNR_DD':>6} | {'C^tt':>11} | {'C^Dt':>11} | {'rho':>7}"
    print(hdr)
    print("  " + "-"*(len(hdr)-2))
    for i, ell in enumerate(ell_arr):
        snr = cDD[i] / NDD
        print(f"  {ell:4.0f} | {cDD[i]:11.3e} | {snr:6.2f} | "
              f"{cTT[i]:11.3e} | {cDT[i]:11.3e} | {rho[i]:7.4f}")

    print()
    print("  Derivative check (dC/dfNL at ell=10):")
    dD = dC_DD_dfNL(ell_arr, 2.0, Pmm, z_of_chi)
    dT = dC_Dt_dfNL(ell_arr, 2.0, Pmm, z_of_chi)
    for i, ell in enumerate(ell_arr):
        print(f"    ell={ell:.0f}: dC^DD/dfNL={dD[i]:.3e}, "
              f"dC^Dt/dfNL={dT[i]:.3e}")
    print("\n  spectra.py  OK")
