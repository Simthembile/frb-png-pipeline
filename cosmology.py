"""
cosmology.py
============
Core cosmological quantities for the FRB DM x Timing cross-spectrum paper.

Planck 2018 ΛCDM cosmology. All distances in Mpc, times in seconds,
velocities in km/s. Gravitational potential Phi is dimensionless (Phi/c^2).

Exports
-------
H0, h, OM_M, OM_B, C_KMS, AU_MPC          -- constants
E_z(z)                                      -- dimensionless Hubble
build_chi_z()                               -- chi(z) and z(chi) interpolators
build_Pmm()                                 -- CAMB P_mm(k,z) interpolator
f_PhiPhi(k,z), f_mPhi(k,z)                 -- Poisson conversion factors
b_e(z,b0e,zfb), delta_b_PNG(k,z,fNL,...)  -- electron bias and PNG correction
fIGM(z), nz_frb(z,alpha)                   -- astrophysical functions
build_nchi(alpha,z_of_chi)                  -- FRB source distribution n(chi)
build_WD(alpha,z_of_chi)                    -- DM weight kernel W_D(chi)
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline, interp1d
import camb

# ---------------------------------------------------------------------------
# Planck 2018 fiducial cosmology
# ---------------------------------------------------------------------------
H0     = 67.4          # km/s/Mpc
h      = H0 / 100.0
ombh2  = 0.0224
omch2  = 0.120
ns     = 0.965
As     = 2.101e-9
tau    = 0.054
OM_M   = (ombh2 + omch2) / h**2
OM_B   = ombh2 / h**2
C_KMS  = 2.998e5        # km/s  (speed of light)
AU_MPC = 1.0 / 206264.806  # Mpc per AU

# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

def E_z(z):
    """Dimensionless Hubble rate H(z)/H0 in flat ΛCDM."""
    return np.sqrt(OM_M * (1.0 + z)**3 + (1.0 - OM_M))


def build_chi_z(z_max=6.0, nz=600):
    """
    Returns (chi_of_z, z_of_chi) cubic-spline interpolators.
    chi in Mpc, z dimensionless.
    """
    z_arr   = np.linspace(0.0, z_max, nz)
    chi_arr = np.zeros(nz)
    for i in range(1, nz):
        chi_arr[i], _ = quad(
            lambda zp: C_KMS / (H0 * E_z(zp)), 0.0, z_arr[i]
        )
    chi_of_z = interp1d(z_arr,   chi_arr, kind='cubic', fill_value='extrapolate')
    z_of_chi = interp1d(chi_arr, z_arr,   kind='cubic', fill_value='extrapolate')
    return chi_of_z, z_of_chi

# ---------------------------------------------------------------------------
# Matter power spectrum via CAMB
# ---------------------------------------------------------------------------

def build_Pmm(kmin=1e-4, kmax=200.0, nk=400, nz=52):
    """
    Build a fast P_mm(k, z) interpolator using CAMB linear power spectrum.

    Parameters
    ----------
    kmin, kmax : float   k range in h/Mpc (CAMB native); converted to Mpc^-1
    nk, nz     : int     grid resolution

    Returns
    -------
    Pmm   : callable  Pmm(k, z) with k in Mpc^-1, returns P in Mpc^3
    k_mpc : ndarray   k grid in Mpc^-1
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(
        redshifts=np.linspace(0.0, 5.0, nz)[::-1],
        kmax=kmax, nonlinear=False
    )
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)

    kh_arr, z_out, pk_arr = results.get_matter_power_spectrum(
        minkh=kmin, maxkh=kmax, npoints=nk
    )
    # Convert h/Mpc -> Mpc^-1, h^-3 Mpc^3 -> Mpc^3
    k_mpc  = kh_arr * h
    pk_mpc = pk_arr / h**3          # shape (nz, nk), z ascending

    # Bicubic log-log spline over (log k, z)
    log_k  = np.log(k_mpc)
    log_pk = np.log(pk_mpc.T + 1e-200)  # shape (nk, nz)
    spl    = RectBivariateSpline(log_k, z_out, log_pk, kx=3, ky=3)
    lk_lo, lk_hi = log_k[0], log_k[-1]
    z_lo,  z_hi  = z_out[0], z_out[-1]

    def Pmm(k, z):
        k  = np.atleast_1d(np.asarray(k, float))
        z  = np.atleast_1d(np.asarray(z, float))
        lk = np.log(np.clip(k, np.exp(lk_lo) * 1.001, np.exp(lk_hi) * 0.999))
        zc = np.clip(z, z_lo, z_hi)
        return np.exp(spl(lk, zc, grid=False))

    return Pmm, k_mpc

# ---------------------------------------------------------------------------
# Poisson equation conversion factors
# ---------------------------------------------------------------------------

def f_PhiPhi(k, z):
    """
    Factor such that P_PhiPhi(k,z) = f_PhiPhi(k,z) * P_mm(k,z).

    From Poisson: k^2 Phi = -(3/2) Omega_m (H0/c)^2 a^{-1} delta_m
    => P_PhiPhi = [(3/2) Omega_m (H0/c)^2]^2 / (k^4 a^2)  * P_mm
    Units: [(km/s/Mpc)^2 / (km/s)^2]^2 / Mpc^{-4} = Mpc^4 (dimensionless Phi)
    """
    a = 1.0 / (1.0 + z)
    return (1.5 * OM_M * H0**2 / C_KMS**2)**2 / (k**4 * a**2)


def f_mPhi(k, z):
    """
    Factor such that P_{m,Phi}(k,z) = f_mPhi(k,z) * P_mm(k,z).

    P_{m,Phi} = -(3/2) Omega_m (H0/c)^2 / (k^2 a)  * P_mm   [negative]
    Units: Mpc^2
    """
    a = 1.0 / (1.0 + z)
    return -(1.5 * OM_M * H0**2 / C_KMS**2) / (k**2 * a)

# ---------------------------------------------------------------------------
# Electron bias model  (Reischke et al. 2020, Sec. II B)
# ---------------------------------------------------------------------------

def b_e(z, b0e=0.75, zfb=5.0):
    """
    Electron bias with linear redshift evolution.
    b_e(z) = b0e + (1-b0e)*(z/zfb)  for z < zfb
    b_e(z) = 1                        for z >= zfb
    """
    z = np.atleast_1d(np.asarray(z, float))
    return np.where(z >= zfb,
                    1.0,
                    b0e + (1.0 - b0e) * (z / zfb))


def delta_b_PNG(k, z, fNL, b0e=0.75, zfb=5.0, delta_c=1.686):
    """
    Scale-dependent PNG bias correction (Slosar et al. 2008; Reischke eq.10):

    Delta_b^NG(k,z) = 3 fNL delta_c (b_e - 1) Omega_m H0^2 / (c^2 a k^2)

    This k^{-2} signature is the primary PNG signal in C^DD(ell).
    """
    a  = 1.0 / (1.0 + z)
    be = b_e(z, b0e, zfb)
    return 3.0 * fNL * delta_c * (be - 1.0) * OM_M * H0**2 / (C_KMS**2 * a * k**2)

# ---------------------------------------------------------------------------
# IGM electron fraction  (Reischke et al. 2020, Sec. II A)
# ---------------------------------------------------------------------------

def fIGM(z):
    """
    Fraction of baryons in the IGM as a function of redshift.
    10% locked in galaxies at z > 1.5, 20% at z < 0.4, linear between.
    """
    z = np.atleast_1d(np.asarray(z, float))
    return np.where(z > 1.5, 0.90,
           np.where(z < 0.4, 0.80,
                    0.90 - (0.10 / 1.1) * (1.5 - z)))

# ---------------------------------------------------------------------------
# FRB source distribution
# ---------------------------------------------------------------------------

def nz_frb(z, alpha):
    """Unnormalised FRB source distribution: n(z) ∝ z^2 exp(-alpha*z)."""
    return z**2 * np.exp(-alpha * z)


def build_nchi(alpha, z_of_chi, chi_max=6500.0, n=500):
    """
    Returns (nchi_fn, chi_arr) where nchi_fn(chi) is the normalised
    comoving FRB source density [Mpc^-1] with ∫ nchi dchi = 1.
    """
    chi_arr  = np.linspace(1.0, chi_max, n)
    z_arr    = np.clip(z_of_chi(chi_arr), 0.0, 10.0)
    dzdchi   = H0 * E_z(z_arr) / C_KMS       # Mpc^-1
    nchi_raw = nz_frb(z_arr, alpha) * dzdchi
    norm     = np.trapezoid(nchi_raw, chi_arr)
    nchi_fn  = interp1d(chi_arr, nchi_raw / norm,
                         bounds_error=False, fill_value=0.0)
    return nchi_fn, chi_arr

# ---------------------------------------------------------------------------
# DM weight kernel  W_D(chi)   (Reischke et al. 2020, eqs 15-16)
# ---------------------------------------------------------------------------
# We set the overall amplitude A = 1 (dimensionless).  The physical DM
# amplitude A ~ 1000 pc/cm^3 is applied in spectra.py where necessary.
# The kernel encodes the line-of-sight geometry: sources behind chi
# contribute to the DM of a source at chi.

def build_WD(alpha, z_of_chi, chi_max=6500.0, n=500):
    """
    Build the DM weight kernel W_D(chi) [Mpc^-1, A=1 units].

    W_D(chi) = W(chi) * integral_{chi}^{chi_H} n(chi') dchi'

    W(chi) = fIGM(z) / [(1+z) E(z)] * |dz/dchi|

    Parameters
    ----------
    alpha     : float  FRB distribution parameter
    z_of_chi  : callable  comoving distance to redshift mapping
    chi_max   : float  maximum comoving distance [Mpc]
    n         : int    grid resolution

    Returns
    -------
    WD_fn   : interp1d interpolator of W_D(chi)
    chi_arr : ndarray  chi grid used
    z_arr   : ndarray  z values at chi_arr
    """
    chi_arr  = np.linspace(1.0, chi_max, n)
    z_arr    = np.clip(z_of_chi(chi_arr), 0.0, 10.0)
    Ez       = E_z(z_arr)
    F        = fIGM(z_arr)
    dzdchi   = H0 * Ez / C_KMS              # Mpc^-1
    W        = F / ((1.0 + z_arr) * Ez) * dzdchi   # Mpc^-1

    nchi_fn, _ = build_nchi(alpha, z_of_chi, chi_max, n)
    nchi_arr   = nchi_fn(chi_arr)

    # Tail integral: T(chi) = integral_{chi}^{chi_max} n(chi') dchi'
    tail = np.array([
        np.trapezoid(nchi_arr[i:], chi_arr[i:]) for i in range(n)
    ])

    WD_arr = W * tail   # Mpc^-1 (dimensionless in A=1 units)
    WD_fn  = interp1d(chi_arr, WD_arr, bounds_error=False, fill_value=0.0)
    return WD_fn, chi_arr, z_arr


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    print("=" * 55)
    print("cosmology.py  self-test")
    print("=" * 55)

    chi_of_z, z_of_chi = build_chi_z()
    print(f"  chi(z=0.5) = {chi_of_z(0.5):.1f} Mpc")
    print(f"  chi(z=1.0) = {chi_of_z(1.0):.1f} Mpc")
    print(f"  chi(z=3.0) = {chi_of_z(3.0):.1f} Mpc")
    print(f"  E(z=1)     = {E_z(1.0):.5f}")
    print(f"  b_e(z=1, b0e=0.75, zfb=5) = {float(b_e(1.0, 0.75, 5.0)):.5f}")
    print(f"  fIGM(z=1)  = {float(fIGM(1.0)):.3f}")

    print("\n  Building P_mm (CAMB, ~20s) ...")
    t0 = time.time()
    Pmm, k_mpc = build_Pmm()
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  P_mm(k=0.1 Mpc^-1, z=0) = {float(Pmm(0.1, 0.0)):.3e} Mpc^3")
    print(f"  P_mm(k=0.1 Mpc^-1, z=1) = {float(Pmm(0.1, 1.0)):.3e} Mpc^3")
    print(f"  f_PhiPhi(0.01, 0) = {float(f_PhiPhi(0.01, 0.0)):.3e}")
    print(f"  f_mPhi(0.01, 0)   = {float(f_mPhi(0.01, 0.0)):.3e}")

    WD_fn, _, _ = build_WD(2.0, z_of_chi)
    print(f"  W_D(chi=2000 Mpc, deep) = {float(WD_fn(2000.0)):.3e}")
    print("\n  cosmology.py  OK")
