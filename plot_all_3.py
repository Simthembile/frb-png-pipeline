"""
plot_all.py
===========
Generate all five publication figures from results.pkl.

Requires:  run_all.py to have been executed first.

Figures produced
----------------
fig1_spectra.pdf/png
    C^DD, C^tt, C^Dt vs ell for shallow and deep surveys.
    Right axis: correlation coefficient |rho|.

fig2_sigma_be.pdf/png
    sigma(b0e) vs N_FRB for both surveys and two baselines.
    Shows self-calibration from the cross-spectrum.

fig3_sigma_fNL_bar.pdf/png
    Bar chart comparing sigma(fNL) across the four cases A/B/C/D
    at N_FRB=1e4 and l=100/500 AU.

fig4_sigma_fNL_vs_N.pdf/png
    sigma(fNL) vs N_FRB for cases A, B, C.

fig5_lmin_vs_N.pdf/png
    Minimum baseline l_min required for sigma(b0e) < 0.05 vs N_FRB.
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open('/home/sim/Documents/frb_pipeline/results.pkl', 'rb') as f:
    res = pickle.load(f)

# Build spectra for Figure 1 (need to recompute at fine ell grid)
from cosmology import build_chi_z, build_Pmm
from spectra   import C_DD, C_tt, C_Dt, noise_DD, noise_tt
from fisher    import ELL_ARR

_, z_of_chi = build_chi_z()
Pmm, _      = build_Pmm()

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 150,
    'lines.linewidth': 1.9,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
})

BLUE   = '#1a5fa8'
RED    = '#c0392b'
ORANGE = '#e67e22'
GREEN  = '#27ae60'
PURPLE = '#7d3c98'
GRAY   = '#555555'

OUT = '/home/sim/Documents/frb_pipeline/'

def cl_factor(ell):
    return ell * (ell + 1) / (2.0 * np.pi)

def save(fig, name):
    fig.savefig(OUT + name + '.pdf', bbox_inches='tight')
    fig.savefig(OUT + name + '.png', bbox_inches='tight', dpi=150)
    print(f"  Saved {name}.pdf/.png")
    plt.close(fig)

# ===========================================================================
# Figure 1: Angular power spectra
# ===========================================================================
print("Generating Figure 1 ...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=False)
NDD = noise_DD(N_FRB=1e4)
Ntt = noise_tt(N_FRB=1e4)

for ax, (alpha, label) in zip(axes, [
        (3.5, r'Shallow survey  ($\alpha=3.5$)'),
        (2.0, r'Deep survey  ($\alpha=2.0$)')]):

    kw  = dict(alpha=alpha, Pmm=Pmm, z_of_chi=z_of_chi)
    cDD = C_DD(ELL_ARR, **kw)
    cTT = C_tt(ELL_ARR, **kw, l_AU=100.)
    cDT = C_Dt(ELL_ARR, **kw, l_AU=100.)
    fac = cl_factor(ELL_ARR)

    # Normalise for display — each spectrum scaled by its own peak
    norm_DD = np.max(fac * cDD)
    norm_tt = np.max(fac * cTT)
    cDT_abs  = np.where(np.abs(cDT) > 0, np.abs(cDT), np.nan)
    norm_DT  = np.nanmax(fac * cDT_abs)

    ax.loglog(ELL_ARR, fac * cDD / norm_DD,
              color=BLUE,   lw=2.0, ls='-',
              label=r'$C^{\rm DD}(\ell)$  [normalised]')
    ax.loglog(ELL_ARR, fac * cDT_abs / norm_DT * 0.8,
              color=RED,    lw=2.0, ls='--',
              label=r'$|C^{{\rm D}\Delta t}(\ell)|$  [rescaled]')
    # C^tt rescaled to 80% of C^DD peak for visual comparison
    ax.loglog(ELL_ARR, fac * cTT / norm_tt * 0.6,
              color=ORANGE, lw=2.0, ls=':',
              label=r'$C^{\Delta t\Delta t}(\ell)$  [rescaled]')

    # Noise levels (scaled same way as their spectra)
    ax.axhline(NDD / norm_DD,           color=BLUE,   ls=(0,(4,2)), lw=1.0, alpha=0.55)
    ax.axhline(Ntt / norm_tt * 0.6,     color=ORANGE, ls=(0,(4,2)), lw=1.0, alpha=0.55)

    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$\ell(\ell{+}1)C(\ell)/2\pi$  [normalised]')
    ax.set_xlim(2, 100)
    ax.set_ylim(5e-4, 3.0)
    ax.set_title(label)
    # Correlation coefficient on twin axis
    rho = cDT / np.sqrt(np.abs(cDD * cTT))
    ax2 = ax.twinx()
    rho_line, = ax2.semilogx(ELL_ARR, np.abs(rho), color=PURPLE, lw=1.4, ls='-.',
                              alpha=0.85, label=r'$|\rho(\ell)|$')
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel(
        r'$|\rho(\ell)| = |C^{{\rm D}\Delta t}|/\sqrt{C^{\rm DD}\,C^{\Delta t\Delta t}}$',
        color=PURPLE, fontsize=9)
    ax2.tick_params(axis='y', colors=PURPLE, labelsize=9)
    ax2.yaxis.label.set_color(PURPLE)

    # Merge handles from both axes into one legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(rho_line)
    labels.append(r'$|\rho(\ell)|$')
    ax.legend(handles, labels, loc='lower left', framealpha=0.9, fontsize=9)

    ax.text(0.97, 0.97, r'$l=100\,\mathrm{AU},\ N_{\rm FRB}=10^4$',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(fc='white', ec='none', alpha=0.85))

fig.tight_layout()
save(fig, 'fig1_spectra')

# ===========================================================================
# Figure 2: sigma(b0e) vs N_FRB
# ===========================================================================
print("Generating Figure 2 ...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
TARGET = 0.05

for ax, (survey, label) in zip(axes, [
        ('shallow', r'Shallow survey  ($\alpha=3.5$)'),
        ('deep',    r'Deep survey  ($\alpha=2.0$)')]):

    for l_AU, col, ls, lbl in [
            (100., BLUE,   '-',  '$l=100$ AU'),
            (500., RED,    '--', '$l=500$ AU')]:
        key = f"sweep_N_{survey}_l{int(l_AU)}"
        N       = res[key]['N_arr']
        sb_B    = res[key]['sb_B']
        sb_C    = res[key]['sb_C']
        # No-cross result (nearly flat – limited by prior, not shot noise)
        ax.loglog(N, sb_B, color=col, ls=':', lw=1.4, alpha=0.5)
        # With cross-spectrum
        ax.loglog(N, sb_C, color=col, ls=ls, lw=2.0, label=f'{lbl} (with $C^{{\\rm D\\Delta t}}$)')

    ax.axhline(TARGET, color=GRAY,   ls='--', lw=1.3, label=r'$\sigma(b_e^0)=0.05$ target')
    ax.axhline(res[f'{survey}_l100']['sb_no_cross'], color='gray', ls=':', lw=1.0)
    ax.text(1.1e3, res[f'{survey}_l100']['sb_no_cross'] * 1.12,
            r'No cross ($C^{{\rm D}\Delta t}=0$)', color='gray', fontsize=8.5)

    ax.set_xlabel(r'$N_{\rm FRB}$')
    if ax is axes[0]:
        ax.set_ylabel(r'$\sigma(b_e^0)$')
    ax.set_xlim(1e3, 1e5)
    ax.set_ylim(1e-2, 1.2)
    ax.set_title(label)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

fig.suptitle(
    r'Self-calibration of electron bias $b_e^0$ via the $C^{{\rm D}\Delta t}(\ell)$ cross-spectrum',
    fontsize=12)
fig.tight_layout()
save(fig, 'fig2_sigma_be')

# ===========================================================================
# Figure 3: Bar chart sigma(fNL) for four cases
# ===========================================================================
print("Generating Figure 3 ...")

case_labels = ['A\n(fixed bias)', 'B\n(marg, no cross)', 'C\n(marg + cross)', 'D\n(timing only)']
case_keys   = ['sA', 'sB', 'sC', 'sD']
bar_colors  = [BLUE, ORANGE, GREEN, PURPLE]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

for ax, (survey, label) in zip(axes, [
        ('shallow', r'Shallow survey  ($\alpha=3.5$)'),
        ('deep',    r'Deep survey  ($\alpha=2.0$)')]):

    v100 = [res[f'{survey}_l100'][k] for k in case_keys]
    v500 = [res[f'{survey}_l500'][k] for k in case_keys]
    # Cap inf at a display value
    cap  = 2e4
    v100 = [min(v, cap) for v in v100]
    v500 = [min(v, cap) for v in v500]

    x = np.arange(4)
    w = 0.36
    ax.bar(x - w/2, v100, w, color=bar_colors, alpha=0.85,
           edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, v500, w, color=bar_colors, alpha=0.40,
           edgecolor=bar_colors, linewidth=1.0, hatch='//')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=9.5)
    ax.set_ylabel(r'$\sigma(f_{\rm NL})$')
    ax.set_ylim(50, cap * 1.5)
    ax.set_title(label)
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs='auto'))

    # Annotate ratio sB -> sC
    rB = v100[1]; rC = v100[2]
    ymax_annot = max(rB, rC) * 1.5
    ax.annotate('', xy=(2-w/2, ymax_annot*0.9), xytext=(1-w/2, ymax_annot*0.9),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=1.6))
    ax.text(1.5-w/2, ymax_annot * 1.05,
            f'×{rB/rC:.1f}',
            ha='center', fontsize=10, color=RED, fontweight='bold')

    # Legend
    leg = [
        Patch(facecolor='gray', alpha=0.85, label='$l=100$ AU  (solid)'),
        Patch(facecolor='gray', alpha=0.40, hatch='//', label='$l=500$ AU  (hatch)'),
    ]
    ax.legend(handles=leg, fontsize=9, loc='upper right')
    ax.text(0.02, 0.97, r'$N_{\rm FRB}=10^4$',
            transform=ax.transAxes, va='top', fontsize=9)

fig.suptitle(
    r'$\sigma(f_{\rm NL})$ for four analysis cases — impact of $C^{{\rm D}\Delta t}$ cross-spectrum',
    fontsize=12)
fig.tight_layout()
save(fig, 'fig3_sigma_fNL_bar')

# ===========================================================================
# Figure 4: sigma(fNL) vs N_FRB, cases A/B/C
# ===========================================================================
print("Generating Figure 4 ...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)

for ax, (survey, label) in zip(axes, [
        ('shallow', r'Shallow survey  ($\alpha=3.5$)'),
        ('deep',    r'Deep survey  ($\alpha=2.0$)')]):

    key = f"sweep_N_{survey}_l100"
    N   = res[key]['N_arr']
    ax.loglog(N, res[key]['sA'], color=BLUE,   lw=2.0, ls='-',
              label=r'Case A: fixed $b_e$')
    ax.loglog(N, res[key]['sB'], color=ORANGE, lw=2.0, ls='--',
              label=r'Case B: marg., no cross')
    ax.loglog(N, res[key]['sC'], color=GREEN,  lw=2.0, ls='-.',
              label=r'Case C: marg. $+$ cross')

    ax.axhline(10., color=GRAY, ls=':', lw=1.0)
    ax.text(1.05e3, 11.5, r'$f_{\rm NL}=10$  (CMB bound)',
            fontsize=8.5, color=GRAY)
    ax.axhline(1., color=GRAY, ls=':', lw=1.0, alpha=0.5)
    ax.text(1.05e3, 1.15, r'$f_{\rm NL}=1$',
            fontsize=8.5, color=GRAY, alpha=0.7)

    ax.set_xlabel(r'$N_{\rm FRB}$')
    if ax is axes[0]:
        ax.set_ylabel(r'$\sigma(f_{\rm NL})$')
    ax.set_xlim(1e3, 1e5)
    ax.set_ylim(0.3, 3e4)
    ax.set_title(label)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(0.02, 0.04, '$l=100$ AU', transform=ax.transAxes, fontsize=9)

fig.suptitle(
    r'$\sigma(f_{\rm NL})$ vs $N_{\rm FRB}$ — cross-spectrum recovers fixed-bias sensitivity',
    fontsize=12)
fig.tight_layout()
save(fig, 'fig4_sigma_fNL_vs_N')

# ===========================================================================
# Figure 5: Minimum baseline l_min vs N_FRB for sigma(b0e) < 0.05
# ===========================================================================
print("Generating Figure 5 ...")

fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))

N_plot  = np.logspace(3, 5, 80)
target  = 0.05

for survey, col, ls in [('shallow', BLUE, '-'), ('deep', RED, '--')]:
    sw_l = res[f'sweep_l_{survey}']
    l_arr    = sw_l['l_arr']
    sb_C_N4  = sw_l['sb_C']       # sigma(b0e) at N=1e4

    l_min_arr = []
    for N in N_plot:
        # sigma(b0e, N) ~ sigma(b0e, N=1e4) * sqrt(1e4/N)  [shot-noise scaling]
        sb_N = sb_C_N4 * np.sqrt(1e4 / N)
        idx  = np.where(sb_N < target)[0]
        l_min_arr.append(float(l_arr[idx[0]]) if len(idx) > 0 else np.nan)

    lbl = ('Shallow' if survey == 'shallow' else 'Deep') + \
          r'  ($\alpha=' + ('3.5' if survey=='shallow' else '2.0') + r'$)'
    ax.loglog(N_plot, l_min_arr, color=col, ls=ls, lw=2.0, label=lbl)

# Observational regime shading
ax.axhspan(0.1,  50.,   alpha=0.10, color=ORANGE, zorder=0)
ax.axhspan(50.,  200.,  alpha=0.10, color=GREEN,  zorder=0)
ax.text(1.2e3, 3.0,   r'Lensed FRBs  ($\sim0.1$–$50$ AU)',
        fontsize=9, color=ORANGE)
ax.text(1.2e3, 70.,   r'Near-term space mission  ($\sim50$–$200$ AU)',
        fontsize=9, color=GREEN)
ax.axhline(100., color=GREEN, ls=':', lw=1.2, alpha=0.7)
ax.text(3.2e4, 112, r'Lu et al. $l=100$ AU', fontsize=8.5, color=GREEN, alpha=0.9)

ax.set_xlabel(r'$N_{\rm FRB}$')
ax.set_ylabel(r'Minimum baseline $l_{\rm min}$  [AU]')
ax.set_xlim(1e3, 1e5)
ax.set_ylim(0.5, 1200)
ax.set_title(r'Baseline required for $\sigma(b_e^0) < 0.05$')
ax.legend(loc='upper right', framealpha=0.9)
ax.xaxis.set_minor_locator(ticker.LogLocator(subs='auto'))
ax.yaxis.set_minor_locator(ticker.LogLocator(subs='auto'))

fig.tight_layout()
save(fig, 'fig5_lmin_vs_N')

print("\nAll figures complete.")
