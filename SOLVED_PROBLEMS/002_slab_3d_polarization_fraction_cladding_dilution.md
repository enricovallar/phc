# Issue 002: 3D Photonic Crystal Slab Mode Parity vs. Volumetric Polarization Fraction Dilution

## Date
2026-09-22

## Status
SOLVED

---

## 1. Problem Statement & Symptoms

In 3D photonic crystal slab simulations (`phc_mpb`), band structure calculations were conducted using two complementary approaches:
1. **Mirror Parity Subspace Solves**: Solving for TE-like (even, $\sigma_z = +1$, using `ms.run_zeven()`) and TM-like (odd, $\sigma_z = -1$, using `ms.run_zodd()`) modes separately.
2. **All-Bands Solve with Continuous Polarization Fractions**: Solving for all bands simultaneously (`ms.run()`) and computing the electric transverse energy fraction:
   $$f_{\mathrm{TE}} = \frac{\int_{\text{cell}} \operatorname{Re}(E_x^* D_x + E_y^* D_y)\, dV}{\int_{\text{cell}} \operatorname{Re}(\mathbf{E}^* \cdot \mathbf{D})\, dV}$$

When plotting all modes on a unified band diagram (`coolwarm_r` colormap where $f_{\mathrm{TE}} = 1.0$ is blue and $f_{\mathrm{TE}} = 0.0$ is red):
- TM-like parity modes were indicated by red circles ($\sigma_z = -1$).
- However, the underlying all-bands squares for those exact same TM modes were colored **white, light blue, or dark blue** ($f_{\mathrm{TE}} \approx 0.55\text{–}0.99$), rather than the expected **red** ($f_{\mathrm{TE}} \approx 0.0$).
- This created a false visual appearance that the parity solver and all-bands solver were finding different modes, despite their eigenfrequencies matching to 4 decimal places.

---

## 2. Root Cause Analysis

### A. Volume Mismatch in 3D Slab Supercells
In a 3D slab simulation with normalized pitch $a = 0.45\ \mu\text{m}$:
- Silicon membrane thickness: $t = 0.22\ \mu\text{m}$ ($t/a \approx 0.49$)
- Supercell domain height: $s_z = 4.5\ a \approx 2.025\ \mu\text{m}$
- High-index silicon slab occupies **less than 10%** of the computational volume; the remaining **>90% is empty air/cladding buffer**.

### B. Field Symmetry & Boundary Conditions
For an odd ($z$-odd, TM-like) eigenmode:
- $\hat{\sigma}_z$ symmetry dictates that at the slab midplane ($z = 0$):
  $$E_x(z=0) = 0, \quad E_y(z=0) = 0, \quad E_z(z=0) \neq 0$$
- However, Maxwell's divergence equation $\nabla \cdot \mathbf{D} = 0$ requires:
  $$\partial_z D_z = -(i k_x D_x + i k_y D_y)$$
  Because the in-plane wavevector $k_{\parallel} \neq 0$, the transverse components $E_x, E_y$ **grow linearly away from the midplane** ($E_{x,y} \propto z$).
- In addition, at the dielectric slab boundary $z = \pm t/2$, continuity of normal displacement $D_z = \varepsilon E_z$ causes the electric field outside silicon to spike by a factor of $\varepsilon_{\text{core}}/\varepsilon_{\text{clad}} \approx 12 / 1 = 12$.
- When integrating $\int \operatorname{Re}(E_{xy}^* D_{xy})\, dV$ over the **entire tall air box** ($4.5\ a$), the accumulated transverse field energy in the air tails diluted $f_{\mathrm{TE}}$ to $0.55\text{–}0.65$ (and up to $0.99$ for leaky/higher-order cladding modes).
- On the `coolwarm_r` divergent colormap ($0.0 = \text{red}, 0.5 = \text{white}, 1.0 = \text{blue}$), any value $> 0.50$ is rendered light blue or blue, so TM-like modes never appeared red!

---

## 3. Generalization to Asymmetric Systems

In symmetric membranes ($\varepsilon(z) = \varepsilon(-z)$), parity $\sigma_z = \pm 1$ is a conserved symmetry operator. However, in realistic integrated photonic platforms (e.g. Silicon-on-Insulator with $\text{SiO}_2$ substrate and air cladding), **$z$-mirror symmetry is completely broken**:
$$[\hat{\Theta}, \hat{\sigma}_z] \neq 0$$

Therefore:
- **Option 2 (mirror parity $\sigma_z$) fails completely for asymmetric structures.** MPB cannot solve with `run_zeven` or `run_zodd` because parity is not a valid symmetry eigenvalue.
- **Option 1 (continuous polarization fraction) is the only universally valid formulation** for both symmetric membranes and asymmetric substrate stacks.

To ensure Option 1 correctly identifies TE-like and TM-like modes without empty-cladding dilution, the evaluation must probe the field inside the guiding core rather than the unbounded cladding.

---

## 4. Numerical Comparison: Volumetric vs. Midplane Evaluation

Evaluating the exact same eigenmodes at $k = M$ ($k_1 = 0, k_2 = 0.5, k_3 = 0$) confirms the resolution:

| Band | MPB Parity Mode | Frequency $\tilde{\omega}$ | Volumetric $f_{\mathrm{TE}}$ (Whole Box) | Midplane $f_{\mathrm{TE}}$ ($z = 0$) | Magnetic Ratio $H_z^2 / H_{\text{tot}}^2$ | True Modal Character |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Even (TE-like) | 0.2341 | 0.999 | **1.000** | 0.737 | **TE-like (Blue)** |
| **2** | Even (TE-like) | 0.3030 | 0.991 | **1.000** | 0.766 | **TE-like (Blue)** |
| **3** | Odd (TM-like)  | 0.3377 | 0.574 (diluted) | **0.021** | 0.005 | **TM-like (Pure Red)** |
| **4** | Odd (TM-like)  | 0.3475 | 0.637 (diluted) | **0.036** | 0.038 | **TM-like (Pure Red)** |
| **5** | Even (TE-like) | 0.3778 | 0.998 | **1.000** | 0.815 | **TE-like (Blue)** |
| **6** | Odd (TM-like)  | 0.4069 | 0.993 (diluted) | **0.675** | 0.367 | Cladding / Higher-order |
| **7** | Odd (TM-like)  | 0.4141 | 0.554 (diluted) | **0.018** | 0.089 | **TM-like (Pure Red)** |
| **8** | Even (TE-like) | 0.4182 | 0.963 | **1.000** | 0.793 | **TE-like (Blue)** |

At the midplane $z = 0$:
- TM modes yield $f_{\mathrm{TE}} \approx 0.018\text{–}0.036$ (**Deep Red**), matching the TM-like parity solve perfectly.
- TE modes yield $f_{\mathrm{TE}} = 1.000$ (**Deep Blue**), matching the TE-like parity solve perfectly.

---

## 5. Implementation & Permanent Resolution

In `packages/phc_mpb/src/phc_mpb/solver.py`:
1. Updated `compute_polarization_fractions` to accept an explicit `method` parameter:
   - `method="midplane"` (**new default**): Probes the midplane slice $z = z_{\text{mid}}$ of the guiding core. For 2D systems, this reduces identically to the 2D plane. For 3D slabs, it avoids tall-cladding dilution.
   - `method="volumetric"`: Preserves full 3D cell volume integration.
   - `method="slab"`: Restricts electric energy integration to the high-index slab region ($\varepsilon > 1.5$).
   - `method="magnetic"`: Evaluates continuous magnetic field energy $H_z^2 / (H_{xy}^2 + H_z^2)$.
2. Updated `run_parallel_band_solver` in `packages/phc_mpb/src/phc_mpb/parallel.py` to forward `polarization_method` to worker processes.
3. Updated `examples/demo_slab_3d_mode_parity_te_fraction_mpb.py` with the `--polarization-method` CLI flag (defaulting to `midplane`).

---

## 6. Supercell $z$-Periodicity Modes & Magnetic Metric Breakdown Near $\Gamma$

### A. Supercell $z$-Periodicity Radiation Modes
At and near the zone center $\Gamma$ ($\mathbf{k}_{\parallel} \to 0$), the light cone begins at $\omega = 0$.
Any state computed above $\omega > 0$ at $\Gamma$ is inside the light cone (unbound radiative continuum).
In physical open slabs, this forms an unquantized radiation continuum.
However, in supercell MPB simulations with vertical height $s_z$ and periodic boundary conditions in $z$:
- The continuum is artificially discretized into standing waves with $k_z \approx m \frac{2\pi}{s_z}$.
- For the mode around $\tilde{\omega} \approx 0.156\text{–}0.22$, field integration shows:
  - **95.0% of electromagnetic energy resides in the air/cladding**.
  - **18.6% of energy is localized directly on the top/bottom supercell boundaries**.
  - Slab confinement is only **5.0%**.
- Therefore, states above $\Gamma$ around $\tilde{\omega} \approx 0.16\text{–}0.22$ are **spurious supercell radiation/cladding modes** directly resulting from artificial $z$-periodicity.

### B. Breakdown of Magnetic Metric ($H_z$) Near $\Gamma$
For normal or near-normal incidence ($\mathbf{k}_{\parallel} \to 0$):
$$\nabla_{\parallel} \times \mathbf{E} \to 0 \implies H_z = \frac{1}{i\omega\mu_0}(\partial_x E_y - \partial_y E_x) \to 0$$
Even if a mode has 100% in-plane electric field ($E_x, E_y \neq 0, E_z = 0$, $\sigma_z = +1$ even parity), $H_z$ vanishes at $\Gamma$.
Consequently, the magnetic definition $f_{\mathrm{TE}} = |H_z|^2 / |\mathbf{H}|^2$ drops to $\approx 0.009$, falsely labeling pure TE-like modes as TM-like (red).
The **electric midplane metric** ($u_{E_{\parallel}} / u_{\text{tot}}$ at $z=0$) correctly evaluates to $f_{\mathrm{TE}} = 1.000$ (pure blue), preserving exact agreement with mirror parity.
