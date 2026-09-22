# Analysis of `phc_optimizer_old` Repository

## 1. Executive Summary & Purpose

The `phc_optimizer_old` (previously named `phc-nzi`) repository provides an automated Bayesian optimization and characterization framework for discovering and engineering accidental Dirac cones and Near-Zero-Index (NZI) regimes in photonic crystal (PhC) membranes and 2D lattices using the MIT Photonic Bands (MPB) eigensolver.

Key physical goals of the old repository:
1. Identify geometric parameter sets where a target multiplet of bands (e.g. $A_1 + E_1$ or $A_2 + E$ or triplet states) intersect degenerately at the high-symmetry $\Gamma$ point ($\mathbf{k} = \mathbf{0}$).
2. Minimize the frequency splitting (cost function) between these target bands to form a Dirac-like point with linear dispersion.
3. Extract group velocities $\vec{v}_g = \nabla_\mathbf{k} \omega(\mathbf{k})$ near $\Gamma$.
4. Trace continuous 1D degenerate manifold curves (loci) in multi-dimensional parameter space.
5. Verify physical fabricability constraints (such as minimum dielectric neck width and continuous slab connectivity across periodic boundary conditions).

---

## 2. Repository Architecture & File Inventory

```
phc_optimizer_old/
├── ctl/                               # MPB Scheme (.ctl) control files and helper modules
│   ├── materials.ctl                 # Material definitions (indices and epsilons)
│   ├── parity_functions.ctl          # Symmetry operator matrices and Scheme callbacks
│   ├── shapes.ctl                    # Primitive geometric shape definitions
│   ├── wyckoff.ctl                   # Wyckoff position coordinate tables
│   ├── lattices.ctl                  # Square and hexagonal lattice basis builders
│   └── custom_nonbloch_output.ctl   # Custom field export hooks
├── main.ctl                          # Master MPB simulation Scheme script
├── bo_config.yaml                    # Master YAML configuration for Bayesian Optimization
├── src/phc_nzi/                      # Python package modules
│   ├── bo.py                         # Core BayesianOptimizer class (skopt + Hydra)
│   ├── symmetry.py                   # Point-group character projections and irrep identification
│   ├── extractor.py                  # MPB log file parser (frequencies, symmetries, velocities)
│   ├── runner.py                     # HPC and local MPB subprocess execution wrapper
│   ├── locus.py                      # 1D degeneracy locus extraction & spline tracing
│   ├── geometry_check.py             # Dielectric slab topology & neck width validator
│   ├── utils.py                      # Common mathematical, image processing, and parsing utilities
│   ├── plotter.py                    # Matplotlib visualizers for band structures, epsilon, and loci
│   ├── slab_sweeper.py               # Normalized slab thickness (h/a) batch sweeper
│   ├── design_curves.py              # Scale-invariance universal design curve generator
│   └── workflow/                     # 5-step automated discovery pipeline
│       ├── step1_2d_screening.py
│       ├── step2_3d_screening.py
│       ├── step3_optimization.py
│       ├── step4_design_curves.py
│       ├── step5_validation.py
│       └── pipeline.py
```

---

## 3. Core Physics, Algorithms & Mathematical Formulations

### A. Figure of Merit (FOM) & Cost Function Formulation

The optimization minimizes the normalized frequency splitting between target bands at the $\Gamma$ point ($\mathbf{k} = \mathbf{0}$):

1. **Target Band Frequency Range**:
   Let $\mathcal{B}_{\text{target}} \subset \{1, 2, \dots, N_{\text{bands}}\}$ be the set of target band indices (e.g. $[2, 3, 4]$).
   $$f_{\text{high}} = \omega_{\max(\mathcal{B}_{\text{target}})}, \quad f_{\text{low}} = \omega_{\min(\mathcal{B}_{\text{target}})}$$

2. **Mid-Frequency Reference**:
   $$f_{\text{mid}} = \frac{f_{\text{high}} + f_{\text{low}}}{2}$$
   *(If $f_{\text{mid}} \le 0$, the frequency of the median/central band is used).*

3. **Normalized Cost**:
   $$\text{cost}_{\text{raw}} = |f_{\text{high}} - f_{\text{low}}|$$
   $$\text{cost}_{\text{norm}} = \frac{\text{cost}_{\text{raw}}}{f_{\text{mid}}} \quad (\text{if } f_{\text{mid}} > 0, \text{ else } \text{cost}_{\text{raw}})$$

4. **Target Cost Clamping**:
   An optional noise-floor threshold $C_{\text{target}}$ prevents over-fitting numerical jitter:
   $$C_{\text{eff}} = \max(\text{cost}_{\text{norm}}, C_{\text{target}}) \quad (\text{when } \text{cost}_{\text{norm}} < C_{\text{target}})$$

5. **Gaussian Process Objective (Minimization)**:
   In `"log"` objective mode (default):
   $$y(\mathbf{x}) = \log_{10}\left(\max(C_{\text{eff}}, 10^{-12})\right)$$

6. **Figure of Merit (FOM)**:
   The FOM represents the inverse residual gap:
   $$\text{FOM}(\mathbf{x}) = \frac{1}{\max(\text{cost}_{\text{norm}}, 10^{-12})}$$
   Under the GP surrogate with log-normal transformation ($y \sim \mathcal{N}(\mu, \sigma^2)$):
   $$\mathbb{E}[C] = 10^{\mu + \frac{\ln 10}{2}\sigma^2} \implies \text{FOM}_{\text{surrogate}} = \frac{1}{\max(\mathbb{E}[C], 10^{-12})}$$

---

### B. Point-Group Symmetry & Irreducible Representation (Irrep) Classification

At the zone center $\Gamma$ ($\mathbf{k} = \mathbf{0}$), eigenmodes transform according to point-group representations:
- **$C_{4v}$ (Square Lattice)**: Order $g = 8$. Conjugacy classes: $\{E (1), C_4 (2), C_2 (1), \sigma_v (2), \sigma_d (2)\}$.
  Irreps: $A_1 (1\text{D}), A_2 (1\text{D}), B_1 (1\text{D}), B_2 (1\text{D}), E (2\text{D})$.
- **$C_{6v}$ (Hexagonal Lattice)**: Order $g = 12$. Conjugacy classes: $\{E (1), C_6 (2), C_3 (2), C_2 (1), \sigma_v (3), \sigma_d (3)\}$.
  Irreps: $A_1 (1\text{D}), A_2 (1\text{D}), B_1 (1\text{D}), B_2 (1\text{D}), E_1 (2\text{D}), E_2 (2\text{D})$.

#### Character Projection Formula:
For single-state character expectation values $\chi_{\text{obs}}(R) = \langle \psi_b | \hat{R} | \psi_b \rangle$:
$$a_i = \frac{d_i}{g} \sum_{R \in \text{classes}} w_R \, \chi_i(R)^* \, \chi_{\text{obs}}(R)$$
where $d_i = \chi_i(E)$ is the representation dimension, $w_R$ is the conjugacy class weight, and $a_i \in [0, 1]$ is the projection confidence score.

#### Degenerate Mode Irrep Correction (`failsafe_irrep_mapping`):
When two or more eigenmodes are numerically degenerate (e.g. $E$ doublets or accidental Dirac triplets), MPB diagonalizes within an invariant subspace, yielding arbitrary orthogonal superpositions. Consequently:
- Non-commuting reflection operators ($\sigma_v, \sigma_d$) yield mixed expectation values on individual eigenvectors, depressing the single-band confidence score $a_i < 0.85$.
- The old implementation applies a failsafe heuristic:
  1. Identifies frequency clusters where adjacent bands satisfy $|\omega_{b} - \omega_{b+1}| < \delta\omega_{\text{deg}}$.
  2. If the cluster size matches target irrep dimension sum (e.g. $1 + 2 = 3$ for $A_1 + E$) and the majority of individual confidence scores are low ($< 0.85$), but prior parity rules hold (e.g. even number of $E$ modes before the cluster), it relabels the cluster bands to the target irreps.
- **Physical Invariant Improvement**: The trace over the cluster subspace $\text{Tr}_B(R) = \sum_{b \in B} \langle \psi_b | \hat{R} | \psi_b \rangle$ is strictly basis-invariant and can directly compute exact subspace decomposition without relying solely on heuristic relabeling.

---

### C. Group Velocity Extraction ($\vec{v}_g$)

1. **Analytical Momentum Operator in MPB**:
   Group velocity is evaluated via Hellmann-Feynman / $\mathbf{k} \cdot \mathbf{p}$ perturbation theory:
   $$\vec{v}_g = \nabla_\mathbf{k} \omega_n(\mathbf{k}) = \frac{\langle \psi_{n\mathbf{k}} | \nabla_\mathbf{k} \hat{\Theta}_\mathbf{k} | \psi_{n\mathbf{k}} \rangle}{2 \omega_n(\mathbf{k})}$$
   MPB provides direct built-in calls: `ms.compute_group_velocities()`.
2. **Dirac Cone Dispersion Evaluation**:
   At exact $\Gamma$ ($\mathbf{k}=\mathbf{0}$), linear Dirac bands have degenerate conical intersections where $v_g$ is isotropic and non-zero along radial directions $\hat{k}$, but undefined at the exact cusp tip. The old framework executes a second low-overhead evaluation at $\mathbf{k} = (\Delta k, 0, 0)$ (typically $\Delta k = 0.01$) to extract the Dirac slope $v_g = \frac{\partial \omega}{\partial k_\parallel}$.

---

### D. Dielectric Permittivity Topology & Connectivity Check

To prevent unphysical, floating, or disjoint dielectric structures during parameter exploration:
1. Evaluates MPB's 2D/3D dielectric grid $\varepsilon(\mathbf{r})$:
   $$\text{mask}(\mathbf{r}) = (\varepsilon(\mathbf{r}) > \varepsilon_{\text{threshold}})$$
2. Tiles the unit cell $3 \times 3$ (or $3 \times 3 \times 1$) to enforce periodic boundary conditions (PBC).
3. Applies mathematical morphological opening:
   $$\text{opened} = (\text{tiled\_mask} \ominus S) \oplus S$$
   using a structuring element of size `min_neck_width_px`. This tests whether neck regions between holes remain at least `min_neck_width_px` wide.
4. Uses `scipy.ndimage.label` to confirm that the central unit cell connects continuously across both $x$ and $y$ periodic domain boundaries.
5. If disconnected, assigns maximum penalty cost ($C = 1.0$) and marks status as `"FAILED: Disconnected slab"`.

---

## 4. Key Differences & Evolution for the New Architecture

| Feature | Old Implementation (`phc_optimizer_old`) | New Implementation (`phc_optimization` + `phc`) |
| :--- | :--- | :--- |
| **Geometry & Layout Engine** | MPB Scheme `.ctl` files with fixed parameters (`r1`, `r2`, `h`) | **GDSFactory** components from `phc_layout` (arbitrary parameter count and names) |
| **MPB Interface** | Shell command `mpb ... main.ctl > output.out` + regex log parsing | Direct Python `meep.mpb.ModeSolver` API (`packages/phc_mpb`) |
| **Symmetry Computation** | Scheme callbacks in `parity_functions.ctl` outputting text blocks | Native Python `ms.compute_symmetry` with $C_{4v}$/$C_{6v}$ symmetry matrices |
| **Degenerate Mode Correction**| Regex-based log parser with `failsafe_irrep_mapping` | Subspace trace projection $\text{Tr}_B(R) +$ failsafe mapping |
| **Group Velocity** | Parsed from `*velocity:` log output | Direct `ms.compute_group_velocities()` in Python |
| **Connectivity Checking** | Read `main-epsilon.h5` from disk | Direct in-memory inspection of `ms.get_epsilon()` or `phc_mpb.get_epsilon_grid` |
| **Output Directory Hierarchy** | Custom `bo_output/` | Monorepo standard `outputs/mpb/optimization/<geometry>/<timestamp>/` |
