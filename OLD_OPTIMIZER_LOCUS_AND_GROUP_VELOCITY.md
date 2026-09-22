# Locus Extraction, Refinement, and Group Velocity in `phc_optimizer_old`

## 1. Executive Summary

In [`phc_optimizer_old`](https://github.com/enricovallar/phc_optimizer_old), the postprocessing pipeline was designed to discover and characterize accidental Dirac-like degeneracies and near-zero-index (NZI) regimes in photonic crystal (PhC) unit cells. 

When Bayesian Optimization identifies a high-Figure-of-Merit (FOM) region, the optimal solutions typically do not form isolated zero-dimensional points, but rather **continuous 1D manifolds (loci)** in multi-dimensional geometric parameter space (e.g. $r_1$ vs. $r_2$). 

This document details how the old implementation:
1. **Isolated a single locus** when configured with `max_loci: 1` (or in polar ring mode).
2. **Refined the sampled locus points** to exact $\Gamma$-point degeneracy using a fast 1D normal line search.
3. **Calculated group velocity at an adjacent point** ($\mathbf{k} = (\Delta k, 0, 0)$) rather than at exact $\Gamma$ ($\mathbf{k} = \mathbf{0}$) due to the non-differentiable Dirac cone cusp.
4. **Plotted and exported the group velocity profile** across the refined locus.

---

## 2. Architecture & File Inventory

The locus and group velocity workflow spans Scheme control files (`ctl/`), Python postprocessing modules (`src/phc_nzi/`), and Hydra configuration files:

| Layer / File | Responsibility |
| :--- | :--- |
| [`ctl/init.ctl`](file:///home/enrva/phc/scratch/phc_optimizer_old/ctl/init.ctl) | Declares parameters `delta-k-mode?`, `delta-k`, `display-group-velocity?`. |
| [`ctl/parity_functions.ctl`](file:///home/enrva/phc/scratch/phc_optimizer_old/ctl/parity_functions.ctl) | Appends MPB's built-in callback `display-group-velocities` to solver callbacks. |
| [`main.ctl`](file:///home/enrva/phc/scratch/phc_optimizer_old/main.ctl) | Overrides the simulation k-path to evaluate single point $\mathbf{k} = (\Delta k, 0, 0)$ when `delta-k-mode?` is enabled. |
| [`configs/postprocessing/default.yaml`](file:///home/enrva/phc/scratch/phc_optimizer_old/configs/postprocessing/default.yaml) | Specifies `max_loci: 1`, `threshold_percentile`, refinement tolerance, and $\Delta k$. |
| [`src/phc_nzi/locus.py`](file:///home/enrva/phc/scratch/phc_optimizer_old/src/phc_nzi/locus.py) | Implements Zhang-Suen thinning, polar/Cartesian ridge extraction, normal vector computation, and 3-panel locus plotting. |
| [`src/phc_nzi/bo.py`](file:///home/enrva/phc/scratch/phc_optimizer_old/src/phc_nzi/bo.py) | Coordinates single-point secant refinement (`_refine_single_point_degeneracy`), parallel execution, and MPB group velocity dispatch. |
| [`src/phc_nzi/extractor.py`](file:///home/enrva/phc/scratch/phc_optimizer_old/src/phc_nzi/extractor.py) | Parses MPB `velocity:, ...` text output from simulation logs into structured velocity tensors. |

---

## 3. Isolating a Single Locus (`max_loci: 1`)

In parameter space, the surrogate Figure of Merit is defined from the Gaussian Process model as the inverse expected residual cost:
$$\text{FOM}(\mathbf{x}) = \frac{1}{\max(\mathbb{E}[C(\mathbf{x})], 10^{-12})}$$

The extraction of a single continuous 1D locus from a 2D surrogate grid was handled in [`src/phc_nzi/locus.py`](file:///home/enrva/phc/scratch/phc_optimizer_old/src/phc_nzi/locus.py) via two complementary modes:

### A. Cartesian Skeletonization Mode (`extract_optimal_loci`)

When `mode: "cartesian"` (or default manifold mode):
1. **Percentile Cutoff & Binary Morphological Closing**:
   ```python
   cutoff = float(
       np.percentile(finite_fom, threshold_percentile)
   )  # e.g. 85th or 90th percentile
   mask = (fom_2d >= cutoff) & np.isfinite(fom_2d)
   mask = ndi.binary_closing(mask)
   ```
2. **Connected Component Labeling & Ranking**:
   The binary mask is partitioned into distinct connected islands using 8-connectivity:
   ```python
   labeled, num_features = ndi.label(mask)
   ```
   Each component is evaluated for pixel area and maximum FOM. Components smaller than `min_locus_area_px` are dropped. The surviving components are sorted descending by maximum FOM:
   ```python
   components.sort(key=lambda c: c[2], reverse=True)
   ```
3. **Truncation by `max_loci: 1`**:
   The list of components is explicitly sliced to retain only the top component:
   ```python
   selected_components = components[
       :max_loci
   ]  # When max_loci == 1, exactly 1 component is selected
   ```
4. **Zhang-Suen Topological Thinning**:
   The selected component mask is reduced to a 1-pixel-wide central skeleton using the Zhang-Suen morphological thinning algorithm (`zhang_suen_thinning`), preserving 8-connectivity and endpoints.
5. **Path Ordering & Spline Smoothing**:
   - `order_skeleton_points`: An adjacency graph connects neighbouring skeleton pixels ($\le 1.5 \sqrt{\Delta x^2 + \Delta y^2}$). Starting from a degree-1 endpoint (or arbitrary pixel for closed loops), a greedy nearest-neighbor traversal produces an ordered coordinate sequence.
   - `scipy.interpolate.splprep` / `splev`: Fits a parametric B-spline curve $\mathbf{r}(u) = (r_1(u), r_2(u))$ parameterized by normalized parameter $u \in [0, 1]$ across $N$ sample points (`sample_points: 40` or `50`).

### B. Polar Ray Tracing Mode (`extract_polar_ring_locus`)

For annular or circular degeneracy manifolds (such as symmetric core-shell or concentric cylinder lattices), polar ray tracing was used:
1. Centroid calculation $\mathbf{x}_c = (x_{1c}, x_{2c})$ of the top 15% FOM points.
2. Casting radial rays at angular increments $\theta_i \in [0, 2\pi)$.
3. Finding the maximum-FOM radial coordinate $r_{\max}(\theta_i)$ along each ray via cubic interpolation (`ndi.map_coordinates`).
4. Fitting a closed periodic cubic B-spline (`per=True`).
5. **Inherently yields exactly one closed locus** (`locus_id: 1`).

---

## 4. Local Locus Refinement (Gamma-Only Secant Search)

Points sampled along the GP surrogate spline only approximate the true continuous degeneracy line due to finite GP grid spacing and interpolation uncertainty.

To achieve exact degeneracy ($\text{gap} < 10^{-5}$), [`_refine_locus_degeneracy`](file:///home/enrva/phc/scratch/phc_optimizer_old/src/phc_nzi/bo.py#L1758-L1942) and [`_refine_single_point_degeneracy`](file:///home/enrva/phc/scratch/phc_optimizer_old/src/phc_nzi/bo.py#L1637-L1757) refine each point:

```
               Locus Spline Curve r(t)
               ────────────────────────●──────────────────────
                                       │
                                       │  Normal Vector n
                                       ▼
                       Secant Search at Gamma (k=0)
                       f_high(n) - f_low(n) -> 0
```

### A. Curve Normal Vector Computation
For each sampled point $(r_{1,i}, r_{2,i})$, the local tangent vector $\vec{t}_i$ is computed via central finite differences of the spline coordinates:
$$\vec{t}_i = \left(\frac{dr_1}{dt}, \frac{dr_2}{dt}\right), \quad \vec{n}_i = \frac{(-t_{2,i}, t_{1,i})}{\|\vec{t}_i\|}$$

### B. Fast $\Gamma$-Only 1D Secant Line Search
Along the normal direction $\vec{n}_i$, a 1D coordinate offset $\delta$ is probed:
$$\mathbf{x}(\delta) = \mathbf{x}_0 + \delta \, \vec{n}_i$$

1. **High-Speed Evaluation**: The simulation is executed with `only_gamma? = true`, restricting eigensolver execution strictly to $\mathbf{k} = (0, 0, 0)$. This avoids computing the entire Brillouin zone k-path, speeding up each evaluation by 10x–50x.
2. **Signed Cost & Gap**: The signed frequency gap between target multiplet bands is monitored:
   $$\Delta \omega(\delta) = \omega_{\max(\text{target})} - \omega_{\min(\text{target})}, \quad \text{Cost} = \frac{|\Delta \omega(\delta)|}{\omega_{\text{mid}}}$$
3. **Secant Iteration**:
   After an initial step $\delta_1 = 0.002$:
   $$\delta_{k+1} = \delta_k - \Delta \omega_k \frac{\delta_k - \delta_{k-1}}{\Delta \omega_k - \Delta \omega_{k-1}}$$
   bounded within $[-\delta_{\max}, \delta_{\max}]$ ($\delta_{\max} = 0.035$).
4. **Termination**: The search terminates when $\text{Cost} < \text{tol}$ (typically $10^{-5}$) or when `max_steps` (typically 5 to 10) is exhausted.

### C. Pruning and Arc Preservation (`exclude_unrefined: true`)
If a point cannot converge below `max_residual_gap` (e.g. $10^{-4}$, caused by structural self-intersection or boundary clipping), it is flagged as `is_valid: False`.
When points are pruned from an annular loop, the remaining valid indices are rotated so the locus forms a continuous open arc rather than having an unphysical jump across the gap.

---

## 5. Group Velocity at an Adjacent Point ($\mathbf{k} \neq \mathbf{0}$)

### A. Physical Rationale: Why Not at $\Gamma$?

At an accidental Dirac-like point, three or more bands cross degenerately at $\mathbf{k} = \mathbf{0}$ ($\Gamma$):
- One flat or quadratic band (e.g. $A_1$ or $A_2$).
- Two linear conical dispersion sheets (e.g. $E_1$ or $E$).

Near $\Gamma$, the linear bands disperse as:
$$\omega(\mathbf{k}) \approx \omega_D \pm v_g |\mathbf{k}|$$

At the exact Dirac apex ($\mathbf{k} = \mathbf{0}$):
1. **Mathematical Singularity**: The dispersion surface forms a non-differentiable conical cusp tip. The gradient $\nabla_\mathbf{k} \omega$ is ill-defined (directional derivatives depend on approach direction, and multiple eigenmodes mix arbitrarily).
2. **Eigenmode Mixing**: In MPB, the eigensolver diagonalizes degenerate subspaces arbitrarily. Hellmann-Feynman perturbation theory:
   $$\vec{v}_g = \nabla_\mathbf{k} \omega_n(\mathbf{k}) = \frac{\langle \psi_{n\mathbf{k}} | \nabla_\mathbf{k} \hat{\Theta}_\mathbf{k} | \psi_{n\mathbf{k}} \rangle}{2 \omega_n(\mathbf{k})}$$
   evaluates to zero or indeterminate values at the exact stationary/cusp point $\mathbf{k} = \mathbf{0}$.

**Solution**: Shift $\mathbf{k}$ by a small offset $\Delta k$ along the radial direction away from $\Gamma$:
$$\mathbf{k}_{\text{adj}} = (\Delta k, 0, 0)$$
Typically $\Delta k = 0.001$ or $0.01$ (in units of $2\pi/a$). At $\mathbf{k}_{\text{adj}}$, the linear branches have split cleanly from the apex, allowing MPB's Hellmann-Feynman solver to return the genuine radial slope $v_g = \frac{\partial \omega}{\partial k_x}$.

### B. Implementation in Scheme MPB Scripts

1. In [`ctl/init.ctl`](file:///home/enrva/phc/scratch/phc_optimizer_old/ctl/init.ctl#L43-L48):
   ```scheme
   ; Dynamic delta-k group velocity calculation flags
   (define-param delta-k-mode? false)
   (define-param delta_k_mode? false)
   (define-param delta-k 0.01)
   (define-param delta_k 0.01)
   ```

2. In [`main.ctl`](file:///home/enrva/phc/scratch/phc_optimizer_old/main.ctl#L51-L55):
   ```scheme
   (define-param delta-k 0.01)
   ; Override k-points for small delta-k group velocity run
   (if (or delta-k-mode? delta_k_mode?)
       (set! k-points (list (vector3 delta-k 0 0))))
   ```

3. In [`ctl/parity_functions.ctl`](file:///home/enrva/phc/scratch/phc_optimizer_old/ctl/parity_functions.ctl#L69-L84):
   ```scheme
   (define-param display-group-velocity? false)
   (define-param display_group_velocity? false)

   (define (run-solver-with-callbacks solver)
     ...
     (let ((callbacks '()))
       (if (or display-symmetry? display_symmetry?)
           (set! callbacks (append callbacks (list display-symmetries))))
       (if (or display-group-velocity? display_group_velocity?)
           (set! callbacks (append callbacks (list display-group-velocities))))
       (apply solver callbacks)))
   ```

### C. Python Orchestration in `bo.py`

When postprocessing evaluates the refined locus points ([`bo.py` lines 2110–2188](file:///home/enrva/phc/scratch/phc_optimizer_old/src/phc_nzi/bo.py#L2110-L2188)):
```python
vg_params = {**self.fixed_params, **p_dict}
vg_params["display_symmetry?"] = "false"
vg_params["display_group_velocity?"] = "true"
vg_params["only_gamma?"] = "false"
vg_params["delta_k_mode?"] = "true"
vg_params["delta_k"] = delta_k  # e.g. 0.001

run_hpc(
    script=self._get_script_path(),
    mpb_command_line_params=vg_params,
    use_mpi=False,
    cores=1,
    wd=vg_dir,
    auto_extract=True,
    verbose=False,
)
```

The output log is parsed via `phc_nzi.extractor.extract_group_velocities`:
```python
vg_data = extract_group_velocities(output_path=vg_log, save_data=True)
# Target group velocity extracted along x-direction for target multiplet bands:
target_vgs = [
    float(rec["vx"])
    for rec in vg_data["flat_records"]
    if rec["band"] in target_bands and rec["parity"] == polarization
]
vg_val = float(max(target_vgs))
```

---

## 6. Plotting the Group Velocity of the Refined Points

Once all refined points along a locus are simulated, [`plot_locus_profiles`](file:///home/enrva/phc/scratch/phc_optimizer_old/src/phc_nzi/locus.py#L508-L634) generates a unified 3-panel figure (`bo_locus_profile.png`):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Optimal Locus Analysis                                │
├─────────────────────────┬─────────────────────────┬─────────────────────────┤
│ (a) Parameter Space     │ (b) Group Velocity      │ (c) Figure of Merit     │
│                         │                         │                         │
│   r2/a                  │   vg / c                │   FOM (log scale)       │
│    ▲                    │    ▲                    │    ▲                    │
│    │  ●--●--●           │    │  /\                │    │   -------------    │
│    │ /   (color=vg)     │    │ /  \               │    │  /             \   │
│    │●           ●       │    │/    \              │    │ /               \  │
│    └─────────────► r1/a │    └──────────────► t   │    └─────────────► t    │
│    Colorbar: vg/c       │     t in [0, 1]         │     t in [0, 1]         │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

1. **Panel (a) Parameter Space Trajectory**:
   - Plots the trajectory in $(r_1/a, r_2/a)$ space.
   - Points are plotted as a scatter plot **color-coded by group velocity $v_g / c$** using the `plasma` colormap:
     ```python
     sc = ax_param.scatter(
         r1_vals,
         r2_vals,
         c=vg_vals,
         cmap="plasma",
         s=40,
         edgecolors="black",
         linewidths=0.5,
         zorder=4,
     )
     cbar = fig.colorbar(sc, cax=cax)
     cbar.set_label(r"$v_g / c$", fontsize=10, fontweight="bold")
     ```
   - Start ($t = 0$) is annotated with a green circle and End ($t = 1$) with a red square.
2. **Panel (b) Group Velocity Profile**:
   - Plots $v_g(t) / c$ against normalized curve arc length $t \in [0, 1]$:
     ```python
     ax_vg.plot(t_vals, vg_vals, "-o", color="#0070C0", linewidth=2.0, markersize=4.5)
     ax_vg.set_ylabel(r"Group Velocity $v_g / c$")
     ax_vg.set_xlabel(r"Normalized Trajectory $t \in [0, 1]$")
     ```
3. **Panel (c) Figure of Merit Profile**:
   - Plots $\text{FOM}(t)$ on a logarithmic scale to verify that degeneracy remains uniform across the entire curve.
4. **Tabular Export**:
   Exports `bo_locus.csv` with columns:
   `[point_idx, t_norm, r1, r2, fom, residual_gap, group_velocity, is_valid]`.
