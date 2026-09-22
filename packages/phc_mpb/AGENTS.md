# `phc_mpb` Guidelines & Rules

## Scope
`phc_mpb` interfaces with MIT Photonic Bands (MPB) via `pymeep` (`meep.mpb`) for photonic crystal band structures, band gap analysis, and dielectric field profiling.

## Core Rules

1. **Strictly No Fallback or Mock Calculations**:
   - Never generate sinusoidal dummy bands or synthetic permittivity grids if Meep fails.
   - If `pymeep` is not available or simulation parameters are invalid, raise `ImportError` or `ValueError` immediately.

2. **Generalization Requirements**:
   - **Lattice Types**: Support both `hexagonal` (triangular) and `square` lattices.
   - **Dimensions**:
     - `2D`: In-plane periodic rods or holes, infinite $z$ ($k_z = 0$).
     - `3D_slab`: Finite thickness PhC slab / membrane with supercell height in $z$.
   - **Polarizations**: Support `TE` and `TM` modes for 2D, and even/odd (TE-like / TM-like) symmetries for symmetric 3D slabs.

3. **Meep / MPB Specific Guidelines**:
   - **Import**: Always use `import meep as mp` (never `from meep import mp`).
   - **GDS Extraction**: Always use `klayout.db` in `extract_polygons_from_gds` with recursive shape iteration (`top.begin_shapes_rec(layer_idx)`). No fallback chains.
   - **Circular Geometries**:
     - For circular holes extracted from GDS layout, instantiate native `mp.Cylinder(radius=r, height=h, center=c, material=mat)`.
     - *Rationale*: Avoids libctl's C-library error `non-coplanar vertices in init_prism` caused by collinear vertices on digitized circle polygons.
   - **Epsilon Profiling**:
     - Always call `ms.init_params(mp.NO_PARITY, True)` before calling `ms.get_epsilon()`.
     - Support rectangular grid rectification and period replication (`periodicity=3`) for visualization.
   - **Reciprocal Space & High-Symmetry Paths**:
     - Triangular lattice basis:
       $\mathbf{a}_1 = (0.5\sqrt{3}, 0.5, 0)$, $\mathbf{a}_2 = (0.5\sqrt{3}, -0.5, 0)$.
       High-symmetry path: $\Gamma(0,0,0) \to \text{M}(0, 0.5, 0) \to \text{K}(-\frac{1}{3}, \frac{1}{3}, 0) \to \Gamma(0,0,0)$.
     - Square lattice basis:
       $\mathbf{a}_1 = (1, 0, 0)$, $\mathbf{a}_2 = (0, 1, 0)$.
       High-symmetry path: $\Gamma(0,0,0) \to \text{X}(0, 0.5, 0) \to \text{M}(0.5, 0.5, 0) \to \Gamma(0,0,0)$.

4. **Band Gap Analysis & Plotting**:
   - Automate complete omnidirectional bandgap search across all calculated bands.
   - Compute fractional band gap percentage:
     $$\Delta\omega / \omega_0 = 2(\omega_{\text{top}} - \omega_{\text{bottom}}) / (\omega_{\text{top}} + \omega_{\text{bottom}}) \times 100\%$$
   - Highlight identified band gaps on band diagrams with semi-transparent shading and percentage labels.
   - **Band Diagrams as Discrete Dots**: Band diagrams must always plot calculated eigenfrequencies as discrete dots (`marker="o"`, `linestyle="none"`), one dot per calculated $k$-point. Never connect eigenfrequencies with continuous lines. Analytical thresholds (such as the light line) remain dashed lines.
   - **Epsilon Profiling Without Interpolation**: Permittivity grid plots (`plot_epsilon`) must use `interpolation="none"` by default to visualize exact discrete grid voxels without artificial smoothing.
   - **3D Slab Permittivity Profiling**: For 3D slabs, permittivity distribution ($\varepsilon(\mathbf{r})$) plotting must generate a single unified figure with two subplots side by side:
     - **Left Subplot**: In-plane $xy$ mid-plane slice ($z = z_{\text{mid}}$ / $z = 0$).
     - **Right Subplot**: Vertical $xz$ cross-section cut ($y = y_{\text{mid}}$) displaying the slab thickness, vertical air cladding, and etch hole profile.
     - **Vertical Periodicity Standard**: Vertical periodicity along $z$ must strictly remain 1 (`periods_z=1`) so that only a single physical slab membrane is displayed without artificial vertical repetitions.
