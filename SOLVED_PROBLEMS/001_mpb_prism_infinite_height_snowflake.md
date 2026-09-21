# Issue 001: Floating-Point Cancellation in MPB 2D Prism Extrusion (Snowflake Hole Inversion)

## Date
2026-09-21

## Status
SOLVED

---

## 1. Problem Statement & Symptoms

When running band structure calculations for unit cells defined by non-circular polygons (specifically `c6v_painter_snowflake`), the generated permittivity map (`epsilon_map.png`) exhibited severe visual distortion:
- Discrete 6-pointed star / snowflake air holes were completely absent.
- Instead, the image displayed a periodic network of thin yellow lines ($\varepsilon = 12$, silicon) enclosing large rhombic purple voids ($\varepsilon = 1$, air).
- The air fill fraction in MPB was measured at **$71.2\%$**, whereas the true geometrical fill fraction of the snowflake hole in GDS was only **$34.8\%$**.

---

## 2. Root Cause Analysis

The conversion from GDS polygons to MPB simulation objects is handled by `packages/phc_mpb/src/phc_mpb/converter.py`. Circular holes are identified and converted to `mp.Cylinder`, which uses an exact analytical radial formula $\sqrt{\Delta x^2 + \Delta y^2} \le r$ and is immune to polygon clipping issues. Arbitrary polygons, however, are converted into `mp.Prism`.

In the previous implementation of `converter.py`, 2D simulations configured `mp.Prism` as:
```python
if dimension == "2D":
    prism_height = mp.inf  # In Meep, mp.inf == 1e+20
    prism_center = mp.Vector3(0, 0, 0)
...
prism = mp.Prism(
    vertices=v_list,
    height=prism_height,
    center=p_center,  # Vector3(0, 0, 0)
    material=medium,
)
```

Inside Meep's internal `Prism` constructor (`meep/geom.py`):
```python
centroid = sum(vertices, Vector3(0)) * (1.0 / len(vertices))
original_center = centroid + (0.5 * height) * axis
if center is not None and len(vertices):
    center = Vector3(*center)
    shift = center - original_center
    vertices = list(map(lambda v: v + shift, vertices))
```

When `center = Vector3(0, 0, 0)` is passed alongside `height = 1e+20`:
$$\text{shift} = \mathbf{0} - \left(\text{centroid} + \frac{1}{2} \cdot 10^{20} \hat{\mathbf{z}}\right) = -5 \times 10^{19} \hat{\mathbf{z}}$$

This translated the $z$-coordinate of every polygon vertex from $z = 0$ to $z = -5 \times 10^{19}$.

### Catastrophic Floating-Point Cancellation
Standard 64-bit IEEE 754 double precision floating-point numbers provide 53 bits of mantissa ($\approx 15\text{–}17$ decimal digits). At a magnitude of $5 \times 10^{19}$, the distance between representable floating-point numbers (machine epsilon) is:
$$\epsilon_{\text{mach}}(5 \times 10^{19}) = 5 \times 10^{19} \times 2^{-52} \approx 1.11 \times 10^4 = 11,100$$

The unit cell grid coordinates in MPB lie in the interval $[-0.5, +0.5]$ (a spatial scale of $1.0$). At $z \approx -5 \times 10^{19}$, differences on the order of $1.0$ completely lose all precision.

When Meep's C++ polyhedron ray-tracing kernel evaluated whether grid points $(x, y, z \approx 0)$ were inside the prism:
1. The ray-triangle / edge clipping algorithms suffered catastrophic numerical breakdown.
2. The inside/outside winding test failed, erroneously classifying the entire unit cell interior as "inside" the prism (air, $\varepsilon = 1$).
3. Only points right at the lattice cell periodic boundaries hit boundary conditions, persisting as the silicon grid lines ($\varepsilon = 12$).

---

## 3. Direct Numerical Proof

Testing various values of `height` in `mp.Prism` while retaining `center = Vector3(0, 0, 0)` confirmed the breakdown threshold:

| Prism Height $h$ | Air Fill Fraction in MPB | Permittivity Map Result |
| :--- | :--- | :--- |
| **$h = 1.0$** | **$30.9\%$** | Crisp, discrete 6-pointed snowflake holes |
| **$h = 10.0$** | **$30.9\%$** | Crisp, discrete 6-pointed snowflake holes |
| **$h = 10^{3}$** | **$30.9\%$** | Crisp, discrete 6-pointed snowflake holes |
| **$h = 10^{6}$** | **$30.9\%$** | Crisp, discrete 6-pointed snowflake holes |
| **$h = 10^{10}$** | **$30.9\%$** | Crisp, discrete 6-pointed snowflake holes |
| **$h = 10^{15}$** | **$71.2\%$** | Floating-point cancellation (inverted rhombus) |
| **$h = 10^{20}$ (`mp.inf`)** | **$71.2\%$** | Floating-point cancellation (inverted rhombus) |
| **$h = \infty$, `center=None`** | **$30.9\%$** | Crisp, discrete 6-pointed snowflake holes |

---

## 4. Solution Implemented

In `packages/phc_mpb/src/phc_mpb/converter.py`:
1. **Omit `center` for `mp.Prism`**:
   - For `mp.Prism`, polygon vertices extracted from GDS already represent the exact, true in-plane coordinates $(x, y)$.
   - By omitting `center` (or passing `center=None`), Meep does **not** shift the polygon vertices in $(x, y)$ or $z$.
2. **Explicit Base Z Positioning**:
   - In 2D: `z_base = 0.0`, `height = mp.inf`. Vertices stay at $z = 0.0$ and extrude infinitely along $+z$, which covers the full 2D MPB simulation domain without any precision loss.
   - In 3D: `z_base = (z_center - 0.5 * slab_thickness) / pitch`, `height = slab_thickness / pitch`. Vertices start exactly at the bottom of the slab and extend to the top without any translation shift applied to $(x, y)$.

---

## 5. Verification

1. Unit tests pass with `pytest`.
2. Execution of `examples/simulate_unit_cell.py geometry=c6v_painter_snowflake`:
   - `epsilon_map.png` displays sharp, discrete 6-pointed snowflake holes centered in the unit cell.
   - MPB fill fraction accurately matches the theoretical GDS value ($\approx 31\%$).
