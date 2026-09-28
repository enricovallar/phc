# Issue 003: MPB Degeneracy Mode-Mixing, Subspace Invariance, and Irreducible Representation Resolution

## Date
2026-09-25

## Status
SOLVED

---

## 1. Problem Statement & Symptoms

When calculating point-group irreducible representations at the zone center ($\Gamma$, $k = 0$) for high-symmetry photonic crystals—such as the $C_{6v}$ 100 nm hBN membrane in `analysis/test_c6v_2b_6d.py`—the generated ASCII irreps table (`irreps.data`) and results JSON exhibited two critical anomalies:

```text
# kmag/2pi    band_index  frequency     parity
  0.000000    8           0.828353      E_1      <-- ANOMALY 1: Isolated mode labeled E_1 without a doublet partner
  0.000000    9           0.863638      E_1      <-- ANOMALY 2: Accidental triplet misclassified as E_1, B_1, A_2
  0.000000    10          0.866373      B_1          instead of E_1, E_1, A_2
  0.000000    11          0.868403      A_2
```

1. **Non-Degenerate Singlet Labeled as 2D Doublet (`E_1`)**:
   Band 8 is an isolated mode at $\tilde{\omega} = 0.828353$ ($\lambda \approx 460\text{ nm}$), separated by $>4\%$ in frequency from neighboring bands. By fundamental group theory, an isolated singlet cannot transform as a 2-dimensional representation ($E, E_1, E_2$).
2. **Accidental Triplet Misclassified (`E_1, B_1, A_2` vs. `E_1, E_1, A_2`)**:
   Bands 9, 10, and 11 form an accidental Dirac-like degeneracy at $\tilde{\omega} \approx 0.866$ ($\lambda \approx 440\text{ nm}$), composed of a 2D doublet and a 1D singlet ($E_1 \oplus A_2$, 3 modes total). Band 10 was falsely labeled as `B_1` with low confidence ($0.522$), splitting the doublet.

---

## 2. Root Cause Analysis

### A. Single-State Projections in Multi-Dimensional Representations
In MPB, `ms.compute_symmetry(b, op, origin)` evaluates the expectation value of a symmetry operator $\hat{R}$ for an **individual eigenstate**:
$$D_{bb}(R) = \langle \psi_b | \hat{R} | \psi_b \rangle$$

However, group character tables tabulate the **trace** over the full invariant subspace:
$$\chi^{(i)}(R) = \text{Tr}[D(R)] = \sum_{m=1}^{d_i} D_{mm}(R)$$

For a 2D representation ($E_1$ in $C_{6v}$), the two basis states ($|x\rangle$ and $|y\rangle$) have opposite parity under diagonal reflection $\sigma_d$:
- Component 1 ($|x\rangle$, Band 9): $\langle \psi_9 | \hat{\sigma}_d | \psi_9 \rangle \approx +1$
- Component 2 ($|y\rangle$, Band 10): $\langle \psi_{10} | \hat{\sigma}_d | \psi_{10} \rangle \approx -1$

Both states have $\langle C_2 \rangle \approx -1$. When evaluated on a single-band basis, Band 10 had:
$$\langle C_2 \rangle = -0.946, \quad \langle \sigma_d \rangle = -0.999$$
In the $C_{6v}$ character table, the 1D irrep $B_1$ has $\chi(C_2) = -1$ and $\chi(\sigma_d) = -1$. Consequently, Band 10 projected onto $B_1$ with score $0.5221$ and onto $E_1$ with score $0.5192$. Because $0.5221 > 0.5192$, the single-state classifier selected `B_1`, masking the $E_1$ doublet.

### B. Representation Dimension Bias ($d_i$)
The raw projection formula implemented in `compute_projections`:
$$a_i = \frac{d_i}{g} \sum_{R} w_R \, \text{Re}[\chi_i(R)^* \chi_{\text{obs}}(R)]$$
multiplied the score by the irrep dimension $d_i$. For 2D irreps ($d_i = 2$), this doubled the raw overlap relative to 1D irreps. For Band 8, the raw $E_1$ overlap was $0.468$, which became $2 \times 0.468 = 0.937$, artificially promoting $E_1$ above the true 1D irrep ($A_1$).

### C. MPB Callback Frequency Timing
In parallel mode, when the $\Gamma$ callback (`_sym_cb`) was executed, `ms.all_freqs` was not yet populated on the worker solving the first k-point of its chunk, resulting in `freq = 0.0` for all bands in the raw symmetry records. Without frequencies, downstream clustering could not resolve degeneracies.

### D. Deficiency of the Legacy Method (`failsafe_irrep_mapping`)
The older codebase used a heuristic relabeler (`failsafe_irrep_mapping`) that required explicit knowledge of `target_irreps` (e.g. during optimization). It was inapplicable to general multi-band dispersion simulations where target irreps are unknown in advance.

---

## 3. Permanent Resolution

### 1. Invariant Multiplet Resolution (`resolve_multiplet_symmetries`)
Implemented `resolve_multiplet_symmetries` in `packages/phc_mpb/src/phc_mpb/symmetry.py`:
- **Subspace Clustering**: Clusters bands at $\Gamma$ based on frequency proximity ($|\Delta\omega| / \omega < \text{degeneracy\_tol}$).
- **Singlet Protection**: For clusters of size 1, restricts classification to 1D irreps ($A_1, A_2, B_1, B_2$). Band 8 is correctly classified as `A_1`.
- **Doublet Subspace Trace**: For clusters of size 2, evaluates the unitary-invariant subspace trace $\chi(R) = \langle \psi_1 | R | \psi_1 \rangle + \langle \psi_2 | R | \psi_2 \rangle$. The trace yields $\chi(C_2) = -2.0$ and $\chi(\sigma_d) = 0.0$, matching pure $E_1$ with high confidence.
- **Accidental Triplet Decomposition**: For clusters of size 3 (such as Bands 9, 10, 11), evaluates all partitions into a candidate singlet $s$ and doublet pair $\{d_1, d_2\}$. It chooses the partition maximizing total projection score:
  - Singlet: Band 11 $\to A_2$ (conf 0.603)
  - Doublet: Bands 9 & 10 $\to E_1, E_1$ (conf 0.616)

### 2. Live Frequency Retrieval in Solvers
Updated `compute_band_symmetries` to query `ms.freqs` or `ms.get_freqs()` directly at the callback, ensuring eigenfrequencies are immediately available for clustering even during parallel execution.

### 3. Safely Removed Legacy Method
- Removed `failsafe_irrep_mapping` and its unit test.
- Migrated `find_bands_from_irreps` (used by `phc_optimization`) to use `resolve_multiplet_symmetries`.

---

## 4. Verification & Output Validation

The corrected `irreps.data` output now faithfully reflects the physical group theory of the structure:

```text
# kmag/2pi    band_index  frequency     parity  
  0.000000    1           0.000000      A_2     
  0.000000    2           0.229061      E_1     
  0.000000    3           0.229100      E_1     
  0.000000    4           0.464081      E_1     
  0.000000    5           0.464113      E_1     
  0.000000    6           0.704825      E_1     
  0.000000    7           0.704902      E_1     
  0.000000    8           0.828353      A_1     <-- Correctly resolved as 1D singlet
  0.000000    9           0.863638      E_1     <-- Triplet component 1: E_1
  0.000000    10          0.866373      E_1     <-- Triplet component 2: E_1
  0.000000    11          0.868403      A_2     <-- Triplet component 3: A_2
  0.000000    12          0.955632      E_1     
  0.000000    13          0.955874      E_1     
  0.000000    14          0.995999      E_2     
  0.000000    15          0.998131      E_2     
```

### Test Suite Execution
- `pytest packages/phc_mpb/tests/`: 53 passed in 11.69s.
- `pytest packages/phc_optimization/tests/`: 33 passed in 17.10s.
- `ruff check packages/phc_mpb/`: 0 errors, 0 warnings.
