# Issue 004: Asymmetric Slab Subspace Irrep Mixing and Target-Guided Multiplet Resolution

## Date
2026-10-01

## Status
SOLVED

---

## 1. Problem Statement & Symptoms

During Bayesian optimization sweeps for accidental Dirac-like degeneracies in 3D photonic crystal slabs (e.g. hBN or Si membranes on $\text{SiO}_2$ substrate under $C_{6v}$ symmetry, targeting $A_2 \oplus E_1$), surrogate landscape maps (`bo_surrogate_map.png`) exhibited isolated "crater" anomalies:

- **Isolated Low-FOM Outliers**: High-FOM regions ($\text{FOM} \approx 0.85 - 0.95$) contained sporadic points whose FOM collapsed to $\approx 0.010$.
- **Spurious Penalty Triggering**: Inspection of the objective function breakdown revealed that these points incurred the maximum irrep mismatch penalty ($\text{penalty} = +100.0$), forcing $\text{FOM} = \exp(-\text{total\_loss}) \approx \exp(-4.6) \approx 0.010$.
- **Physical Validity**: Field analysis and frequency spectra confirmed that these points actually had the desired physical triplet degeneracy ($\Delta\omega / \omega < 0.001$) and high modal overlap. However, the classifier failed to recognize the $A_2 \oplus E_1$ symmetry, labeling the bands as e.g. $\{E_1, E_1, B_1\}$, $\{E_2, E_2, B_1\}$, or three $E_1$ states.

---

## 2. Root Cause Analysis

### A. Eigensolver Arbitrary Unitary Subspace Rotation
In degenerate or nearly-degenerate eigenspaces ($|\Delta\omega|/\omega \ll 1$), iterative conjugate-gradient eigensolvers (such as MPB's block-Davidson solver) compute an arbitrary unitary rotation of the degenerate basis:
$$|\psi'_i\rangle = \sum_{j} U_{ij} |\psi_j\rangle$$
While the trace over the invariant subspace $\sum_i \langle\psi'_i|\hat{R}|\psi'_i\rangle$ is strictly invariant under $U$, any single-band expectation value $\langle\psi'_i|\hat{R}|\psi'_i\rangle$ is not. Single-state projection scores are therefore corrupted by cross-band mode mixing.

### B. Vertical Asymmetry and Substrate Perturbation
In 3D slab architectures with asymmetric vertical claddings (e.g., air superstrate and $\text{SiO}_2$ substrate), horizontal mirror symmetry ($\sigma_h$) is broken. Furthermore, discrete Yee-grid numerical approximation introduces slight perturbations into in-plane reflection and rotation operators.

### C. Triplet Partition Ambiguity in Unconstrained Resolution
When decomposing an accidental triplet cluster (size 3) into a 1D singlet $s$ and a 2D doublet pair $\{d_1, d_2\}$, there are 3 possible candidate partitions:
$$\{(1, \{2, 3\}), (2, \{1, 3\}), (3, \{1, 2\})\ catch\}$$
In unconstrained resolution (`resolve_multiplet_symmetries` without target guidance), the score of each partition was defined as:
$$\text{score} = \max_{i \in \text{1D}} [p_s(i)] + \max_{j \in \text{2D}} [p_{\{d_1, d_2\}}(j)]$$
Because of numerical noise and substrate asymmetry, a single mixed state's raw projection onto an unphysical 1D irrep (e.g., $B_1$, score $\approx 0.38$) could marginally exceed its projection onto the target 1D irrep ($A_2$, score $\approx 0.35$). Consequently, the unconstrained selector favored an incorrect partition, rejecting the physical $A_2 \oplus E_1$ classification and triggering the $+100.0$ objective penalty.

### D. Separation Between Objective Targets and Irrep Solver
The optimization objective (`ModalOverlapDegeneracyObjective`) explicitly configured `target_irreps=['A_2', 'E_1', 'E_1']`. However, this target specification was never propagated down into `compute_band_symmetries` and `resolve_multiplet_symmetries`. The irrep solver was operating entirely unguided, while the objective downstream enforced a strict exact match.

---

## 3. Permanent Resolution

### 1. Target-Guided Multiplet Disambiguation
Updated `resolve_multiplet_symmetries` and `compute_band_symmetries` in `packages/phc_mpb/src/phc_mpb/symmetry.py` to accept an optional `target_irreps: Sequence[str] | None = None`:

- **Identification of Target Multiplicities**: When `target_irreps` contains a 1D target (e.g. $A_2$) and a 2D target (e.g. $E_1$), triplet resolution specifically evaluates candidates against these targets.
- **Threshold-Gated Subspace Scoring**:
  - Candidate singlets must meet a minimum projection threshold for the target 1D irrep: $p_s(\text{target\_1d}) \ge 0.15$.
  - Candidate doublet pairs must meet a minimum trace projection threshold for the target 2D irrep: $p_{\text{pair}}(\text{target\_2d}) \ge 0.25$.
  - Partitions satisfying these physical thresholds are ranked by their target-specific score:
    $$\text{score}_{\text{target}} = p_s(\text{target\_1d}) + p_{\text{pair}}(\text{target\_2d})$$
- **Graceful Fallback**: If no partition satisfies the target thresholds (indicating a genuinely different physical band configuration), the algorithm smoothly falls back to unconstrained maximum-score selection.
- **Unconstrained Default**: When `target_irreps is None` (e.g. general dispersion band diagrams), the solver preserves original unconstrained partitioning, ensuring 100% backwards compatibility.

### 2. Objective Integration
Updated `packages/phc_optimization/src/phc_optimization/objectives/modal_overlap_degeneracy.py`:
- `ModalOverlapDegeneracyObjective` passes `target_irreps=self.target_irreps` into `find_bands_from_irreps` and `compute_band_symmetries`.
- Cluster matching in `_find_bands_by_clustering` incorporates `target_irreps` so that candidate multiplets are resolved consistently.

### 3. Post-Processing & Analysis Recovery
Updated `load_run_records` in `analysis/optimize_c6v_2b_6d.py` to re-resolve raw symmetry records using `target_irreps`. Historical datasets with raw symmetry data are recovered without re-running expensive simulations.

---

## 4. Verification & Validation

1. **Historical Run Re-evaluation**:
   - Re-evaluated run `outputs/mpb/optimization/c6v_2b_6d_3d_slab_sio2/2026-09-30_16-54-39/`.
   - All 24 previously misclassified iterations (100%) were successfully recovered as valid $A_2 \oplus E_1$ triplets.
   - The artificial low-FOM craters in `bo_surrogate_map.png` were completely eliminated.
2. **Automated Unit Tests**:
   - Added `test_resolve_multiplet_symmetries_target_guided` in `packages/phc_mpb/tests/test_symmetry.py`.
   - Verified all 8 unit tests in `test_symmetry.py` pass without regression.
