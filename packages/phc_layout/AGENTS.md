# `phc_layout` Guidelines & Rules

## Scope
`phc_layout` is responsible for programmatic layout generation of photonic crystal unit cells, supercells, waveguides, cavities, and coupling structures.

## Core Rules

1. **Framework & Engine**:
   - Use **GDSFactory** (`gdsfactory` as `gf`) for all component construction and cell factories.
   - Decorate cell functions with `@gf.cell` and return `gf.Component`.
   - Use `phc_layout.tech.LAYER_PHC` (or `gf.typings.LayerSpec`) for standard layer definitions.

2. **Units & Precision**:
   - All spatial dimensions (pitch $a$, radius $r$, width $w$, length $L$, etc.) must be specified in **micrometers** ($\mu\text{m}$).
   - Respect database units (`dbu`), typically $0.001\,\mu\text{m}$ ($1\,\text{nm}$).

3. **Layer Conventions & Technology**:
   - Etch / hole layer: `LAYER_PHC.ETCH` (`(1, 0)`).
   - Slab / substrate boundaries: `LAYER_PHC.SLAB` (`(2, 0)`), `LAYER_PHC.BOX` (`(3, 0)`).
   - Ports: Explicit optical ports on `c.add_port(...)` for waveguide interfaces.
   - LayerStack: Manage 3D extrusion definitions using `gdsfactory.technology.LayerStack`.

4. **Clean Hierarchy**:
   - Keep unit cells and supercells cleanly isolated.
   - Support both hexagonal (triangular) and square lattice geometries.
   - Ensure GDSII/OASIS export (`component.write_gds(...)`) works cleanly and does not leave lingering open file handles.

5. **No Solver / Material Cross-Contamination**:
   - Do not call MPB, Meep, or Lumerical APIs inside `phc_layout`.
   - Layout components must remain pure geometric definitions.
