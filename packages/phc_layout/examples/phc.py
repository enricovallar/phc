# %%
# %load_ext autoreload
# %autoreload 2
# %%

import gdsfactory as gf
from phc_layout import utils
from phc_layout.components import phc
from phc_layout.tech import LAYER_PHC

# %%

n_rows = 15
n_cols = 20
circle = gf.components.circle(radius=0.2)
circle_m = gf.components.circle(radius=0.1)
square = gf.components.rectangle(size=(0.1, 0.1))


@gf.cell
def combined_basis(A, B):
    top = gf.Component()
    top.add_ref(A)
    top.add_ref(B)
    return top


cb = combined_basis(circle, square)
rect = phc.lattice_patch_c6v_rect(component=cb)
rect.show()
# %%
h = phc.lattice_patch_c6v_hex(component=cb, rings=10)
c = phc.get_component_hull(h, layer=LAYER_PHC.ETCH, margin=1)
mirror = phc.lattice_patch_c6v_rect(circle_m, 99, 99, 1)
mirror.show()
mirror = gf.boolean(mirror, c, operation="-", layer=LAYER_PHC.ETCH)
mirror.show()


@gf.cell
def combined_view(A, B):
    top = gf.Component()
    top.add_ref(A)
    top.add_ref(B)
    return top


top = combined_view(h, mirror)
top.plot()
top.show()


# %%
# Make Hexagonal Photonic Crystal Patch
lattice_constant = 1
# First we define the basis that is repeated to make the lattice

basis = gf.Component()
central_circle = gf.components.circle(radius=0.2 * lattice_constant)
edge_circle = gf.components.circle(radius=0.1 * lattice_constant)


import numpy as np

# Basis displacement scale
d = lattice_constant / 2.0

# 6 symmetric radial directions (60-degree increments)
angles_deg = [0, 60, 120, 180, 240, 300]
edge_vectors = [
    (d * np.cos(np.deg2rad(phi)), d * np.sin(np.deg2rad(phi))) for phi in angles_deg
]

# Add central circle reference
ref_center = basis << central_circle

# Add references to the SAME edge_circle without copying
for vec in edge_vectors:
    ref = basis << edge_circle
    ref.move(vec)

basis.plot()

# the patch size depends on the number of rings and pitch
patch_c6v_raw = phc.lattice_patch_c6v_hex(
    component=basis, rings=10, pitch=lattice_constant
)
patch_c6v = utils.deduplicate_instances(patch_c6v_raw)
patch_c6v.plot()
# %%
# We add the mirror photonic crystal
lattice_constant_m = 0.4
circle_m = gf.components.circle(radius=0.1 * lattice_constant_m)
# The size of the patch depends on the number of columns and rows
n_cols, n_rows = 101, 101
mirror_patch = phc.lattice_patch_c6v_rect(
    component=circle_m, columns=n_cols, rows=n_rows
)
mirror_patch.plot()
# We need to make a hole in the mirror to insert the photonic crystal patch as a cavity
# and open selective waveguides at the ports of the hexagonal patch (e.g. ports 1, 2, 3 out of 6)
margin = 1 * lattice_constant
cavity_hull = utils.get_component_hull(patch_c6v, layer=LAYER_PHC.ETCH, margin=margin)

open_ports = [1, 2, 3]
waveguide_width = 1.5 * lattice_constant
mirror = utils.carve_mirror_waveguides(
    mirror_patch=mirror_patch,
    cavity_hull=cavity_hull,
    open_ports=open_ports,
    waveguide_width=waveguide_width,
)
mirror.plot()
# %%
device = gf.Component()
device << mirror
device << patch_c6v
device.plot()
device.show()
