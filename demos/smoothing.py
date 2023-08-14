from pathlib import Path
# Base libraries
import time
import igl
import numpy as np
from scipy import sparse

# Viewer
import polyscope as ps
import polyscope.imgui as psim

# Multigrid Solver
from gravomg import MultigridSolver
from gravomg.util import neighbors_from_stiffness, normalize_area

# Experiment util
from util import read_mesh

# Read mesh
V, F = read_mesh(Path('path_to_mesh.obj').resolve())
N = igl.per_vertex_normals(V, F)

print(f'Mesh loaded, {V.shape[0]} vertices')
 
# Normalize area and center around mean
V = normalize_area(V, F)

# Compute operators
M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
S = -igl.cotmatrix(V, F)

Minv = sparse.diags(1 / M.diagonal())
neigh = neighbors_from_stiffness(S)

# Create reusable solver
solver = MultigridSolver(V, neigh, M)

# Start UI
ui_tau = 0.001

def smoothing(tau):
    lhs = M + tau * S
    lhs_csr = lhs.tocsr()
    
    rhs = M @ V

    t = time.perf_counter()
    mg_V = solver.solve(lhs_csr, rhs)
    print(f'Our time: {time.perf_counter() - t}')
    print(f'Our residual: {solver.residual(lhs, rhs, mg_V)}')

    mesh.update_vertex_positions(mg_V)

# GUI
def smoothing_panel():
    global ui_tau

    psim.TextUnformatted("Smoothing")
    changed, ui_tau = psim.InputFloat("tau", ui_tau, format='%.6f')

    if(psim.Button("Smooth")):
        smoothing(ui_tau)

ps.init()
ps.set_ground_plane_mode('none')
ps.set_user_callback(smoothing_panel)

mesh = ps.register_surface_mesh('Input Mesh', V, F, enabled=True)
mesh.add_scalar_quantity('Mass', M.diagonal())

ps.show()