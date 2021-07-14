from firedrake_adjoint import *
from thetis import *

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime

import adjoint_functions as adj_fns

data = pd.read_csv('experimental_data.csv', header = None)

sed_flag = False
diff_flag = True
ks_flag = True
d50_flag = True
rho_flag = True
viscosity_flag = False

f = open("intermediate_dual_multi_diff_ks_d50_rho.txt", "w+")

def eval_callback(functional_value, value):
    f.write(str([x.dat.data[:] for x in value]))
    f.write('   ')
    f.write(str(functional_value))
    f.write('\n')
    print([x.dat.data[:] for x in value])
    print(functional_value)

lx = 16
ly = 1.1
nx = lx*4
ny = 5
mesh2d = RectangleMesh(nx, ny, lx, ly)

x, y = SpatialCoordinate(mesh2d)

V = FunctionSpace(mesh2d, "CG", 1)

initialdepth = Constant(0.397)
depth_riv = Constant(initialdepth - 0.397)
depth_trench = Constant(depth_riv - 0.15)
depth_diff = depth_trench - depth_riv

trench = conditional(le(x, 5), depth_riv, conditional(le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,
                         conditional(le(x, 9.5), depth_trench, conditional(le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv, depth_riv))))

def forward(bathymetry_2d, viscosity_norm, ks_norm, average_size_norm, rhos_norm, diffusivity_norm, sed_rate = None):

    # define function spaces
    V = FunctionSpace(mesh2d, "CG", 1)
    DG_2d = FunctionSpace(mesh2d, "DG", 1)
    vector_dg = VectorFunctionSpace(mesh2d, "DG", 1)
    R_1d = FunctionSpace(mesh2d, 'R', 0)

    rhos = Function(R_1d).assign(rhos_norm*1000)
    average_size = Function(R_1d).assign(average_size_norm/(1e4))
    ks = Function(R_1d).assign(ks_norm/100)
    diffusivity = Function(R_1d).assign(diffusivity_norm/10)

    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs'+ st

    print_output('Exporting to '+outputdir)

    morfac = 100
    dt = 0.25
    end_time = 5*3600

    viscosity_hydro = Function(R_1d).assign(viscosity_norm*1e-6)

    # initialise velocity and elevation
    chk = DumbCheckpoint("hydrodynamics_trench_64/elevation", mode=FILE_READ)
    elev = Function(DG_2d, name="elevation")
    chk.load(elev)
    chk.close()

    chk = DumbCheckpoint('hydrodynamics_trench_64/velocity', mode=FILE_READ)
    uv = Function(vector_dg, name="velocity")
    chk.load(uv)
    chk.close()

    # set up solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options

    options.sediment_model_options.solve_suspended_sediment = True
    options.sediment_model_options.use_bedload = True
    options.sediment_model_options.solve_exner = True

    options.sediment_model_options.average_sediment_size = average_size
    options.sediment_model_options.bed_reference_height = Function(R_1d).assign(ks)
    options.sediment_model_options.morphological_acceleration_factor = Constant(morfac)
    options.sediment_model_options.sediment_density = rhos

    options.simulation_end_time = end_time/morfac
    options.simulation_export_time = options.simulation_end_time/45

    options.output_directory = outputdir
    options.check_volume_conservation_2d = True

    options.fields_to_export = ['sediment_2d', 'uv_2d', 'elev_2d', 'bathymetry_2d']  # note exporting bathymetry must be done through export func
    options.sediment_model_options.check_sediment_conservation = True

    # using nikuradse friction
    options.nikuradse_bed_roughness = Function(R_1d).assign(3*average_size)

    # set horizontal diffusivity parameter
    options.horizontal_diffusivity = diffusivity
    options.horizontal_viscosity = viscosity_hydro

    # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
    options.timestepper_type = 'CrankNicolson'
    options.timestepper_options.implicitness_theta = 1.0
    options.norm_smoother = Constant(0.1)

    if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt

    # set boundary conditions

    left_bnd_id = 1
    right_bnd_id = 2

    swe_bnd = {}

    swe_bnd[left_bnd_id] = {'flux': Constant(-0.22)}
    swe_bnd[right_bnd_id] = {'elev': Constant(0.397)}

    solver_obj.bnd_functions['shallow_water'] = swe_bnd

    if sed_rate is not None:
        solver_obj.bnd_functions['sediment'] = {
            left_bnd_id: {'flux': Constant(-0.22), 'value': sed_rate},
            right_bnd_id: {'elev': Constant(0.397)}}

        # set initial conditions
        solver_obj.assign_initial_conditions(uv=uv, elev=elev, sediment=sed_rate)
    else:
        solver_obj.bnd_functions['sediment'] = {
            left_bnd_id: {'flux': Constant(-0.22), 'equilibrium': None},
            right_bnd_id: {'elev': Constant(0.397)}}

        # set initial conditions
        solver_obj.assign_initial_conditions(uv=uv, elev=elev)

    # run model
    solver_obj.iterate()

    return solver_obj.fields.bathymetry_2d

test_derivative = False
taylor_test_flag = False
minimize_flag = True

ks = Constant(0.025*100)
if ks_flag:
    ks2 = Constant(0.01*100)
else:
    ks2 = Constant(0.025*100)
ks_diff = Constant(0.025*100 + 1e-3)

average_size = Constant(160e-2)
if d50_flag:
    average_size2 = Constant(200e-2)
else:
    average_size2 = Constant(160e-2)
average_size_diff = Constant(161e-2)

rhos = Constant(2650/1000)
if rho_flag:
    rhos2 = Constant(2000/1000)
else:
    rhos2 = Constant(2650/1000)
rhos_diff = Constant(2660/1000)

if diff_flag:
    diffusivity2 = Constant(0.1)
else:
    diffusivity2 = Constant(1.5)

diffusivity = Constant(1.5)
diffusivity_diff = Constant(1.5 + 10**(-4))

if viscosity_flag:
    viscosity2 = Constant(1.1)
else:
    viscosity2 = Constant(1.0)

viscosity = Constant(1.0)
viscosity_diff = Constant(1.0001)

if sed_flag:
    sed_rate = Constant(0.3/(0.397*0.51*2650))
    sed_rate2 = Constant(0.4/(0.397*0.51*2650))
    sed_rate_diff = Constant(0.3/(0.397*0.51*2650) + 1e-6)
else:
    sed_rate = None
    sed_rate2 = None

bath1 = Function(V).interpolate(-trench)

old_bath = forward(bath1, viscosity2, ks2, average_size2, rhos2, diffusivity2, sed_rate2)

tape = get_working_tape()
tape.clear_tape()

bath2 = Function(V).interpolate(-trench)

new_bath = forward(bath2, viscosity, ks, average_size, rhos, diffusivity, sed_rate)

form = 0

for i in range(len(data)):
    bump_func = Function(V).project((1/0.01)*exp(-0.5*((x-data.iloc[i][0])/0.01)**2))
    normaliser = inner(bump_func, bump_func)*dx
    form += 0.5*inner(old_bath*bump_func - new_bath*bump_func, old_bath*bump_func - new_bath*bump_func)/(assemble(normaliser))

J = assemble(1e3*form*dx)
print(J)


if test_derivative:
    rf = ReducedFunctional(J, [Control(p) for p in [rhos, average_size, ks, diffusivity]])

    J_h = rf([rhos, average_size, ks, diffusivity])
    print(J_h)

    der = rf.derivative()

    der_sum = (der[0].dat.data[0]*10/1000) + (der[1].dat.data[0]*1e-2) + (der[2].dat.data[0]*1e-3) + (der[3].dat.data[0]*10**(-4))

    print(der_sum)

    J_0 = rf([rhos_diff, average_size_diff, ks_diff, diffusivity_diff])

    print(J_0 - J_h)

    f.close()
    stop

if taylor_test_flag == True:
    rf = ReducedFunctional(J, [Control(p) for p in [rhos, average_size, ks, diffusivity]])
    h = [Constant(10/1000), Constant(5), Constant(100*0.01), Constant(5e-3*10)]
    print([i.dat.data[:] for i in h])
    conv_rate = taylor_test(rf, [rhos, average_size, ks, diffusivity], h)

    if conv_rate > 1.9:
        print('*** test passed ***')
    else:
        print('*** ERROR: test failed ***')

    f.close()
    stop

if minimize_flag:
    rf = ReducedFunctional(J, [Control(p) for p in [rhos, average_size, ks, diffusivity]], eval_cb_post = eval_callback)
    min_value = minimize(rf, options={'gtol':  1e-200, 'ftol': 1e-6, 'maxfun': 1000}, bounds = [[AdjFloat(1.5), AdjFloat(160e-2), AdjFloat(0.5), AdjFloat(0.0)], [AdjFloat(3), AdjFloat(250e-2), AdjFloat(5), AdjFloat(5)]])

    f.close()

