"""
3D-Var Data Assimilation in Latent Space using Graph Neural Network Autoencoders.

This module implements preconditioned and incremental 3D-Var algorithms for
variational data assimilation in the latent space of a graph neural network
autoencoder (GNN-AE).
"""

import os
import sys

import numpy as np
import xarray as xr
from LatentVar.helpers import (
    build_r_matrix,
    build_runtime,
    compute_rh,
    load_background_state,
    parse_args,
    run_single_assimilation,
    write_metadata,
)


def build_observation_coords(runtime, state):
    obs_coords = []
    grid_lats_np = runtime.grid_lats.cpu().numpy()
    grid_lons_np = runtime.grid_lons.cpu().numpy()

    if False:
        lvlstr = '800'
        rh800 = compute_rh(
            lvlstr,
            state.physical_bg_dest.detach().cpu().numpy(),
            runtime.obs_qty_indices,
        ) * 100
        mask = rh800 > 90
        i_selected = np.random.permutation(np.where(mask)[0])
        print(f"Number of obs. locations with RH800 > 90%: {len(i_selected)}")

    if False:
        var = 'siconc2'
        i_var = runtime.obs_qty_indices[var]
        siconc = state.physical_bg_dest[:, i_var].detach().cpu().numpy()
        mask = (siconc > 0) & (siconc < 1)
        i_selected = np.random.permutation(np.where(mask)[0])
        print(f"Number of obs. locations with 0 < siconc < 1: {len(i_selected)}")

    # if True:
    #     lsm = state.dataset_of_interest.static_tensors[3].detach().cpu().numpy()
    #     mask = (lsm == 0) & (abs(grid_lats_np) > 30)

    # i_selected = np.where(mask)[0]
    # print(f"Number of obs. locations with conditions: {len(i_selected)}")

    # for idx in i_selected[50::25]:  # subsample for speed
    #     obs_coords.append((grid_lats_np[idx], grid_lons_np[idx]))

    for lon in range(0, 360, 10):
        for lat in range(30, 90, 10):
            obs_coords.append((lat, lon))

    return obs_coords


def save_diagnostics(diagnostics, runtime):
    obs_lats_list = [float(diag.obs_lat) for diag in diagnostics]
    obs_lons_list = [float(diag.obs_lon) for diag in diagnostics]
    diagnostics = xr.concat(diagnostics, dim='experiment', coords='minimal', compat='override')
    diagnostics = diagnostics.assign_coords(
        obs_lat=('experiment', obs_lats_list),
        obs_lon=('experiment', obs_lons_list),
    )

    folder = f'/ceph/hpc/home/kuglerl/Latent3DVar/output/{runtime.ae_props.EXP_ID}/'
    os.makedirs(folder, exist_ok=True)
    output_file = folder + f'ekman-exp_regular_{runtime.obs_datetime}_obs={str(runtime.obs_qty[0])}_dep={str(runtime.obs_dep[0])}.nc'

    if os.path.isfile(output_file):
        print(f"File {output_file} already exists. It will be extended.")
        diag_old = xr.open_dataset(output_file)
        diag_new = xr.concat([diag_old, diagnostics], dim='experiment', coords='minimal', compat='override')
        os.remove(output_file)
        diag_new.to_netcdf(output_file)
    else:
        diagnostics.to_netcdf(output_file)
    print(f"Diagnostics saved to {output_file}")


def main(argv=None):
    argv = '--singobs_lat=45.0 --singobs_lon=0.0 --obs_datetime=2023-03-01-00 --B_type=climatological --AE_version=v63 --obs_qty=u600 --obs_dep=5 --obs_std=1 --init_lr=0.1'.split()
    args = parse_args(argv)
    print(args)

    runtime = build_runtime(args)
    state = load_background_state(args, runtime)
    write_metadata(runtime)

    obs_coords = build_observation_coords(runtime, state)
    r_matrix_inv = 1 / build_r_matrix(runtime.obs_std, n_obs_loc=1)
    diagnostics = [
        run_single_assimilation(obs_coord, args, runtime, state, r_matrix_inv)
        for obs_coord in obs_coords
    ]
    save_diagnostics(diagnostics, runtime)


if __name__ == '__main__':
    main()
