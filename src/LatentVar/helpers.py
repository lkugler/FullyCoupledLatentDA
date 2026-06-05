import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units

from LatentVar.interpolator import interpolate_irregular_grid, normalize_longitudes
from LatentVar.minimizer import (
    incremental_latent3DVar,
    latent3DVar_algorithm_preconditioned,
)

# AUTOENCODER_DA_PARENT = Path("/ceph/hpc/home/kuglerl")
# if str(AUTOENCODER_DA_PARENT) not in sys.path:
#     sys.path.insert(0, str(AUTOENCODER_DA_PARENT))

from autoencoder_da import datasets
from autoencoder_da.model_variants import build_autoencoder
from autoencoder_da.utilities import clean_state_dict, standardise_destandardise


DETERMINISTIC = False

def _env_path(var_name: str) -> Path:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {var_name}. "
            f"Set it before loading the selected data contract."
        )
    return Path(value).expanduser().resolve()

def get_env_path(var_name: str) -> Path:
    return _env_path(var_name)

def torchload(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.load(*args, **kwargs)


@dataclass
class RuntimeContext:
    device: torch.device
    ae_props: dict
    ae_model: torch.nn.Module
    ae_scalers_mean: torch.Tensor
    ae_scalers_std: torch.Tensor
    grid_lats: torch.Tensor
    grid_lons: torch.Tensor
    obs_datetime: datetime
    obs_qty_indices: dict
    obs_qty: list
    obs_qty_idx: list
    obs_std: list
    obs_dep: list
    figs_dir: str


@dataclass
class BackgroundState:
    dataset_of_interest: datasets.ReallyLazyGraphDataset_v2
    physical_bg: torch.Tensor
    physical_bg_dest: torch.Tensor
    decoded_bg_dest: torch.Tensor
    latent_bg: torch.Tensor
    b_matrix_sqrt: torch.Tensor
    time: datetime


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--AE_version', help="Version of AE, e.g. 'v5'", type=str, required=True)
    parser.add_argument('--VarDA_type', help="'prec_3D-Var' or 'inc_3D-Var'", type=str, default='prec_3D-Var')
    parser.add_argument('--FWD_model', help="Which forward model do I use? Options: 'persistence', 'NNfwd'", type=str, required=False, default='persistence')
    parser.add_argument("--obs_datetime", help="Date and hour of *observation* in format yyyy-mm-dd-hh", type=str, required=False, default='2023-04-15-00')
    parser.add_argument("--forecast_len", help="Number of 1-hourly forecast steps", type=int, required=False, default=24)
    parser.add_argument('--init_lr', help='Initial learning rate for SGD optimizer when performing 3D-Var cost function minimization in latent space', type=float, default=0.5)
    parser.add_argument('--custom_addon', help="Custom addon to output filenames", type=str, default='', required=False)
    parser.add_argument('--pseudo_obs', help="Generate and assimilate pseudo observations", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--obs_dep', help="Observation departure for single observation experiment - if '0.0', regular experiment will be performed with obs. on manually defined grid", type=str, required=False, default='0.0')
    parser.add_argument('--singobs_lat', help="Latitude in case of single observation experiment", type=float, required=False, default=np.nan)
    parser.add_argument('--singobs_lon', help="Latitude in case of single observation experiment", type=float, required=False, default=np.nan)
    parser.add_argument('--plot_singles', help="Plot some figures one by one (only if --plot)", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--obs_qty', help="Observed variable (passed e.g. as Z200,u200)", type=str, required=True)
    parser.add_argument('--obs_std', help="Standard deviation of pseudo observations, arbitrary unit (passed in the same order as obs_qty, e.g. as 1.0,2.5). If set to 0.0 in a singobs experiment, the system will set obs_std to the same value as background std in physical space", type=str, required=True)
    parser.add_argument('--savefig_dir', help='Directory for saving the figure (if specified, it will be saved to experiments/figures/{in_out_ch}ch/args.savefig_dir; otherwise it will be just saved to experiments/figures/{in_out_ch}ch/)', type=str, default='', required=False)
    parser.add_argument('--plot_projection', help="Plotting projection (PlateCarree or NearsidePerspective)", type=str, default='PlateCarree', required=False)
    parser.add_argument('--B_type', help="B-matrix version: climatological or ensemble", type=str, required=True)
    parser.add_argument('--em_idx', help="Ensemble member index (in case of ensemble B)", type=int, default=0)
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    assert args.VarDA_type in ('prec_3D-Var', 'inc_3D-Var')
    return args


def select_device():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        device = torch.device("cpu")
        print("Using device: cpu")
        return device

    free_mem = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    best_gpu = free_mem.index(max(free_mem))
    device = torch.device(f"cuda:{best_gpu}")
    print(f"Free memory on GPUs: {[fm / 1024**3 for fm in free_mem]} GB")
    print(f"Using device: {device}")
    return device


def build_runtime(args):
    runtime_ae_props = build_autoencoder(args.AE_version)

    b_addon = ''
    if args.B_type == 'ensemble':
        b_addon = 'ens_B/'
    figs_dir = f"{runtime_ae_props.vPATH}/figs/DA/algorithm-single/experiments/figs/{runtime_ae_props.EXP_ID}/{b_addon}"
    os.makedirs(figs_dir, exist_ok=True)

    if DETERMINISTIC:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    device = select_device()

    graphs = []
    proper_level_connections = []
    if runtime_ae_props.use_pooling:
        for lev in range(runtime_ae_props.N_SUBGRAPHS + 1):
            graphs.append(torchload(runtime_ae_props.GRAPH / f"{runtime_ae_props.EXP_ID}_edge_index_ae_{lev}.pt"))
        _, _ = torchload(runtime_ae_props.GRAPH / f"{runtime_ae_props.EXP_ID}_pooling_matrices.pt")
        proper_level_connections = torchload(
            runtime_ae_props.GRAPH / f"{runtime_ae_props.EXP_ID}_{runtime_ae_props.level_connections_type}.pt"
        )
        graphs = [graph.to(device) for graph in graphs]
    else:
        _ = torchload(runtime_ae_props.GRAPH / f"{runtime_ae_props.EXP_ID}_edge_index_ae.pt").to(device)

    ae_model = runtime_ae_props.modelslib.ProgressiveGraphAutoencoder(
        in_dim=runtime_ae_props.IN_DIM,
        encoder_hidden_dims=runtime_ae_props.HIDDEN_DIMS,
        decoder_hidden_dims=runtime_ae_props.HIDDEN_DIMS[::-1],
        latent_dim=runtime_ae_props.LATENT_DIM,
        out_dim=runtime_ae_props.OUT_DIM,
        graphs=graphs,
        level_connections=proper_level_connections,
        heads=runtime_ae_props.HEADS,
        gat_dropout=runtime_ae_props.GAT_DROPOUT,
        feature_dropout=runtime_ae_props.FEATURE_DROPOUT,
        latent_dropout=runtime_ae_props.LATENT_DROPOUT,
        latent_noise_std=runtime_ae_props.LATENT_NOISE_STD,
        use_residuals=runtime_ae_props.use_residuals,
        use_residualsIO=runtime_ae_props.use_residualsIO,
    ).to(device)

    ae_best_model = torchload(runtime_ae_props.best_model_pth, map_location=device)
    ae_model.load_state_dict(clean_state_dict(ae_best_model))
    ae_model.eval()

    obs_qty_indices = {
        runtime_ae_props.reconstructed_variables[idx]: idx
        for idx in range(len(runtime_ae_props.reconstructed_variables))
    }
    obs_qty = [quantity for quantity in args.obs_qty.split(',')]
    obs_qty_idx = [obs_qty_indices[quantity] for quantity in obs_qty]
    obs_std = [float(std) for std in args.obs_std.split(',')]
    obs_dep = [float(dep) for dep in args.obs_dep.split(',')]

    ae_scalers_mean_np = np.load(runtime_ae_props.contract.scalers_mean_path)
    ae_scalers_std_np = np.load(runtime_ae_props.contract.scalers_std_path)
    ae_scalers_mean = torch.tensor(
        [torch.from_numpy(ae_scalers_mean_np[rv]) for rv in runtime_ae_props.reconstructed_variables]
    )
    ae_scalers_std = torch.tensor(
        [torch.from_numpy(ae_scalers_std_np[rv]) for rv in runtime_ae_props.reconstructed_variables]
    )

    grid_lats = torch.from_numpy(np.load(runtime_ae_props.contract.grid_lat_path)).to(torch.float32).to(device)
    grid_lons = torch.from_numpy(np.load(runtime_ae_props.contract.grid_lon_path)).to(torch.float32).to(device)
    _ = normalize_longitudes(grid_lons.cpu()).numpy()

    for quantity, quantity_idx in zip(obs_qty, obs_qty_idx):
        print(
            f"{quantity}: Climatological global mean={ae_scalers_mean[quantity_idx]}, "
            f"global std={ae_scalers_std[quantity_idx]}"
        )

    return RuntimeContext(
        device=device,
        ae_props=runtime_ae_props,
        ae_model=ae_model,
        ae_scalers_mean=ae_scalers_mean,
        ae_scalers_std=ae_scalers_std,
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        obs_datetime=datetime.strptime(args.obs_datetime, '%Y-%m-%d-%H'),
        obs_qty_indices=obs_qty_indices,
        obs_qty=obs_qty,
        obs_qty_idx=obs_qty_idx,
        obs_std=obs_std,
        obs_dep=obs_dep,
        figs_dir=figs_dir,
    )


def compute_rh(lvlstr, physical_bg, obs_qty_indices):
    p = int(lvlstr) * 100 * units.pascal
    tmp = physical_bg[:, obs_qty_indices['t' + lvlstr]] * units('kelvin')
    q = physical_bg[:, obs_qty_indices['q' + lvlstr]] * units('kg/kg')
    return relative_humidity_from_specific_humidity(p, tmp, q)



def load_climatological_physical_bg(fwd_start_datetime, fwd_end_datetime, runtime, fwd_model='persistence'):

    dataset_of_interest = datasets.ReallyLazyGraphDataset_v2(
        runtime.ae_props.contract.data_path,
        runtime.ae_props.time_varying_variables,
        runtime.ae_props.static_variables,
        runtime.ae_props.reconstructed_variables,
    )
    
    dataset_of_interest.set_dates(fwd_start_datetime, fwd_end_datetime)

    if fwd_model != 'persistence':
        raise AttributeError(f'{fwd_model} not yet implemented in bg computation')

    init_truth = dataset_of_interest[0].to(runtime.device)
    physical_bg = init_truth.x
    return dataset_of_interest, physical_bg


def load_background_state(args, runtime):
    fwd_end_datetime = datetime.strptime(args.obs_datetime, "%Y-%m-%d-%H")
    fwd_start_datetime = fwd_end_datetime - timedelta(hours=args.forecast_len)

    accepted_fwd_models = ('persistence',)
    if args.FWD_model not in accepted_fwd_models:
        print('Unknown forward model:', args.FWD_model)
        print('Accepted forward models:', accepted_fwd_models)
        raise AttributeError

    if args.B_type == 'climatological':
        dataset_of_interest, physical_bg = load_climatological_physical_bg(
            fwd_start_datetime,
            fwd_end_datetime,
            runtime,
            fwd_model=args.FWD_model,
        )
    
    elif args.B_type == 'ensemble':
        dataset_of_interest = datasets.ReallyLazyGraphDataset_v2(
            range(args.em_idx, args.em_idx + 1),
            runtime.ae_props.contract.ensemble_data_path / args.obs_datetime.replace('-', ''),
            runtime.ae_props.time_varying_variables,
            runtime.ae_props.static_variables,
            runtime.ae_props.reconstructed_variables,
        )
        physical_bg = dataset_of_interest[0].to(runtime.device).x
    else:
        raise NotImplementedError(args.B_type + ' not supported')

    if args.B_type == 'climatological':
        b_matrix_savename = (
            f'{runtime.ae_props.contract.climatological_b_matrix_path}/'
            f'{runtime.ae_props.EXP_ID}_climatological_prediction_len_{args.forecast_len}h_diag.pt'
        )
    elif args.B_type == 'ensemble':
        b_matrix_savename = f'{runtime.ae_props.contract.ensemble_b_matrix_path}/{runtime.ae_props.EXP_ID}_{args.obs_datetime}_diag.pt'
    else:
        raise AttributeError('Unsupported B_type:', args.B_type)

    assert os.path.isfile(b_matrix_savename)
    print('Loading B matrix from', b_matrix_savename)
    b_matrix = torchload(b_matrix_savename, map_location='cpu')

    print('Running Encoder-Decoder')
    decoded_bg, latent_bg = runtime.ae_model(physical_bg, return_latent=True)
    decoded_bg_dest = standardise_destandardise(
        decoded_bg,
        runtime.ae_props,
        runtime.ae_scalers_mean,
        runtime.ae_scalers_std,
        'destandardise',
        runtime.device,
    )
    print('Destandardize')
    physical_bg_dest = standardise_destandardise(
        physical_bg.cpu()[:, :runtime.ae_props.OUT_DIM],
        runtime.ae_props,
        runtime.ae_scalers_mean,
        runtime.ae_scalers_std,
        'destandardise',
    )

    return BackgroundState(
        dataset_of_interest=dataset_of_interest,
        time=fwd_start_datetime,
        physical_bg=physical_bg,
        physical_bg_dest=physical_bg_dest,
        decoded_bg_dest=decoded_bg_dest,
        latent_bg=latent_bg,
        b_matrix_sqrt=torch.sqrt(b_matrix),
    )


def write_metadata(runtime):
    with open('var_list.yaml', 'w') as file_handle:
        yaml.dump(runtime.ae_props.reconstructed_variables, file_handle)


def build_r_matrix(obs_std, n_obs_loc=1):
    n_obs_qty = len(obs_std)
    n_total_obs = n_obs_qty * n_obs_loc
    r_matrix = torch.zeros((n_total_obs, 1))
    for qty_idx, obs_std_value in enumerate(obs_std):
        for loc_idx in range(n_obs_loc):
            flat_idx = qty_idx * n_obs_loc + loc_idx
            r_matrix[flat_idx] = obs_std_value ** 2
    return r_matrix


def build_obs_vector(obs_coord, args, runtime, state):
    singobs_lat, singobs_lon = obs_coord
    if not args.pseudo_obs:
        raise NotImplementedError('Real observations are not yet implemented')

    has_lat = not np.isnan(singobs_lat)
    has_lon = not np.isnan(singobs_lon)
    if has_lat and has_lon:
        print('Single obs')
        obs_lats = torch.tensor([singobs_lat]).to(runtime.device)
        obs_lons = torch.tensor([singobs_lon]).to(runtime.device)
    elif has_lat != has_lon:
        raise AttributeError('Specified either singobs_lat or singobs_lon, but not both')
    else:
        print('Multi obs')
        obs_lats = np.arange(-5, 5.1, 2)
        obs_lons = np.arange(120, 170.1, 2)
        obs_lons, obs_lats = np.meshgrid(obs_lons, obs_lats)
        obs_lons = torch.from_numpy(obs_lons.flatten()).to(torch.float32).to(runtime.device)
        obs_lats = torch.from_numpy(obs_lats.flatten()).to(torch.float32).to(runtime.device)

    obs_values = interpolate_irregular_grid(
        runtime.grid_lats,
        runtime.grid_lons,
        state.decoded_bg_dest[:, runtime.obs_qty_idx],
        obs_lats,
        obs_lons,
    )
    if args.obs_dep != '0.0':
        obs_values = obs_values + torch.tensor(
            [[dep for _ in range(obs_values.shape[1])] for dep in runtime.obs_dep],
            device=runtime.device,
        )
    obs_vec = obs_values.reshape((obs_values.shape[0] * obs_values.shape[1], 1))
    print('Observed values:', obs_vec)
    return obs_vec, obs_lats, obs_lons


def analyze_result(algorithm_output, obs_coord, runtime, state, obs_lats, obs_lons):
    singobs_lat, singobs_lon = obs_coord
    decoded_ana_dest = standardise_destandardise(
        runtime.ae_model.decode(algorithm_output['out_latent']),
        runtime.ae_props,
        runtime.ae_scalers_mean,
        runtime.ae_scalers_std,
        action='destandardise',
        device=runtime.device,
    )

    ana_inc = (decoded_ana_dest - state.decoded_bg_dest).detach().cpu().numpy()
    best_posterior = state.physical_bg_dest.detach().cpu().numpy() + ana_inc

    grid_lats = runtime.grid_lats.detach().cpu().numpy()
    grid_lons = runtime.grid_lons.detach().cpu().numpy()
    obs_lats = obs_lats.detach().cpu().numpy()
    obs_lons = obs_lons.detach().cpu().numpy()

    physical_bg = state.physical_bg_dest.detach().cpu().numpy()
    decoded_bg_numpy = state.decoded_bg_dest.detach().cpu().numpy()
    decoded_ana_dest = decoded_ana_dest.detach().cpu().numpy()

    prior = interpolate_irregular_grid(grid_lats, grid_lons, physical_bg, obs_lats, obs_lons)
    decoded_bg = interpolate_irregular_grid(grid_lats, grid_lons, decoded_bg_numpy, obs_lats, obs_lons)
    decoded_ana = interpolate_irregular_grid(grid_lats, grid_lons, decoded_ana_dest, obs_lats, obs_lons)
    posterior = interpolate_irregular_grid(grid_lats, grid_lons, best_posterior, obs_lats, obs_lons)

    arr = np.stack([prior[:, 0], decoded_bg[:, 0], decoded_ana[:, 0], posterior[:, 0]], axis=0)
    return xr.Dataset(
        {"field": (("stage", "variable"), arr)},
        coords={
            "stage": ["prior", "decoded_bg", "decoded_ana", "posterior"],
            "variable": runtime.ae_props.reconstructed_variables,
            "obs_lat": float(singobs_lat),
            "obs_lon": float(singobs_lon),
        },
    )


def run_single_assimilation(obs_coord, args, runtime, state, r_matrix_inv):
    print(f"Selected obs location lat={obs_coord[0]}, lon={obs_coord[1]}")
    obs_vec, obs_lats, obs_lons = build_obs_vector(obs_coord, args, runtime, state)
    print('obs_vec:', obs_vec)

    input_dict = dict(
        latent_bg=state.latent_bg,
        obs_vec=obs_vec,
        B_matrix_sqrt=state.b_matrix_sqrt,
        obs_lats=obs_lats,
        obs_lons=obs_lons,
        obs_qty_idx=runtime.obs_qty_idx,
        R_matrix_inv=r_matrix_inv,
        grid_lats=runtime.grid_lats,
        grid_lons=runtime.grid_lons,
        AE_model=runtime.ae_model,
        AE_props=runtime.ae_props,
        AE_scalers_mean=runtime.ae_scalers_mean,
        AE_scalers_std=runtime.ae_scalers_std,
    )

    if args.VarDA_type == 'prec_3D-Var':
        algorithm_output = latent3DVar_algorithm_preconditioned(
            input_dict,
            device=runtime.device,
            init_lr=args.init_lr,
        )
    elif args.VarDA_type == 'inc_3D-Var':
        algorithm_output = incremental_latent3DVar(
            input_dict,
            device=runtime.device,
            init_lr=args.init_lr,
        )
    else:
        raise AttributeError(f'{args.VarDA_type} not yet added to the system')

    return analyze_result(algorithm_output, obs_coord, runtime, state, obs_lats, obs_lons)