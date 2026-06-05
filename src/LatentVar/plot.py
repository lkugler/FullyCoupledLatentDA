import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from typing import List
from copy import copy

# dict of units for each variable, for colorbar labels
# 1: condition (to evaluate), 2: unit (str)
def unit_for_var(varname: str) -> str:
    if varname.startswith('siconc'):
        return '1'
    elif varname.startswith('t') or varname in ['stl1', 'sst2']:
        return 'K'
    elif varname.startswith('u') or varname.startswith('v'):
        return 'm/s'
    elif varname.startswith('z') or varname in ['sd']:
        return 'm'
    elif varname.startswith('o3'):
        return 'concentration'
    else:
        return 'unit'
    
def plotkws_for_var(varname: str) -> str:
    """Natural, suitable colormap for a given variable."""
    if varname.startswith('siconc'):
        kw = dict(cmap='Blues_r', vmin=0, vmax=1)
    elif varname == 'sd':
        kw = dict(cmap='Blues', vmin=0, vmax=None)
    elif varname.startswith('t'):
        kw = dict(cmap='nipy_spectral')
    elif varname.startswith('u') or varname.startswith('v'):
        kw = dict(cmap='viridis')
    elif varname.startswith('z'):
        kw = dict(cmap='terrain')
    elif varname.startswith('o3'):
        kw = dict(cmap='nipy_spectral')
    else:
        kw = dict(cmap='viridis')

    # ensure vmin, vmax, levels in kw
    # if 'vmin' not in kw:
    #     kw.update(vmin = data.min())
    # if 'vmax' not in kw:
    #     kw.update(vmax = data.max())
    # if 'levels' not in kw:
    #     locator = mticker.MaxNLocator(nbins=30)
    #     kw.update( #levels = locator.tick_values(kw['vmin'], kw['vmax']),
    #               vmin = kw['vmin'], 
    #               vmax = kw['vmax'])
    return kw


class Triangulation:
    
    def __init__(self, proj, lat, lon):
        """Create a triangulation for plotting.
        
        Initializes a Matplotlib triangulation from the provided
        coordinate arrays and stores it on ``self.tri``.
        
        Args:
            x: 1D array of x-coordinate values (e.g. projected longitudes).
            y: 1D array of y-coordinate values (e.g. projected latitudes).
        """
        xy = proj.transform_points(ccrs.PlateCarree(), lon, lat)
        self.x = xy[:, 0]
        self.y = xy[:, 1]
        self.visible = np.isfinite(self.x) & np.isfinite(self.y)
        # nan points error
        self.triangulation = mtri.Triangulation(self.x[self.visible], self.y[self.visible])
        
    def mask_triangles_with_NaN_values(self, data):
        """        
        With non-PlateCaree maps, points can fail to be mapped
        these need to masked out here
        """
        # set invisible points to nan
        self.data = data[self.visible]
        #self.data[~self.visible] = np.nan
        
        # Mask triangles that include any NaN-valued vertices
        is_nan = np.isnan(self.data)
        tri_mask = np.any(is_nan[self.triangulation.triangles], axis=1)
        self.triangulation.set_mask(tri_mask)
    

def plot_map(
    da_output: dict,
    plot_vars: List[str],
    proj: ccrs.Projection,
    obs_lons: np.ndarray = None,
    obs_lats: np.ndarray = None,
    args = None,  # argparse.Namespace
    output_dir: str = None,
    diagnostics: dict = None
) -> None:
    """Plot field data on a map, one subplot per variable.
    
    Can plot either a single field or background and increment fields side by side.
    
    Args:
        proj: Cartopy projection for the map
        plot_vars: List of variable names to plot (must be in da_output['var_list'])
        obs_lons: Observation longitude locations
        obs_lats: Observation latitude locations
        args: Argument namespace containing experiment parameters
        output_dir: Directory to save the output figure
        da_output: Dictionary containing data to plot, with keys:
            grid_lats: Latitude values for the grid
            grid_lons: Longitude values for the grid
            z_background: Background field, shape (n_points, n_variables). If provided with z_increment,
                            plots side by side.
            z_increment: Increment field, shape (n_points, n_variables). If provided with z_background,
                            plots side by side.
        diagnostics: Optional string of additional info
    """
    
    grid_lats = da_output['grid_lats']
    grid_lons = da_output['grid_lons']
    z_background = da_output['x_background']
    z_increment = da_output['x_increment']
    var_list = da_output['var_list']
    
    # Filter var_list to only include variables in plot_vars
    if plot_vars is None or plot_vars == 'all':
        plot_vars = var_list
    
    n_rows = len(plot_vars)
    n_cols = 2
    figsize = (12, 3.5 + 2.5 * n_rows)
    
    fig = plt.figure(figsize=figsize, layout="constrained")
    triang, mask = create_triangulation(proj, grid_lats, grid_lons)

    for i, varname in enumerate(plot_vars):
        #    varname = plot_vars[i]
        unit = unit_for_var(varname)
        mask_var = var_list.index(varname)
        
        #######################
        # plot background
        col_idx = 0 
        kw = plotkws_for_var(varname)
        data  = z_background[:, mask_var][mask]
        data = np.clip(data, kw['vmin'], kw['vmax'])

        # Calculate subplot index
        subplot_idx = i * n_cols + col_idx + 1
        ax = fig.add_subplot(n_rows, n_cols, subplot_idx, projection=proj)
        if 'levels' not in kw:
            locator = mticker.MaxNLocator(nbins=30)
            kw['levels'] = locator.tick_values(data.min(), data.max())
        tcf = ax.tricontourf(triang, data, **kw)

        if obs_lons is not None and obs_lats is not None:
            ax.scatter(obs_lons, obs_lats, c='gold', s=1, zorder=1000, transform=ccrs.PlateCarree())
        
        cbar = plt.colorbar(tcf, label=unit, extend='both')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        gl = ax.gridlines()
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, step=30))
        gl.ylocator = mticker.FixedLocator(np.arange(-80, 80+1, step=20))
        ax.coastlines()
        ax.set_title(f'{varname} background')
        
        #######################
        # plot increment
        col_idx = 1
        # Calculate subplot index
        subplot_idx = i * n_cols + col_idx + 1
        ax = fig.add_subplot(n_rows, n_cols, subplot_idx, projection=proj)
        tcf = ax.tricontourf(triang, z_increment[:, mask_var][mask],
                             norm = mcolors.CenteredNorm(),  # symmetric colorbar
                             cmap = 'bwr')
        
        if obs_lons is not None and obs_lats is not None:
            ax.scatter(obs_lons, obs_lats, c='gold', s=1, zorder=1000, transform=ccrs.PlateCarree())
        
        cbar = plt.colorbar(tcf, label=unit, extend='both')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        gl = ax.gridlines()
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, step=30))
        gl.ylocator = mticker.FixedLocator(np.arange(-80, 80+1, step=20))
        ax.coastlines()
        ax.set_title(f'{varname} increment')
        
    info = f"AE version: {args.AE_version}, VarDA type: {args.VarDA_type}, obs_datetime: {args.obs_datetime}, obs_qty: {args.obs_qty}, obs_dep: {args.obs_dep}, obs_std: {args.obs_std}, ilr: {args.init_lr}"
    if diagnostics is not None:
        diagnostics_str = f"y_b={float(diagnostics['y_b_decoded_dest']):.2f}, obs={float(diagnostics['obs_vec']):.2f}, innovation={float(diagnostics['innovation']):.2f}, analysis increment={float(diagnostics['ana_inc_obs_loc']):.2f}"
        info += f"\n diagnostics: {diagnostics_str}"
    #fig.text(0.5, 0.02, info, ha='center', va='bottom', fontsize=6)

    
    if args.singobs_lat is not np.nan:
        loc = f"singobs_{args.singobs_lat}_{args.singobs_lon}"
    else:
        loc = "multiobs"
    
    fpath = output_dir + f"/background_and_increment_{args.AE_version}_{args.VarDA_type}_{loc}_{args.custom_addon}_{args.obs_datetime}_obs_{args.obs_qty}_obs_inc_{args.obs_dep}_obs_std_{args.obs_std}_ilr_{args.init_lr}_{proj.__class__.__name__}.png"
    
    plt.savefig(fpath)
    print(fpath)

    with open(fpath+'-diag.txt', "w") as f:
        f.write(info)


def plot_cross_section(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    field: np.ndarray,
    var_list: List[str],
    AE_props,  # AE properties object
    args,  # argparse.Namespace
    output_dir: str,
    latitude_for_lon_section: float = 78.0,
    longitude_for_lat_section: float = 0.0,
    lon_range: tuple = (-10, 10),
    lat_range: tuple = (72, 82),
    n_interp_points: int = 500,
    extension: str = ''
) -> None:
    """Plot cross sections of field data along latitude and longitude circles.
    
    Args:
        grid_lats: Array of latitude values for the grid
        grid_lons: Array of longitude values for the grid
        field: Field data to plot, shape (n_points, n_variables)
        var_list: List of variable names to plot
        AE_props: Autoencoder properties object
        args: Argument namespace containing experiment parameters
        output_dir: Directory to save the output figure
        latitude_for_lon_section: Latitude at which to take longitude cross-section
        longitude_for_lat_section: Longitude at which to take latitude cross-section
        lon_range: Tuple of (min_lon, max_lon) for longitude cross-section
        lat_range: Tuple of (min_lat, max_lat) for latitude cross-section
        n_interp_points: Number of interpolation points for each cross-section
    """
    print('Plotting cross sections of increments')
    
    # Get indices for variables
    ivar_list = [AE_props.reconstructed_variables.index(var) for var in var_list]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # 1) Plot cross section along latitude circle (varying longitude)
    grid_lons_std = (grid_lons + 180) % 360 - 180
    
    if lon_range[0] > lon_range[1]:  # wraps around dateline
        # If the range crosses the dateline, we need to handle it carefully
        interp_lons1 = np.linspace(lon_range[0], lon_range[1], n_interp_points)
        # We will interpolate in two segments: from lon_range[0] to 180, and from -180 to lon_range[1]
        interp_lons1_part1 = np.linspace(lon_range[0], 180, n_interp_points // 2, endpoint=False)
        interp_lons1_part2 = np.linspace(-180, lon_range[1], n_interp_points // 2)
        interp_lons1 = np.concatenate((interp_lons1_part1, interp_lons1_part2))
    else:
        interp_lons1 = np.linspace(lon_range[0], lon_range[1], n_interp_points)
    interp_lats1 = np.ones_like(interp_lons1) * latitude_for_lon_section
    
    # Interpolate the data onto this line
    points = np.column_stack((grid_lons_std, grid_lats))
    values = field[:, ivar_list]
    along_latitude = griddata(points, values, (interp_lons1, interp_lats1), method='linear')
    
    # 2) Plot cross section along longitude circle (varying latitude)
    interp_lats2 = np.linspace(lat_range[0], lat_range[1], n_interp_points)
    interp_lons2 = np.ones_like(interp_lats2) * longitude_for_lat_section
    
    # Interpolate the data onto this line
    points = np.column_stack((grid_lons, grid_lats))
    values = field[:, ivar_list]
    along_longitude = griddata(points, values, (interp_lons2, interp_lats2), method='linear')
    
    # Determine color scale limits
    mmax = max(np.nanmax(np.abs(along_latitude).flatten()), 
               np.nanmax(np.abs(along_longitude).flatten()))
    print('mmax', mmax)
    
    # Plot longitude cross-section
    axs[0].contourf(
        interp_lons1,
        np.array(var_list),
        along_latitude.T,
        levels=20,
        cmap='RdBu_r',
        vmin=-mmax, vmax=mmax
    )
    axs[0].set_xlabel(f'Longitude (deg) at {latitude_for_lon_section}°N')
    
    # Plot latitude cross-section
    h = axs[1].contourf(
        interp_lats2,
        np.array(var_list),
        along_longitude.T,
        levels=20,
        cmap='RdBu_r',
        vmin=-mmax, vmax=mmax
    )
    axs[1].set_xlabel(f'Latitude (deg) at {longitude_for_lat_section}°E')
    axs[0].set_ylabel('Variable')
    
    plt.gca().invert_yaxis()
    plt.colorbar(h, ax=axs, label='T increment (K)')
    plt.suptitle(f"Temperature increments, {args.obs_datetime}, obs: {args.obs_qty}, obs_dep={args.obs_dep}, obs_std={args.obs_std}, ilr={args.init_lr}")
    
    fpath = output_dir + f"/increments_cross2_{args.AE_version}_{args.VarDA_type}_{args.custom_addon}_{args.obs_datetime}_obs_{args.obs_qty}_obs_inc_{args.obs_dep}_obs_std_{args.obs_std}_ilr_{args.init_lr}_{extension}.png"
    plt.savefig(fpath)
    print(fpath)