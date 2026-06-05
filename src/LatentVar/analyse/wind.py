# wind.py
import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree

R = 6371000.0        # Earth radius [m]
Omega = 7.2921e-5    # Earth's rotation rate [1/s]


# --------------------------------------------------
# Precompute least-squares gradient weights
# --------------------------------------------------
def precompute_weights(lat, lon, neighbors):
    """
    lat, lon : (N,) radians
    neighbors: (N, k) integer indices
    """
    N, k = neighbors.shape

    wx = np.zeros((N, k))
    wy = np.zeros((N, k))

    for p in range(N):
        idx = neighbors[p]

        lat0 = lat[p]
        lon0 = lon[p]

        lat_i = lat[idx]
        lon_i = lon[idx]

        # wrapped longitude difference
        dlon = (lon_i - lon0 + np.pi) % (2*np.pi) - np.pi
        dlat = lat_i - lat0

        # local tangent plane
        x = R * np.cos(lat0) * dlon
        y = R * dlat

        A = np.stack([x, y], axis=1)  # (k, 2)

        # solve least squares weights: W = (A^T A)^(-1) A^T
        ATA = A.T @ A + 1e-12 * np.eye(2)
        W = np.linalg.solve(ATA, A.T)

        wx[p, :] = W[0]
        wy[p, :] = W[1]

    return wx, wy


# --------------------------------------------------
# Compute gradient
# --------------------------------------------------
def compute_gradient(phi, neighbors, wx, wy):
    """
    phi      : (N,) scalar field
    neighbors: (N, k)
    wx, wy   : (N, k)
    """
    phi_nb = phi[neighbors]        # (N, k)
    dphi = phi_nb - phi[:, None]   # subtract center

    dphidx = np.sum(wx * dphi, axis=1)
    dphidy = np.sum(wy * dphi, axis=1)

    return dphidx, dphidy


# --------------------------------------------------
# Compute geostrophic wind
# --------------------------------------------------
def geostrophic_wind(dphidx, dphidy, lat, mask_equator=True):
    """
    lat: (N,) radians
    """
    f = 2 * Omega * np.sin(lat)

    if mask_equator:
        mask = np.abs(lat) < 5  * np.pi / 180  # mask within 5 degrees of the equator
        f[mask] = np.nan  # avoid blow-up

    ug = -dphidy / f
    vg =  dphidx / f

    return ug, vg


# --------------------------------------------------
# Example usage (minimal test)
# --------------------------------------------------
def compute_geostrophic(z500):
    # synthetic example

    # load N80 mesh
    f_grid = '/ceph/hpc/home/kuglerl/data/N80/N80_coords.nc'
    ds = xr.open_dataset(f_grid)
    lat_deg = ds['latitude'].values
    lon_deg = ds['longitude'].values
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    N = len(lat)
    
    # number of neighbors for gradient estimation
    k = 12

    # nearest neighbors
    xyz = np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])

    tree = BallTree(xyz)
    dist, neighbors_idx = tree.query(xyz, k=k+1)
    neighbors_idx = neighbors_idx[:, 1:]  # remove self
    
    
    # field
    # test
    assert z500.shape == lat.shape
    phi = z500

    # compute
    wx, wy = precompute_weights(lat, lon, neighbors_idx)
    dphidx, dphidy = compute_gradient(phi, neighbors_idx, wx, wy)
    ug, vg = geostrophic_wind(dphidx, dphidy, lat)

    return ug, vg, lat_deg, lon_deg

def plot_geostrophic(ug, vg, lon, lat):
    # load N80 mesh
    f_grid = '/ceph/hpc/home/kuglerl/data/N80/N80_coords.nc'
    ds = xr.open_dataset(f_grid)
    lat_deg = ds['latitude'].values
    lon_deg = ds['longitude'].values
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    # plot on Robinson map
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    lon_deg = np.rad2deg(lon)
    lat_deg = np.rad2deg(lat)
    speed = np.hypot(ug, vg)
    speed_valid = np.isfinite(speed)
    lon_valid = ((lon_deg[speed_valid] + 180) % 360) - 180
    lat_valid = lat_deg[speed_valid]
    speed_values = speed[speed_valid]

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()

    vmin = np.nanpercentile(speed_values, 5)
    vmax = np.nanpercentile(speed_values, 95)
    sc = ax.scatter(
        lon_valid,
        lat_valid,
        c=speed_values,
        cmap='viridis',
        s=8,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )
    ax.set_title('Geostrophic wind speed magnitude')
    ax.coastlines(linewidth=0.6)
    ax.gridlines(linewidth=0.4, alpha=0.5)
    fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.06, label='Wind speed')

    plt.tight_layout()
    plt.savefig('geostrophic_wind_magnitude_robinson.png', dpi=200)




