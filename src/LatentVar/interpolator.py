import torch


def _interpolate_knn(grid_xyz, grid_values, target_xyz, k=4, eps=1e-6):
    """Inverse-distance weighted KNN interpolation in Cartesian space.

    Args:
        grid_xyz: (M, 3) Cartesian coordinates of grid points
        grid_values: (F, M) values at grid points
        target_xyz: (T, 3) Cartesian coordinates of target points
        k: number of nearest neighbors
        eps: small value to avoid division by zero
    Returns:
        interpolated: (F, T) interpolated values at target points
    """
    if grid_values.shape[0] == grid_xyz.shape[0]:
        grid_values = grid_values.T  # ensure (F, M)

    F, M = grid_values.shape
    T = target_xyz.shape[0]

    dists = torch.cdist(target_xyz, grid_xyz)  # (T, M)
    knn_dists, knn_indices = torch.topk(dists, k, dim=1, largest=False)  # (T, k)

    knn_indices_exp = knn_indices.unsqueeze(0).expand(F, -1, -1)  # (F, T, k)
    knn_values = torch.gather(grid_values.unsqueeze(1).expand(-1, T, -1), 2, knn_indices_exp)  # (F, T, k)

    weights = 1.0 / (knn_dists + eps)  # (T, k)
    weights = weights / weights.sum(dim=1, keepdim=True)
    weights = weights.unsqueeze(0)  # (1, T, k)

    return (weights * knn_values).sum(dim=2)  # (F, T)


def normalize_longitudes(lons):
    # 0 - 360 -> -180 - +180
    # Use for: (1) plotting, (2) interpolation (observation longitudes are given as -180 - +180)
    return ((lons + 180) % 360) - 180

def interpolate_irregular_grid(grid_lats, grid_lons, grid_values, target_lats, target_lons, k=4):
    """Interpolate from an irregular grid to target points using k-nearest neighbors in Cartesian space.
    
    Note: grid_lons are expected to be in the range 0 to 360,
    while target_lons are expected to be in the range -180 to 180.
    
    Args:
        grid_lats: (M,) tensor of latitudes of the irregular grid points
        grid_lons: (M,) tensor of longitudes of the irregular grid points (0 to 360)
        grid_values: (F, M) tensor of values at the irregular grid points
        target_lats: (N,) tensor of target latitudes (-90 to 90)
        target_lons: (N,) tensor of target longitudes (-180 to 180)
        k: number of nearest neighbors to use for interpolation
    """    
    # check user inputs
    #assert all(target_lons >= -180) and all(target_lons <= 180), "Target longitudes must be in the range -180 to 180"
    assert all(target_lats >= -90) and all(target_lats <= 90), "Target latitudes must be in the range -90 to 90"
    #assert all(grid_lons >= 0) and all(grid_lons < 360), "Grid longitudes must be in the range 0 to 360"

    # could we not just use numpy for this function?
    if not isinstance(grid_lons, torch.Tensor):
        grid_lons = torch.as_tensor(grid_lons)
    if not isinstance(grid_lats, torch.Tensor):
        grid_lats = torch.as_tensor(grid_lats)
    if not isinstance(target_lons, torch.Tensor):
        target_lons = torch.as_tensor(target_lons)
    if not isinstance(target_lats, torch.Tensor):
        target_lats = torch.as_tensor(target_lats)
    if not isinstance(grid_values, torch.Tensor):
        grid_values = torch.as_tensor(grid_values)

    # Normalize longitudes
    grid_lons = normalize_longitudes(grid_lons) # grid lons are initially 0 to 358.875
    # Target lons are originally already -180 to 180

    # Convert to Cartesian
    def latlon_to_cartesian(lat, lon):
        # Convert numpy arrays to torch tensors if needed
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        return torch.stack([x, y, z], dim=-1)  # shape: (..., 3)

    grid_xyz = latlon_to_cartesian(grid_lats, grid_lons)  # (M, 3)
    target_xyz = latlon_to_cartesian(target_lats, target_lons)  # (N, 3)

    return _interpolate_knn(grid_xyz, grid_values, target_xyz, k)

