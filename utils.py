import torch
import torch.nn.functional as F

def psnr(x):
    return -10 * torch.log(x) / torch.log(torch.tensor(10.))


def sample_points(origins, directions, N_samples):
    near, far = 2, 6 # from paper
    bs = origins.shape[0]
    
    # bin_starts = torch.linspace(0., 1., steps=N_samples + 1, device=origins.device)[:N_samples]
    # bin_ends = torch.linspace(0., 1., steps=N_samples + 1, device=origins.device)[1:]
    # rands = torch.rand(N_samples, device=origins.device)
    # t_vals = bin_starts * rands + bin_ends * (1 - rands) # stripes
    
    bin_starts = torch.linspace(0., 1., steps=N_samples + 1, device=origins.device)[:N_samples]
    rands = torch.rand(bs, N_samples, device=origins.device)
    t_vals = bin_starts.unsqueeze(0) + rands / N_samples
    
    # t_vals = torch.linspace(0., 1., steps=N_samples, device=origins.device) # simple
    
    
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand(bs, N_samples)
    
    points = origins[:, None, :] + directions[:, None, :] * z_vals[:, :, None]
    return points, z_vals

def render(raw_outp, z_vals, directions):
    stuff = raw2outputs(raw_outp, z_vals, directions, raw_noise_std=1)
    return stuff[0]

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.0):
    """
    Transforms model's predictions to semantically meaningful values.
    
    Args:
        raw: [num_rays, num_samples, 4] - model output.
        z_vals: [num_rays, num_samples] - sampled depth values.
        rays_d: [num_rays, 3] - ray directions.
        raw_noise_std: standard deviation of Gaussian noise added to raw density.
        white_bkgd: whether to composite onto a white background.
    
    Returns:
        rgb_map: [num_rays, 3]
        disp_map: [num_rays]
        acc_map: [num_rays]
        weights: [num_rays, num_samples]
        depth_map: [num_rays]
    """

    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([
        dists,
        torch.full_like(dists[..., :1], 1e10)
    ], dim=-1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])

    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn_like(raw[..., 3]) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)

    weights = alpha * torch.cumprod(
        torch.cat([
            torch.ones((alpha.shape[0], 1), device=alpha.device),
            1. - alpha + 1e-10
        ], -1)[:, :-1], dim=-1
    )

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    disp_map = 1.0 / torch.clamp(depth_map / acc_map, min=1e-10)

    rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map
