import yaml
from data import LegoDatasetLazy, collate_rays
from torch.utils.data import DataLoader
from model import Nerf
from utils import sample_points, render, psnr

import torch
from datetime import datetime
import os
import torchvision
from tqdm import tqdm
from collections import defaultdict
import wandb


timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

with open('cfg.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

wandb.init(
    config=cfg,
    project="homemade_nerf")

train_set = LegoDatasetLazy('train', scale=cfg['train']['data_scale'])
train_loader = DataLoader(
    train_set,
    batch_size=cfg['train']['train_batch_size'],
    collate_fn=collate_rays,
    num_workers=8,
    pin_memory=True,
    shuffle=True
)
val_set = LegoDatasetLazy('val', cap=cfg['train']['val_cap'], scale=cfg['train']['data_scale'])
val_loader = DataLoader(
    val_set,
    batch_size=cfg['train']['val_batch_size'],
    collate_fn=collate_rays,
    num_workers=8,
    pin_memory=True,
    shuffle=False
)
print("Trainset len:", len(train_set))
print("Valset   len:", len(val_set))

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = Nerf(**cfg['model']).to(device)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg['train']['lr'],
    weight_decay=cfg['train']['weight_decay']
)


n_sample = cfg['nerf']['sample_points']
for epoch in range(cfg['train']['epochs']):
    model.train()
    for step, rays in tqdm(enumerate(train_loader), total=cfg['train']['steps_per_epoch']):
        origin = rays['origin'].to(device)
        direction = rays['direction'].to(device)
        target_color = rays['color'].to(device)
        bs = target_color.shape[0]
        
        optimizer.zero_grad()
        # bs, num_points, xyz
        points, z_vals = sample_points(origin, direction, n_sample) 
        # torch.save(points, 'hui2.pt')
        points = points.reshape(-1, 3)
        vals = model(points)
        vals = vals.reshape(bs, n_sample, 4)
        color = render(vals, z_vals, direction)
        loss = ((color  - target_color) ** 2).mean()
        loss.backward()
        optimizer.step()
        wandb.log({
            'loss': loss,
            'psnr': psnr(loss),
            'part_white': ((color > 0.95) * 1.0).mean(),
            'color_mean': color.mean()
        })
        if step == cfg['train']['steps_per_epoch'] - 1:
            break
    
    model.eval()
    with torch.no_grad():
        val_images = {}
        list_is = []
        list_js = []
        list_paths = []
        colors = []
        losses = []
        cnt = 0
        for rays in tqdm(val_loader):
            origin = rays['origin'].to(device)
            direction = rays['direction'].to(device)
            target_color = rays['color'].to(device)
            bs = target_color.shape[0]
            
            points, z_vals = sample_points(origin, direction, n_sample) 
            points = points.reshape(-1, 3)
            vals = model(points)
            vals = vals.reshape(bs, n_sample, 4)
            color = render(vals, z_vals, direction)
            losses.append(((color  - target_color) ** 2).sum())
            cnt += bs
            
            list_is.extend(rays['i'])
            list_js.extend(rays['j'])
            list_paths.extend(rays['image_path'])
            colors.append(color.cpu())
        
        loss = torch.tensor(losses).sum() / cnt
        wandb.log({
            'val_loss': loss,
            'val_psnr': psnr(loss)},
            commit=False
        )
            
        path_to_indices = defaultdict(list)
        for idx, path in enumerate(list_paths):
            path_to_indices[path].append(idx)
        
        save_root = f'val_outputs_{timestamp}/epoch_{epoch}/'
        os.makedirs(save_root, exist_ok=True)
        print("saving images")
        
        colors = torch.cat(colors, dim=0)  # [N, 3]
        is_tensor = torch.tensor(list_is, dtype=torch.long)
        js_tensor = torch.tensor(list_js, dtype=torch.long)
        
        for path, indices in path_to_indices.items():
            idxs = torch.tensor(indices, dtype=torch.long)
            i = is_tensor[idxs]
            j = js_tensor[idxs]
            c = colors[idxs]  # [N, 3]

            img = torch.ones(3, 800 // cfg['train']['data_scale'], 800 // cfg['train']['data_scale'])
            img[1, :, :] = 0
            img[:, i, j] = c.T  # assign all pixels at once
            torchvision.utils.save_image(img, save_root + os.path.basename(path))
