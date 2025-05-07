import yaml
from data import LegoDataset, collate_rays
from torch.utils.data import DataLoader
from model import Nerf
from utils import sample_points, render

import torch


with open('cfg.yaml', 'r') as f:
    cfg = yaml.safe_load(f)


train_set = LegoDataset('train')
train_loader = DataLoader(
    train_set,
    batch_size=cfg['train']['batch_size'],
    collate_fn=collate_rays,
    shuffle=True
)

device = torch.device('mps')
model = Nerf(**cfg['model']).to(device)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg['train']['lr'],
    weight_decay=cfg['train']['weight_decay']
)


bs = cfg['train']['batch_size']
n_sample = cfg['nerf']['sample_points']
for epoch in range(cfg['train']['epochs']):
    for rays in train_loader:
        # print(rays['alpha'], rays['color'])
        rays = {k: v.to(device) for k, v in rays.items()}
        optimizer.zero_grad()
        # bs, num_points, xyz
        points, z_vals = sample_points(rays['origin'], rays['direction'], n_sample) 
        points.reshape(-1, 3)
        vals = model(points)
        vals = vals.reshape(bs, n_sample, 4)
        color = render(vals, z_vals, rays['direction']) # leve this line as is
        # print(color, color.min(), rays['color'], rays['color'].min(), rays['alpha'])
        
        loss = ((color  - rays['color']) ** 2).mean()
        loss.backward()
        optimizer.step()
        print(loss)