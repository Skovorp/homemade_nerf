import yaml
from data import LegoDatasetLazy, collate_rays
from torch.utils.data import DataLoader
from model import Nerf
from utils import sample_points, render

import torch
from datetime import datetime
import os
import torchvision


with open('cfg.yaml', 'r') as f:
    cfg = yaml.safe_load(f)


train_set = LegoDatasetLazy('train')
train_loader = DataLoader(
    train_set,
    batch_size=cfg['train']['batch_size'],
    collate_fn=collate_rays,
    shuffle=True
)
val_set = LegoDatasetLazy('val')
val_loader = DataLoader(
    train_set,
    batch_size=cfg['train']['batch_size'],
    collate_fn=collate_rays,
    shuffle=False
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
    model.train()
    for rays in train_loader:
        origin = rays['origin'].to(device)
        direction = rays['direction'].to(device)
        target_color = rays['color'].to(device)
        
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
        print(((color > 0.95) * 1.0).mean(), color.mean())
        print('loss', loss)
    
    model.eval()
    with torch.no_grad():
        val_images = {}
        for rays in val_loader:
            origin = rays['origin'].to(device)
            direction = rays['direction'].to(device)
            target_color = rays['color'].to(device)
            
            points, z_vals = sample_points(origin, direction, n_sample) 
            points = points.reshape(-1, 3)
            vals = model(points)
            vals = vals.reshape(bs, n_sample, 4)
            color = render(vals, z_vals, direction)
            loss = ((color  - target_color) ** 2).mean()
            
            for k in range(len(rays['image_path'])):
                pth = rays['image_path'][k]
                if pth not in val_images:
                    val_images[pth] = torch.zeros(3, 800, 800)
                val_images[pth][:, rays['i'][k], rays['j'][k]] = color[k, :].cpu()
                    
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_root = f'val_outputs_{timestamp}_epoch_{epoch}/'
        os.makedirs(save_root, exist_ok=True)
        for pth, img in val_images.items():
            torchvision.utils.save_image(img, save_root + os.path.basename(pth))
