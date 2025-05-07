import yaml
from data import LegoDatasetLazy, collate_rays
from torch.utils.data import DataLoader
from model import Nerf
from utils import sample_points, render

import torch
from datetime import datetime
import os
import torchvision
from tqdm import tqdm
from collections import defaultdict


with open('cfg.yaml', 'r') as f:
    cfg = yaml.safe_load(f)


train_set = LegoDatasetLazy('train')
train_loader = DataLoader(
    train_set,
    batch_size=cfg['train']['train_batch_size'],
    collate_fn=collate_rays,
    shuffle=True
)
val_set = LegoDatasetLazy('val', cap=1)
val_loader = DataLoader(
    val_set,
    batch_size=cfg['train']['val_batch_size'],
    collate_fn=collate_rays,
    shuffle=False
)
print("Trainset len:", len(train_set))
print("Valset   len:", len(val_set))

device = torch.device('cuda')
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
    for step, rays in enumerate(train_loader):
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
        print(((color > 0.95) * 1.0).mean(), color.mean())
        print(step, '\tloss', loss)
        if step > 1000:
            break
    
    model.eval()
    with torch.no_grad():
        val_images = {}
        list_is = []
        list_js = []
        list_paths = []
        colors = []
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
            loss = ((color  - target_color) ** 2).mean()
            
            # for idx, path in enumerate(rays['image_path']):
            # image_data[path]['i'].append(rays['i'][idx])
            # image_data[path]['j'].append(rays['j'][idx])
            # image_data[path]['color'].append(color[idx].cpu())
            list_is.extend(rays['i'])
            list_js.extend(rays['j'])
            list_paths.extend(rays['image_path'])
            colors.append(color.cpu())
            
            
            # for k in range(len(rays['image_path'])):
            #     pth = rays['image_path'][k]
            #     if pth not in val_images:
            #         val_images[pth] = torch.ones(3, 800, 800)
            #         val_images[pth][1, :, :] = 0
            #     val_images[pth][:, rays['i'][k], rays['j'][k]] = color[k, :].cpu()
        path_to_indices = defaultdict(list)
        for idx, path in enumerate(list_paths):
            path_to_indices[path].append(idx)
        
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_root = f'val_outputs_{timestamp}_epoch_{epoch}/'
        os.makedirs(save_root, exist_ok=True)
        print("saving images")
        
        colors = torch.cat(colors, dim=0)  # [N, 3]
        is_tensor = torch.tensor(list_is, dtype=torch.long)
        js_tensor = torch.tensor(list_js, dtype=torch.long)
        
        # print(is_tensor)
        # print(js_tensor)
        
        for path, indices in path_to_indices.items():
            idxs = torch.tensor(indices, dtype=torch.long)
            i = is_tensor[idxs]
            j = js_tensor[idxs]
            c = colors[idxs]  # [N, 3]

            img = torch.ones(3, 800, 800)
            img[1, :, :] = 0
            img[:, i, j] = c.T  # assign all pixels at once
            torchvision.utils.save_image(img, save_root + os.path.basename(path))

        
        # for pth, img in val_images.items():
        #     torchvision.utils.save_image(img, save_root + os.path.basename(pth))
