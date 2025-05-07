import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

import json
import os


class LegoDataset(Dataset):
    def __init__(self, partition):
        if partition == 'train':
            self.images_pth = '/Users/ksc/PycharmProjects/cv_nerf_proj/homemade_nerf/nerf_synthetic/lego/'
            self.transforms = '/Users/ksc/PycharmProjects/cv_nerf_proj/homemade_nerf/nerf_synthetic/lego/transforms_train.json'

        with open(self.transforms, 'r') as f:
            self.transforms = json.load(f)
            self.camera_angle_x = self.transforms['camera_angle_x']
            self.frames = self.transforms['frames']
        self.rays = []
        self.process_images()
        
    
    def process_images(self, ):
        for frame in self.frames:
            img = os.path.join(self.images_pth, frame['file_path'][2:] + '.png')
            img = read_image(img)
            fov = torch.tensor(frame['rotation'])
            focal_length = 800 / (2.0 * torch.tan(fov / 2.0))
            transform = torch.tensor(frame['transform_matrix'])
            self.get_rays_from_image(img, transform, focal_length)
            break
        
    def get_rays_from_image(self, img, transform, focal_length):
        H, W = img.shape[1], img.shape[2]
        img = img / 255
        image = img[:3, :, :] 
        alpha = img[3:, :, :]
        image = image * alpha + (1 - alpha)
        print("image", ((image == 1) * 1.0).mean())
        res = []
        for i in range(H):
            for j in range(W):
                # Camera-space coordinates
                x = (j - W / 2) / focal_length
                y = -(i - H / 2) / focal_length
                z = -1.0  # pointing outwards in OpenGL convention

                direction_camera = torch.tensor([x, y, z], dtype=torch.float32)

                # Convert to world space
                rotation = transform[:3, :3]
                direction_world = rotation @ direction_camera
                direction_world = direction_world / torch.norm(direction_world)

                ray = {
                    'origin': transform[:3, 3],  # (3,)
                    'direction': direction_world,  # (3,)
                    'color': image[:, i, j],       # (3,)
                    'alpha': alpha[0, i, j]       # scalar
                }
                res.append(ray)
        self.rays.extend(res)

    def __len__(self, ):
        return len(self.rays)
    
    def __getitem__(self, i):
        # outputs color, ray origin, direction, lower and upper bounds
        return self.rays[i] 
    
    
def collate_rays(batch):
    origins = torch.stack([ray['origin'] for ray in batch], dim=0)    # (N, 3)
    directions = torch.stack([ray['direction'] for ray in batch], dim=0)  # (N, 3)
    colors = torch.stack([ray['color'] for ray in batch], dim=0)      # (N, 3)
    alphas = torch.stack([ray['alpha'] for ray in batch], dim=0) # (N,)

    return {
        'origin': origins,
        'direction': directions,
        'color': colors,
        'alpha': alphas
    }

    
if __name__ == "__main__":
    data = LegoDataset('train')
    print(data[0])
    for i in range(len(data)):
        print(data[i]['color'])
    