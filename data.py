import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize

import json
import os
from tqdm import tqdm
import pickle


class LegoDatasetLazy(Dataset):
    def __init__(self, partition, scale=1, cap=None):
        # self.images_pth = '/Users/ksc/PycharmProjects/cv_nerf_proj/homemade_nerf/nerf_synthetic/lego/'
        # self.transforms = f'/Users/ksc/PycharmProjects/cv_nerf_proj/homemade_nerf/nerf_synthetic/lego/transforms_{partition}.json'
        self.images_pth = '/root/nerf_synthetic/lego/'
        self.transforms = f'/root/nerf_synthetic/lego/transforms_{partition}.json'

        self.s = 800 // scale
        with open(self.transforms, 'r') as f:
            self.transforms = json.load(f)
            self.camera_angle_x = self.transforms['camera_angle_x']
            self.frames = self.transforms['frames']
            if cap is not None:
                self.frames = self.frames[:cap]
        
        self.images = []
        self.image_paths = []
        for frame in self.frames:
            fov = torch.tensor(self.camera_angle_x)
            focal_length = self.s / (2.0 * torch.tan(fov / 2.0))
            img_path = os.path.join(self.images_pth, frame['file_path'][2:] + '.png')
            self.image_paths.append(img_path)
            img = read_image(img_path)
            if scale != 1:
                img = Resize(size=(self.s,self.s))(img)
            img = img / 255
            image = img[:3, :, :]
            alpha = img[3:, :, :]
            image = image * alpha + (1 - alpha)
            transform = torch.tensor(frame['transform_matrix'])
            self.images.append((image, alpha, focal_length, transform))
    
    def __len__(self, ):
        return len(self.frames) * self.s * self.s
    
    def get_ray(self, transform_ind, i, j):
        # print(transform_ind, i, j)
        image, alpha, focal_length, transform = self.images[transform_ind]

        x = (j - self.s / 2) / focal_length
        y = -(i - self.s / 2) / focal_length
        z = -1.0  # pointing outwards in OpenGL convention

        direction_camera = torch.tensor([x, y, z], dtype=torch.float32)

        # Convert to world space
        rotation = transform[:3, :3]
        direction_world = rotation @ direction_camera
        direction_world = direction_world / torch.norm(direction_world)

        return {
            'origin': transform[:3, 3],  # (3,)
            'direction': direction_world,  # (3,)
            'color': image[:, i, j],       # (3,)
            'alpha': alpha[0, i, j]       # scalar
        }
    
    def __getitem__(self, ind):
        transform_ind = ind // (self.s * self.s)
        ind = ind % (self.s * self.s)
        i = ind // self.s
        j = ind % self.s
        res = self.get_ray(transform_ind, i, j)
        res['i'] = i
        res['j'] = j
        res['image_path'] = self.image_paths[transform_ind]
        return res
    
def collate_rays(batch):
    origins = torch.stack([ray['origin'] for ray in batch], dim=0)    # (N, 3)
    directions = torch.stack([ray['direction'] for ray in batch], dim=0)  # (N, 3)
    colors = torch.stack([ray['color'] for ray in batch], dim=0)      # (N, 3)
    alphas = torch.stack([ray['alpha'] for ray in batch], dim=0) # (N,)

    return {
        'origin': origins,
        'direction': directions,
        'color': colors,
        'alpha': alphas,
        'i': [x['i'] for x in batch],
        'j': [x['j'] for x in batch],
        'image_path': [x['image_path'] for x in batch],
    }

    
if __name__ == "__main__":
    data = LegoDatasetLazy('val')
    print(data[0])
    print(data[0])
    print(data[97482])
    # for i in range(len(data)):
    #     print(data[i]['color'])
    