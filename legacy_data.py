class LegoDataset(Dataset):
    def __init__(self, partition, load_cache=True):
        if partition == 'train':
            self.images_pth = '/Users/ksc/PycharmProjects/cv_nerf_proj/homemade_nerf/nerf_synthetic/lego/'
            self.transforms = '/Users/ksc/PycharmProjects/cv_nerf_proj/homemade_nerf/nerf_synthetic/lego/transforms_train.json'
        

        with open(self.transforms, 'r') as f:
            self.transforms = json.load(f)
            self.camera_angle_x = self.transforms['camera_angle_x']
            self.frames = self.transforms['frames']
        self.rays = []
        self.cache_file = f'{partition}.pickle'
        if load_cache and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.rays = pickle.load(f)
            print(f"Loaded cached rays from {self.cache_file}")
        else:
            self.process_images()
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.rays, f)
            print(f"Processed images and cached rays to {self.cache_file}")
    
    def process_images(self, ):
        for frame in tqdm(self.frames[:5]):
            img = os.path.join(self.images_pth, frame['file_path'][2:] + '.png')
            img = read_image(img)
            fov = torch.tensor(frame['rotation'])
            focal_length = 800 / (2.0 * torch.tan(fov / 2.0))
            transform = torch.tensor(frame['transform_matrix'])
            self.get_rays_from_image(img, transform, focal_length)
        
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
    
    