import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
#import cv2
import numpy as np


class CityScapesDataset(Dataset):
    def __init__(self, root_dir, image_transform, seg_transform, depth_transform):
        self.root_dir = root_dir                    # CityScapes/train or CityScapes/val
        self.image_transform = image_transform
        self.seg_transform = seg_transform
        self.depth_transform = depth_transform
        self.rgb_paths = []
        self.seg_paths = []
        self.depth_paths = []
        
        rgb_dir = os.path.join(self.root_dir, 'rgb')
        seg_dir = os.path.join(self.root_dir, 'semantic')
        depth_dir = os.path.join(self.root_dir, 'depth')
        
        for file_name in os.listdir(rgb_dir):
            self.rgb_paths.append(os.path.join(rgb_dir, file_name))
            self.seg_paths.append(os.path.join(seg_dir, file_name))
            self.depth_paths.append(os.path.join(depth_dir, file_name))
            
    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_paths[idx])
        seg_map = Image.open(self.seg_paths[idx])
        depth_map = Image.open(self.depth_paths[idx])

        if self.image_transform:
            rgb_image = self.image_transform(rgb_image)
        if self.seg_transform:
            seg_map = self.seg_transform(seg_map)
        if self.depth_transform:
            depth_map = self.depth_transform(depth_map)

        return rgb_image, seg_map, depth_map
    
    
class ImageTransform:
    def __call__(self, image):
        #resize = transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST)
        resize = transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST)
        image = resize(image)
        
        image = np.array(image, dtype=np.float32)
        #image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_AREA)
        image = image / 255.0
        
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        return image
    
    
class SegTransform:
    def __call__(self, map):
        #resize = transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST)
        resize = transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST)
        map = resize(map)
        #map = cv2.resize(map, (1024, 512), interpolation=cv2.INTER_AREA)
        
        map = np.asarray(map, dtype=np.uint8)
        map = torch.tensor(map, dtype=torch.long).unsqueeze(0)  # Add channel dimension (C, H, W)
        
        return map
    
    
class CityDepthTransform:
    def __call__(self, map):
        #resize = transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST)
        resize = transforms.Resize((256, 512))
        map = resize(map)
        
        #dmin, dmax= 0.0001, 80
        map = np.array(map, dtype=np.float32)
        #map = (map - map.min()) / (map.max() - map.min())
        #map = np.log(map.clip(dmin, dmax)/dmin)/np.log(dmax/dmin)
        assert (np.max(map) > 255), "Depth map might not be 16-bit!"
        scale = 1024/2048  # new_width/old_width
        ignore_depth = -1.0
        map = (map / 100) * scale
        map[map <= 0] = ignore_depth
        
        valid_mask = map > 0
        if np.any(valid_mask):
            #map[valid_mask] = (map[valid_mask] - 0.005) / (161.285 - 0.005)
            map[valid_mask] = (map[valid_mask] - map[valid_mask].min()) / (map[valid_mask].max() - map[valid_mask].min())
        
        map = torch.tensor(map, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (C, H, W)
        
        return map
    
class SynDepthTransform:
    def __call__(self, map):
        #map = np.array(map, dtype=np.float32)
        map = torch.tensor(map).unsqueeze(0).unsqueeze(0)
        map = torch.nn.functional.interpolate(map, size=(256, 512), mode='nearest')
        map = map.squeeze(0).squeeze(0).numpy()
        scale = 1024/1440  # new_width/old_width
        ignore_depth = -1.0
        epsilon = 1e-6
        map = map / 100
        map[map <= 0] = -1
        map = 0.22 * 2262 / (map + epsilon) * scale
        map[map <= 0] = ignore_depth
        map[map == 65535] = 0
        
        valid_mask = map > 0
        if np.any(valid_mask):
            #map[valid_mask] = normalize_image(map[valid_mask], min_depth=3.8920486, max_depth=178591.83)
            map[valid_mask] = (map[valid_mask] - map[valid_mask].min()) / (map[valid_mask].max() - map[valid_mask].min())
        
        map = torch.tensor(map, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (C, H, W)
        
        return map
    
    
class SynScapesDataset(Dataset):
    def __init__(self, root_dir, image_transform, seg_transform, depth_transform):
        self.root_dir = root_dir                    # SynScapes/train or test or val
        self.image_transform = image_transform
        self.seg_transform = seg_transform
        self.depth_transform = depth_transform
        self.rgb_paths = []
        self.seg_paths = []
        self.depth_paths = []
        
        rgb_dir = os.path.join(self.root_dir, 'rgb')
        seg_dir = os.path.join(self.root_dir, 'semantic')
        depth_dir = os.path.join(self.root_dir, 'depth_npy')
        
        for file_name in os.listdir(rgb_dir):
            self.rgb_paths.append(os.path.join(rgb_dir, file_name))
        for file_name in os.listdir(seg_dir):
            self.seg_paths.append(os.path.join(seg_dir, file_name))
        for file_name in os.listdir(depth_dir):
            self.depth_paths.append(os.path.join(depth_dir, file_name))
            
    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_paths[idx])
        seg_map = Image.open(self.seg_paths[idx])
        depth_map = np.load(self.depth_paths[idx])

        if self.image_transform:
            rgb_image = self.image_transform(rgb_image)
        if self.seg_transform:
            seg_map = self.seg_transform(seg_map)
        if self.depth_transform:
            depth_map = self.depth_transform(depth_map)

        return rgb_image, seg_map, depth_map
