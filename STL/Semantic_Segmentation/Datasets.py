import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
#import cv2
import numpy as np


class CityScapesDataset(Dataset):
    def __init__(self, root_dir, image_transform, seg_transform):
        self.root_dir = root_dir                    # CityScapes/train or CityScapes/val
        self.image_transform = image_transform
        self.seg_transform = seg_transform
        self.rgb_paths = []
        self.seg_paths = []
        
        rgb_dir = os.path.join(self.root_dir, 'rgb')
        seg_dir = os.path.join(self.root_dir, 'semantic')
        
        for file_name in os.listdir(rgb_dir):
            self.rgb_paths.append(os.path.join(rgb_dir, file_name))
            self.seg_paths.append(os.path.join(seg_dir, file_name))
            
    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_paths[idx])
        depth_map = Image.open(self.seg_paths[idx])

        if self.image_transform:
            rgb_image = self.image_transform(rgb_image)
        if self.seg_transform:
            depth_map = self.seg_transform(depth_map)

        return rgb_image, depth_map
    
    
class ImageTransform:
    def __call__(self, image):
        resize = transforms.Resize((512, 1024))
        image = resize(image)
        
        image = np.asarray(image)
        #image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_AREA)
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        return image

      
class SegTransform:
    def __call__(self, map):
        resize = transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST) # Nearest neighbour interpolation
        map = resize(map)
        #map = cv2.resize(map, (1024, 512), interpolation=cv2.INTER_AREA)
        
        map = np.asarray(map)
        map = torch.tensor(map, dtype=torch.long).unsqueeze(0)  # Add channel dimension (C, H, W)
        
        return map


class SynScapesDataset(CityScapesDataset):
    def __init__(self, root_dir, image_transform, depth_transform):
        super().__init__(root_dir, image_transform, depth_transform)
