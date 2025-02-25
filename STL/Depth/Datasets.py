import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
#import cv2
import numpy as np


class CityScapesDataset(Dataset):
    def __init__(self, root_dir, image_transform, depth_transform):
        self.root_dir = root_dir                    # CityScapes/train or test or val
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.rgb_paths = []
        self.depth_paths = []
        
        rgb_dir = os.path.join(self.root_dir, 'rgb')
        depth_dir = os.path.join(self.root_dir, 'depth')
        
        for file_name in os.listdir(rgb_dir):
            self.rgb_paths.append(os.path.join(rgb_dir, file_name))
            self.depth_paths.append(os.path.join(depth_dir, file_name))
            
    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_paths[idx])
        depth_map = Image.open(self.depth_paths[idx])

        if self.image_transform:
            rgb_image = self.image_transform(rgb_image)
        if self.depth_transform:
            depth_map = self.depth_transform(depth_map)

        return rgb_image, depth_map
    
    
class ImageTransform:
    def __call__(self, image):
        resize = transforms.Resize((512, 1024))
        image = resize(image)
        image = np.asarray(image, dtype=np.float32)
        #image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_AREA)
        image = image / 255.0
        
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        return image

      
class CityDepthTransform:
    def __call__(self, map):
        resize = transforms.Resize((512, 1024))
        map = resize(map)
        map = np.asarray(map, dtype=np.float32)
        #map = (map - map.min()) / (map.max() - map.min())
        dmin= 0.0001
        dmax= 80
        map = np.log(map.clip(dmin, dmax)/dmin)/np.log(dmax/dmin)
        
        map = torch.tensor(map, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (C, H, W)
        
        return map
    

class SynDepthTransform:
    def __call__(self, map):
        resize = transforms.Resize((512, 1024))
        map = resize(map)
        map = np.asarray(map, dtype=np.float32)
        #map = (map - map.min()) / (map.max() - map.min())
        dmin= 7
        dmax= 80
        map = np.log(map.clip(dmin, dmax)/dmin)/np.log(dmax/dmin)
        
        map = torch.tensor(map, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (C, H, W)
        
        return map


class SynScapesDataset(CityScapesDataset):
    def __init__(self, root_dir, image_transform, depth_transform):
        super().__init__(root_dir, image_transform, depth_transform)