import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CameraDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.images_dir = os.path.join(main_dir, "images")
        self.labels_dir = os.path.join(main_dir, "labels")
        
        # Get list of all JSON files
        self.label_files = [f for f in os.listdir(self.labels_dir) if f.endswith('.json')]
        self.label_files.sort()
        
        # Image pre-processing (Resize to 144x256 to match the model input size)
        self.transform = transforms.Compose([
            transforms.Resize((144, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        # 1. Load label
        json_filename = self.label_files[idx]
        json_path = os.path.join(self.labels_dir, json_filename)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Extract features (adjust keys to match your blender output exactly)
        loc = data.get("location", [0.0, 0.0, 0.0])
        pos = data.get("position", [0.0, 0.0, 0.0]) or data.get("rotation", [0.0, 0.0, 0.0])
        focal = [data.get("focal_length", 50.0)]
        
        # Combine into a single target list of 7 floats
        target_list = loc + pos + focal
        target_tensor = torch.tensor(target_list, dtype=torch.float32)
        
        # 2. Load matching image
        image_filename = json_filename.replace(".json", ".png")
        image_path = os.path.join(self.images_dir, image_filename)
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        
        return image_tensor, target_tensor
