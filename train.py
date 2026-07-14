import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import CameraDataset
from model import CameraAI

# --- 1. DETECT HARDWARE ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 HARDWARE ACCELERATION: NVIDIA CUDA GPU DETECTED!")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    USE_THERMAL_LIMITS = False 
else:
    device = torch.device("cpu")
    print("💻 RUNNING ON CPU: Applying safety limits...")
    USE_THERMAL_LIMITS = True
    os.environ["OMP_NUM_THREADS"] = "4"
    torch.set_num_threads(4)

# --- 2. PARSE ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument("target_epochs", type=int, help="The absolute epoch number to reach")
parser.add_argument("save_interval", type=int, help="How often to save")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

# Hyperparameters
MAX_SAMPLE_SIZE = 1000 
BATCH_SIZE = 16

# --- 3. LOAD DATASET ---
main_folder = "AI_Camera_Dataset"
if not os.path.exists(main_folder):
    print(f"Error: Could not find '{main_folder}' directory. Please copy your dataset folder over!")
    sys.exit(1)

dataset = CameraDataset(main_folder)
dataset_size = len(dataset)

# --- 4. DATA SAMPLER ---
if dataset_size > MAX_SAMPLE_SIZE:
    print(f"🔥 BIG DATA MODE: Randomly sampling {MAX_SAMPLE_SIZE} out of {dataset_size} images.")
    indices = list(range(dataset_size))
    random.shuffle(indices)
    subset_indices = indices[:MAX_SAMPLE_SIZE]
    sampler = SubsetRandomSampler(subset_indices)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
else:
    print(f"Dataset size: {dataset_size}. Training on all images.")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# --- 5. INITIALIZE MODEL & OPTIMIZER ---
model = CameraAI().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

start_epoch = 0
CHECKPOINT_FILE = "camera_ai_checkpoint.pth"

# --- 6. RESUME CHECK ---
if args.resume and os.path.exists(CHECKPOINT_FILE):
    print("Resuming from checkpoint...")
    try:
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Successfully resumed! Starting at epoch {start_epoch}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        sys.exit(1)
else:
    print("Starting fresh training...")

print(f"Goal: {args.target_epochs} epochs | Saving a checkpoint every {args.save_interval} epochs.")

if start_epoch >= args.target_epochs:
    print("TRAINING FULLY COMPLETE & FINAL WEIGHTS SAVED")
    sys.exit(0)

# --- 7. MAIN TRAINING SPRINT ---
model.train()
for epoch in range(start_epoch + 1, args.target_epochs + 1):
    running_loss = 0.0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Only apply micro-naps if we are training on CPU to prevent overheat
        if USE_THERMAL_LIMITS:
            import time
            time.sleep(1.0)
        
    print(f"Epoch: {epoch:02d} / {args.target_epochs} | Total Epoch Loss: {running_loss:.4f}")
    
    if epoch % args.save_interval == 0 or epoch == args.target_epochs:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
        }, CHECKPOINT_FILE)
        print(f"---> Interval Checkpoint saved at Epoch {epoch}")

print("TRAINING FULLY COMPLETE & FINAL WEIGHTS SAVED")
