import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

# CONFIGURATION
MODEL_NAME = "SimpleCNN"
CONFIG = {
    "model_name": MODEL_NAME,
    "dataset_name": "JHU-Crowd",
    "batch_size": 32,
    "epochs": 150,
    "lr": 1e-4,
    "input_size": (224, 224),
    "heatmap_sigma": 4,
    "loss_fn": "MSELoss",  
}

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = f"outputs/CNN/{MODEL_NAME}_{timestamp}"
os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

#CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

#DATASET
class JHUDataset(Dataset):
    def __init__(self, images_dir, heatmaps_dir, transform=None):
        self.samples = []
        self.transform = transform
        for fname in os.listdir(images_dir):
            if not fname.endswith(".jpg"): continue
            hpath = os.path.join(heatmaps_dir, fname.replace(".jpg", ".npy"))
            if os.path.exists(hpath):
                try:
                    h = np.load(hpath)
                    if h.ndim != 2 or h.size == 0 or np.max(h) < 1e-5: continue # Skip empty heatmaps
                    self.samples.append((os.path.join(images_dir, fname), hpath)) 
                except: continue
                

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, hmap_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB") 
        hmap = np.load(hmap_path)
        
        # Normalize heatmap 
        hmap_img = Image.fromarray((np.clip(hmap, 0, None) * 255 / (hmap.max() + 1e-8)).astype(np.uint8)).convert("L") # Convert to grayscale
        hmap_img = hmap_img.resize(CONFIG["input_size"], resample=Image.BILINEAR) # Resize heatmap to match input size
        if self.transform:
            img = self.transform(img)
            hmap_img = self.transform(hmap_img)
        return img, hmap_img

# DATA AUGMENTATION TRANSFORMS
transform = T.Compose([
    T.Resize(CONFIG["input_size"]),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

train_dataset = JHUDataset("jhu_crowd/train/images", "jhu_crowd/train/heatmaps", transform)
val_dataset   = JHUDataset("jhu_crowd/valid/images", "jhu_crowd/valid/heatmaps", transform)
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

# MODEL INITIALIZATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

if CONFIG["loss_fn"] == "MSELoss":
    criterion = nn.MSELoss()
elif CONFIG["loss_fn"] == "L1Loss":
    criterion = nn.L1Loss()
elif CONFIG["loss_fn"] == "BCEWithLogitsLoss":
    criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

# TRAINING
train_losses, val_losses = [], []

for epoch in range(CONFIG["epochs"]):
    model.train()
    total_train_loss = 0
    for imgs, hmaps in train_loader:
        imgs, hmaps = imgs.to(device), hmaps.to(device)
        preds = model(imgs)
        loss = criterion(preds, hmaps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for imgs, hmaps in val_loader:
            imgs, hmaps = imgs.to(device), hmaps.to(device)
            preds = model(imgs)
            total_val_loss += criterion(preds, hmaps).item()

        if epoch % 2 == 0: # Visualize predictions every 2 epochs for reviewing progress
            imgs_cpu = imgs.cpu()
            preds_cpu = torch.sigmoid(preds.cpu())
            hmaps_cpu = hmaps.cpu()
            for i in range(min(2, imgs_cpu.size(0))):
                img_np = imgs_cpu[i].permute(1, 2, 0).numpy()
                pred_np = preds_cpu[i][0].numpy()
                gt_np = hmaps_cpu[i][0].numpy()
                pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
                gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img_np); axes[0].set_title("Original")
                axes[1].imshow(img_np); axes[1].imshow(gt_np, cmap='jet', alpha=0.6); axes[1].set_title("Ground Truth")
                axes[2].imshow(img_np); axes[2].imshow(pred_np, cmap='jet', alpha=0.6); axes[2].set_title("Prediction")
                for ax in axes: ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "visualizations", f"epoch_{epoch+1}_sample_{i+1}.png"))
                plt.close()

    train_losses.append(total_train_loss / len(train_loader))
    val_losses.append(total_val_loss / len(val_loader))
    print(f"[{epoch+1}/{CONFIG['epochs']}] Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f}")

# SAVE MODEL AND CONGIGURATION
torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=4)

#PLOT LOSS CURVES
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
plt.close()