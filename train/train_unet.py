import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import json
from datetime import datetime

# CONFIGURATION
MODEL_NAME = "UNet"
CONFIG = {
    "model_name": MODEL_NAME,
    "batch_size": 32,
    "epochs": 150,
    "lr": 1e-4,
    "input_size": (224, 224),
    "heatmap_sigma": 4,
    "loss_fn": "MSELoss", 
}

def get_loss_function(name):
    if name == "MSELoss":
        return nn.MSELoss()
    elif name == "L1Loss":
        return nn.L1Loss()
    elif name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = f"outputs/UNet/{MODEL_NAME}_{timestamp}"
os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

#UNET
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.middle = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        m = self.middle(p4)

        u4 = self.up4(m)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4)
        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return self.out_conv(u1)

#DATASET
class JHUDataset(Dataset):
    def __init__(self, images_dir, heatmaps_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        
        for img_name in os.listdir(images_dir):
            if not img_name.endswith(".jpg"):
                continue
            img_path = os.path.join(images_dir, img_name)
            heatmap_path = os.path.join(heatmaps_dir, img_name.replace(".jpg", ".npy"))
            if os.path.exists(heatmap_path):
                try:
                    h = np.load(heatmap_path)
                    if h.ndim != 2 or h.size == 0 or np.isnan(h).any() or np.max(h) < 1e-5:# Skip empty heatmaps
                        continue
                    self.samples.append((img_path, heatmap_path))
                except:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, heatmap_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        heatmap = np.load(heatmap_path)

        # Normalize heatmap
        heatmap_img = Image.fromarray((np.clip(heatmap, 0, None) * 255 / (heatmap.max() + 1e-8)).astype(np.uint8)).convert("L") # Convert to grayscale
        heatmap_img = heatmap_img.resize(CONFIG["input_size"], resample=Image.BILINEAR) # Resize heatmap to match input size

        if self.transform:
            image = self.transform(image)
            heatmap_img = self.transform(heatmap_img)

        return image, heatmap_img

## DATA AUGMENTATION TRANSFORMS
transform = T.Compose([
    T.Resize(CONFIG["input_size"]),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomHorizontalFlip(),
    T.RandomAffine(degrees=5, translate=(0.02, 0.02)),
    T.ToTensor()
])


train_dataset = JHUDataset("jhu_crowd/train/images", "jhu_crowd/train/heatmaps", transform)
val_dataset = JHUDataset("jhu_crowd/valid/images", "jhu_crowd/valid/heatmaps", transform)
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

#MODEL INITIALIZATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = get_loss_function(CONFIG["loss_fn"])
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

#TRAINING
train_losses, val_losses = [], []

for epoch in range(CONFIG["epochs"]):
    model.train()
    train_loss = 0
    for imgs, heatmaps in train_loader:
        imgs, heatmaps = imgs.to(device), heatmaps.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, heatmaps in val_loader:
            imgs, heatmaps = imgs.to(device), heatmaps.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, heatmaps)
            val_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

    print(f"[{epoch+1}/{CONFIG['epochs']}] Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f}")

    if epoch % 2 == 0: # Visualize predictions every 2 epochs for reviewing progress
        imgs_cpu = imgs.cpu()
        outputs_cpu = torch.sigmoid(outputs.cpu())
        heatmaps_cpu = heatmaps.cpu()
        for i in range(min(2, imgs_cpu.size(0))):
            img_np = imgs_cpu[i].permute(1, 2, 0).numpy()
            pred_np = outputs_cpu[i][0].numpy()
            gt_np = heatmaps_cpu[i][0].numpy()

            pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
            gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np)
            axes[0].set_title("Original")
            axes[1].imshow(img_np)
            axes[1].imshow(gt_np, cmap='jet', alpha=0.6)
            axes[1].set_title("Ground Truth")
            axes[2].imshow(img_np)
            axes[2].imshow(pred_np, cmap='jet', alpha=0.6)
            axes[2].set_title("Prediction")
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "visualizations", f"epoch_{epoch+1}_sample_{i+1}.png"))
            plt.close()

# SAVE MODEL AND CONFIGURATION
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=4)

# SAVE LOSS CURVES
def save_combined_plot(train_vals, val_vals, title, ylabel, filename):
    plt.figure()
    plt.plot(train_vals, label="Train")
    plt.plot(val_vals, label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

save_combined_plot(train_losses, val_losses, "Loss Curve", "Loss", "loss_curve.png")