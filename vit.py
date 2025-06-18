import os, json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm

# CONFIGURATION
CONFIG = {
    "model_name": "TransUNet",
    "backbone": "vit_base_patch16_224",
    "batch_size": 32,
    "epochs": 150,
    "lr": 1e-4,
    "input_size": (224, 224),
    "loss_fn": "L1Loss", 
}

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = f"outputs/ViT/{CONFIG['model_name']}_{timestamp}"
os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

# DATASET
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
        image = Image.open(img_path).convert("RGB")
        heatmap = np.load(hmap_path)

        #Normalize heatmap
        heatmap = np.clip(heatmap, 0, None)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).convert("L") # Convert to grayscale
        heatmap_img = heatmap_img.resize(CONFIG["input_size"], resample=Image.BILINEAR) # Resize heatmap to match input size

        if self.transform:
            image = self.transform(image)
        heatmap_tensor = T.ToTensor()(heatmap_img)  # No Normalization for heatmap

        return image, heatmap_tensor

#DATA AUGMENTATION TRANSFORMS
transform = T.Compose([
    T.Resize(CONFIG["input_size"]),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

train_dataset = JHUDataset("jhu_crowd/train/images", "jhu_crowd/train/heatmaps", transform)
val_dataset = JHUDataset("jhu_crowd/valid/images", "jhu_crowd/valid/heatmaps", transform)
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

#MODEL
class TransUNet(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, 2, stride=2), nn.ReLU(),    # 14 -> 28
            nn.ConvTranspose2d(512, 256, 2, stride=2), nn.ReLU(),    # 28 -> 56
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),    # 56 -> 112
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),     # 112 -> 224
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.encoder.patch_embed(x)
        cls_token = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.encoder.pos_embed
        x = self.encoder.pos_drop(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        x = x[:, 1:, :]
        H, W = CONFIG["input_size"]
        x = x.view(B, H // 16, W // 16, -1).permute(0, 3, 1, 2)  # (B, C=768, H/16, W/16)
        return self.decoder(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransUNet(CONFIG["backbone"]).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

#TRAINING
train_losses, val_losses = [], []
for epoch in range(CONFIG["epochs"]):
    model.train()
    train_loss = 0
    for imgs, hmaps in train_loader:
        imgs, hmaps = imgs.to(device), hmaps.to(device)
        preds = model(imgs)
        loss = criterion(preds, hmaps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, hmaps in val_loader:
            imgs, hmaps = imgs.to(device), hmaps.to(device)
            preds = model(imgs)
            val_loss += criterion(preds, hmaps).item()

        # Visualize predictions every 2 epochs for reviewing progress
        if epoch % 2 == 0:
            imgs_cpu = imgs.cpu()
            preds_cpu = torch.sigmoid(preds.cpu())
            hmaps_cpu = hmaps.cpu()
            for i in range(min(2, imgs_cpu.size(0))):
                img_np = imgs_cpu[i].permute(1, 2, 0).numpy()
                pred_np = preds_cpu[i][0].numpy()
                gt_np = hmaps_cpu[i][0].numpy()
                pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
                gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)

                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(img_np); ax[0].set_title("Original")
                ax[1].imshow(img_np); ax[1].imshow(gt_np, cmap='jet', alpha=0.6); ax[1].set_title("Ground Truth")
                ax[2].imshow(img_np); ax[2].imshow(pred_np, cmap='jet', alpha=0.6); ax[2].set_title("Prediction")
                for a in ax: a.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "visualizations", f"epoch_{epoch+1}_sample_{i+1}.png"))
                plt.close()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    scheduler.step(val_losses[-1])

    print(f"[{epoch+1}/{CONFIG['epochs']}] Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f}")

#SAVE MODEL AND CONFIGURATION
torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=4)

# PLOT LOSS CURVES
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