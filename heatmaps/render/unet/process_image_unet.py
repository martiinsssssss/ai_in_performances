import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import torch.nn as nn
import cv2

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

MODEL_PATH = 'UNet/BCE/model_bcelogits.pth'
INPUT_IMAGE_PATH = 'seated.jpg'
OUTPUT_GRAYSCALE_HEATMAP_DIR = 'final_heightmaps_for_blender'
OUTPUT_GRAYSCALE_HEATMAP_FILENAME = 'heightmap_seated_bce_unet.png'

os.makedirs(OUTPUT_GRAYSCALE_HEATMAP_DIR, exist_ok=True)
OUTPUT_GRAYSCALE_HEATMAP_FULL_PATH = os.path.join(OUTPUT_GRAYSCALE_HEATMAP_DIR, OUTPUT_GRAYSCALE_HEATMAP_FILENAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    model = UNet()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"Model 'UNet' loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

try:
    input_image_pil = Image.open(INPUT_IMAGE_PATH).convert('RGB')
    original_width, original_height = input_image_pil.size
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

input_tensor = preprocess(input_image_pil).unsqueeze(0).to(device)
print(f"Input image preprocessed. Shape: {input_tensor.shape}")

with torch.no_grad():
    output = model(input_tensor)
    heatmap_tensor = torch.sigmoid(output)

heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
heatmap_np = np.clip(heatmap_np, 0, 1)

print(f"Heatmap shape: {heatmap_np.shape}")
print(f"Min/Max heatmap: {np.min(heatmap_np):.4f}/{np.max(heatmap_np):.4f}")

heatmap_uint8 = ((heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8) * 255).astype(np.uint8)

heatmap_resized = cv2.resize(
    heatmap_uint8,
    (original_width, original_height),
    interpolation=cv2.INTER_LINEAR
)

Image.fromarray(heatmap_resized, mode='L').save(OUTPUT_GRAYSCALE_HEATMAP_FULL_PATH)
print(f"Heatmap saved to: {OUTPUT_GRAYSCALE_HEATMAP_FULL_PATH}")