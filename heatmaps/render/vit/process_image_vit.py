import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2

class ViT(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained)
        self.input_size = (224, 224)

        encoder_output_channels = self.encoder.embed_dim
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_output_channels, 512, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
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
        
        patch_height = self.encoder.patch_embed.patch_size[0]
        patch_width = self.encoder.patch_embed.patch_size[1]
        
        H_patch = self.input_size[0] // patch_height
        W_patch = self.input_size[1] // patch_width
        
        x = x.view(B, H_patch, W_patch, -1).permute(0, 3, 1, 2)
        
        return self.decoder(x)

VIT_BACKBONE_NAME = "vit_base_patch16_224" 
MODEL_PATH = 'ViT/L12/model_l12.pth'
INPUT_IMAGE_PATH = 'stand.jpg'    
OUTPUT_GRAYSCALE_HEATMAP_DIR = 'final_heightmaps_for_blender'
OUTPUT_GRAYSCALE_HEATMAP_FILENAME = 'heightmap_stand_l1_vit.png'

os.makedirs(OUTPUT_GRAYSCALE_HEATMAP_DIR, exist_ok=True)
OUTPUT_GRAYSCALE_HEATMAP_FULL_PATH = os.path.join(OUTPUT_GRAYSCALE_HEATMAP_DIR, OUTPUT_GRAYSCALE_HEATMAP_FILENAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    model = ViT(model_name=VIT_BACKBONE_NAME, pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(device)
    print(f"Model '{VIT_BACKBONE_NAME}' loaded successfully from {MODEL_PATH}")

except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'. Please verify the path.")
    exit()
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Please ensure the ViT class definition matches your trained model.")
    print("Also ensure 'VIT_BACKBONE_NAME' is correct.")
    exit()

try:
    input_image_pil = Image.open(INPUT_IMAGE_PATH).convert('RGB')
    original_width, original_height = input_image_pil.size
except FileNotFoundError:
    print(f"Error: Input image not found at '{INPUT_IMAGE_PATH}'. Please verify the path.")
    exit()
except Exception as e:
    print(f"Error opening input image: {e}")
    exit()

preprocess = transforms.Compose([
    transforms.Resize(model.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image_pil).unsqueeze(0).to(device)

print(f"Input image preprocessed. Tensor shape: {input_tensor.shape}")

with torch.no_grad():
    raw_output = model(input_tensor)
    heatmap_tensor = torch.sigmoid(raw_output)

heatmap_np = heatmap_tensor.squeeze().cpu().numpy()

heatmap_np[heatmap_np < 0] = 0

print(f"Raw heatmap dimensions: {heatmap_np.shape}")
print(f"Min/max values of raw heatmap: {np.min(heatmap_np):.4f} / {np.max(heatmap_np):.4f}")

heatmap_norm = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np) + 1e-8)
heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

final_heightmap_for_blender = cv2.resize(
    heatmap_uint8,
    (original_width, original_height),
    interpolation=cv2.INTER_LINEAR
)

Image.fromarray(final_heightmap_for_blender, mode='L').save(OUTPUT_GRAYSCALE_HEATMAP_FULL_PATH)

print(f'Mapa de alturas en escala de grises guardado para Blender en: {OUTPUT_GRAYSCALE_HEATMAP_FULL_PATH}')