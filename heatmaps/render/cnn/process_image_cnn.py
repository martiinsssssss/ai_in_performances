import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import cv2

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
        self.input_size = (224, 224) 

    def forward(self, x):
        return self.decoder(self.encoder(x))

MODEL_PATH = 'CNN/MSE2/model_mse.pth'
INPUT_IMAGE_PATH = 'stand.jpg'    
OUTPUT_GRAYSCALE_HEATMAP_DIR = 'final_heightmaps_for_blender'
OUTPUT_GRAYSCALE_HEATMAP_FILENAME = 'heightmap_stand_mse_cnn.png'

os.makedirs(OUTPUT_GRAYSCALE_HEATMAP_DIR, exist_ok=True)
OUTPUT_GRAYSCALE_HEATMAP_FULL_PATH = os.path.join(OUTPUT_GRAYSCALE_HEATMAP_DIR, OUTPUT_GRAYSCALE_HEATMAP_FILENAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    model = SimpleCNN()
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(device)
    print(f"Model 'SimpleCNN' loaded successfully from {MODEL_PATH}")

except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'. Please verify the path.")
    exit()
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Please ensure the SimpleCNN class definition matches your trained model.")
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

print(f'Mapa de alturas en escala de grises para Blender (usando SimpleCNN) guardado en: {OUTPUT_GRAYSCALE_HEATMAP_FULL_PATH}')