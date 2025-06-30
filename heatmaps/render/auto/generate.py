import torch
from model_unet import UNet
from utils import load_image, save_heatmap, load_model
import subprocess
import os

IMAGE_PATH = 'stand.jpg'
CHECKPOINT_PATH = 'model_unet.pth'
OUTPUT_PATH = 'heatmaps/heightmap.png'
BLENDER_SCENE = 'blender/montaña_render.blend'
BLENDER_SCRIPT = 'blender/blender_update_displacement.py'
BLENDER_EXECUTABLE = '/Applications/Blender.app/Contents/MacOS/Blender'  # macOS

model = UNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model(model, CHECKPOINT_PATH, device)

input_tensor = load_image(IMAGE_PATH, size=(224, 224)).to(device)
with torch.no_grad():
    output = model(input_tensor)

save_heatmap(output, OUTPUT_PATH, original_size=(1024, 1024))
print(f"✅ Heatmap guardado en: {OUTPUT_PATH}")

subprocess.run([
    BLENDER_EXECUTABLE,
    BLENDER_SCENE,
    '--python', BLENDER_SCRIPT
])