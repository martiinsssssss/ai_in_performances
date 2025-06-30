import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

def load_image(path, size=(224, 224)):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def save_heatmap(tensor, path, original_size=(1024, 1024)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    heatmap = torch.sigmoid(tensor.squeeze()).cpu().detach().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    image = Image.fromarray(heatmap, mode='L').resize(original_size)
    image.save(path)

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)