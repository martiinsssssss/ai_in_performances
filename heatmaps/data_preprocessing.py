import os
import numpy as np
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURA TU SPLIT ---
split = 'test'  # 'train' o 'valid'
root = f'archive/{split}'
img_dir = os.path.join(root, 'images')
label_dir = os.path.join(root, 'labels')
heatmap_dir = os.path.join(root, 'heatmaps')
vis_dir = os.path.join(root, 'visualizations')
os.makedirs(heatmap_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# --- CONFIGURACIÓN HEATMAP ---
sigma = 10  # puedes ajustar esto según el tamaño medio de cabezas
debug_overlay = True  # muestra puntos blancos sobre el heatmap
normalize_heatmap = True  # normaliza para visualización

# --- GENERAR HEATMAPS ---
for img_file in os.listdir(img_dir):
    if not img_file.endswith('.jpg'):
        continue

    img_path = os.path.join(img_dir, img_file)
    img = Image.open(img_path).convert("RGB")
    width, height = img.size

    base_name = os.path.splitext(img_file)[0]
    label_file = base_name + ".txt"
    label_path = os.path.join(label_dir, label_file)

    if not os.path.exists(label_path):
        print(f"⚠️ No label found for {img_file}")
        continue

    # Leer puntos (YOLO: class_id x_center_rel y_center_rel width_rel height_rel)
    points = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x_rel = float(parts[1])
                    y_rel = float(parts[2])
                    x = x_rel * width
                    y = y_rel * height
                    points.append((x, y))
                except Exception as e:
                    print(f"❌ Error parsing {label_file}: {e}")

    # Crear heatmap y aplicar puntos
    heatmap = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        xi = min(max(int(round(x)), 0), width - 1)
        yi = min(max(int(round(y)), 0), height - 1)
        heatmap[yi, xi] = 1.0

    # Aplicar blur gaussiano
    heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=sigma)

    # Opcional: conservar conteo total
    if np.sum(heatmap) > 0:
        heatmap *= len(points) / (np.sum(heatmap) + 1e-8)

    # Guardar heatmap .npy (sin normalizar para entrenamiento)
    out_name = img_file.replace('.jpg', '.npy')
    np.save(os.path.join(heatmap_dir, out_name), heatmap)

    # 🔍 VISUALIZACIÓN
    plt.figure(figsize=(10, 5))

    # Imagen original
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Imagen original")
    plt.axis("off")

    # Imagen + heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(img)

    # Normaliza a [0,1] para visualizar
    heatmap_disp = heatmap / (heatmap.max() + 1e-8) if normalize_heatmap else heatmap
    plt.imshow(heatmap_disp, cmap='jet', alpha=0.6)

    # Opcional: overlay puntos
    if debug_overlay:
        for px, py in points:
            plt.scatter(px, py, s=8, c='white', edgecolors='black', linewidths=0.3)

    plt.title("Heatmap superpuesto")
    plt.axis("off")

    plt.tight_layout()
    vis_path = os.path.join(vis_dir, img_file.replace('.jpg', '.png'))
    plt.savefig(vis_path, dpi=100)
    plt.close()

print(f"\n✅ Heatmaps guardados en: {heatmap_dir}")
print(f"🖼️ Visualizaciones guardadas en: {vis_dir}")