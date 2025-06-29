import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from PIL import Image
import imageio.v3 as iio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn

# 1. GLOBAL CONFIGURATION
INPUT_IMAGE_PATH = 'stand.jpg'
MODEL_INPUT_SIZE = (224, 224)
SCALE_FACTOR_HEIGHT = 20.0
RESOLUTION_SCALE_MODEL = 0.7
TEXTURE_FOLDER = 'render_out/textures/cliff_side_2k.blend/textures'
TEXTURE_FILENAMES_DICT = {
    'base_color': 'cliff_side_diff_2k.jpg',
    'normal': 'cliff_side_nor_gl_2k.exr',
    'roughness': 'cliff_side_rough_2k.exr',
    'displacement': 'cliff_side_disp_2k.png',
    'ao': '',
}

model_info = {
    'unet_mse': 'UNet/MSE/model_mse.pth',
    'unet_l1': 'UNet/L1/model_l1.pth',
    'unet_bce': 'UNet/BCE/model_bcelogits.pth',
}

# 2. UNet Definition
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

# 3. Inference Transformation
inference_transform = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
])

# 4. Function to Generate Heatmap Array AND Save PNG
def generate_heatmap_array_and_save_png_from_model(model_path, input_image_path, model_input_size,
                                                  transform_pipeline, output_png_dir, output_png_filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = UNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        print(f"Model '{model.__class__.__name__}' loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model not found at '{model_path}'. Please verify the path.")
        return None, None, None
    except Exception as e:
        print(f"Error loading the model or its state_dict: {e}")
        print("Ensure that the UNet class definition is correct and matches the saved model.")
        return None, None, None

    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at '{input_image_path}'. Please verify the path.")
        return None, None, None

    original_image_pil = Image.open(input_image_path).convert('RGB')
    original_width, original_height = original_image_pil.size

    input_tensor = transform_pipeline(original_image_pil).unsqueeze(0).to(device)

    print(f"Input image preprocessed. Tensor shape: {input_tensor.shape}")

    with torch.no_grad():
        raw_output_tensor = model(input_tensor)

    heatmap_tensor = torch.sigmoid(raw_output_tensor)
    heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
    heatmap_np[heatmap_np < 0] = 0

    print(f"Raw density map generated. Dimensions: {heatmap_np.shape}")
    print(f"Min/max values of the density map: {np.min(heatmap_np):.4f} / {np.max(heatmap_np):.4f}")

    normalized_heatmap_for_save = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np) + 1e-8)
    normalized_heatmap_for_save = (normalized_heatmap_for_save * 255).astype(np.uint8)

    final_heatmap_image_for_save = cv2.resize(
        normalized_heatmap_for_save,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR
    )

    original_image_cv = cv2.imread(input_image_path)
    if original_image_cv is None:
        print(f"Error: Could not load original image with OpenCV at '{input_image_path}'.")
        return None, None, None

    heatmap_color = cv2.applyColorMap(final_heatmap_image_for_save, cv2.COLORMAP_JET)
    heatmap_overlay = cv2.addWeighted(original_image_cv, 0.6, heatmap_color, 0.4, 0)

    os.makedirs(output_png_dir, exist_ok=True)
    output_png_path = os.path.join(output_png_dir, output_png_filename)
    iio.imwrite(output_png_path, heatmap_overlay)
    print(f"Heatmap overlaid on original image and saved to: {output_png_path}")

    return heatmap_np, original_width, original_height

# 5. Function to Generate 3D Model and Rotating GIF
def generate_3d_model_and_gif_from_heatmap(heatmap_array, output_base_name, scale_factor, resolution_scale,
                                          texture_paths_dict, save_visualization=True, vis_output_dir=None, gif_filename='3d_rotation.gif'):
    output_dir = os.path.dirname(output_base_name) or '.'
    os.makedirs(output_dir, exist_ok=True)

    if resolution_scale != 1.0:
        new_h = int(heatmap_array.shape[0] * resolution_scale)
        new_w = int(heatmap_array.shape[1] * resolution_scale)
        heatmap_processed = np.array(Image.fromarray((heatmap_array * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)) / 255.0
        print(f"Heatmap array resized for model generation to: {heatmap_processed.shape}")
    else:
        heatmap_processed = heatmap_array

    if np.max(heatmap_processed) > 0:
        heatmap_normalized = heatmap_processed / np.max(heatmap_processed)
    else:
        heatmap_normalized = np.zeros_like(heatmap_processed, dtype=np.float32)

    height, width = heatmap_normalized.shape

    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)

    scale_xy = 1.0 / max(width, height) * 50
    X_scaled = X * scale_xy - (width * scale_xy / 2)
    Y_scaled = Y * scale_xy - (height * scale_xy / 2)

    Z = heatmap_normalized * scale_factor

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("3D Mountain Generated from Heatmap")
    ax.set_axis_off()

    frames = []
    frame_to_save_separately = None

    for angle in range(0, 360, 10):
        ax.clear()
        ax.plot_surface(X_scaled, Y_scaled, Z, cmap='terrain', rstride=1, cstride=1, antialiased=False)
        ax.view_init(elev=30, azim=angle)
        plt.tight_layout()

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        frames.append(image)

        if angle == 90:
            frame_to_save_separately = image

    plt.close(fig)

    if save_visualization and vis_output_dir:
        os.makedirs(vis_output_dir, exist_ok=True)
        gif_path = os.path.join(vis_output_dir, gif_filename)
        iio.imwrite(gif_path, frames, duration=0.5,loop=0)
        print(f"3D rotating visualization GIF saved to: {gif_path}")

        if frame_to_save_separately is not None:
            frame_output_filename = os.path.splitext(gif_filename)[0] + "_frame_0.png"
            frame_output_path = os.path.join(vis_output_dir, frame_output_filename)
            iio.imwrite(frame_output_path, frame_to_save_separately)
            print(f"Selected frame (angle 0) saved to: {frame_output_path}")

if __name__ == "__main__":
    needs_dummy_data = False
    for _, path in model_info.items():
        if not os.path.exists(path):
            needs_dummy_data = True
            break
    if not os.path.exists(INPUT_IMAGE_PATH):
        needs_dummy_data = True

    if needs_dummy_data:
        print("\n--- CREATING DUMMY MODELS AND IMAGE FOR TESTING ---")
        print("!!! WARNING: Please replace paths in 'model_info' with your actual trained model paths.")
        print(f"And '{INPUT_IMAGE_PATH}' with the path to your actual input image.")

        for model_key, model_path_dummy in model_info.items():
            os.makedirs(os.path.dirname(model_path_dummy) or '.', exist_ok=True)
            if not os.path.exists(model_path_dummy):
                dummy_model = UNet()
                torch.save(dummy_model.state_dict(), model_path_dummy)
                print(f"Dummy UNet model '{model_key}' created at '{model_path_dummy}'.")

        os.makedirs(os.path.dirname(INPUT_IMAGE_PATH) or '.', exist_ok=True)
        if not os.path.exists(INPUT_IMAGE_PATH):
            dummy_img_res = (640, 480)
            dummy_img = np.zeros((dummy_img_res[1], dummy_img_res[0], 3), dtype=np.uint8)
            cv2.circle(dummy_img, (dummy_img_res[0] // 2, dummy_img_res[1] // 2),
                       min(dummy_img_res) // 4, (0, 0, 255), -1)
            iio.imwrite(INPUT_IMAGE_PATH, dummy_img)
            print(f"Dummy input image created at '{INPUT_IMAGE_PATH}'.")

        os.makedirs(TEXTURE_FOLDER, exist_ok=True)
        for tex_name, tex_filename in TEXTURE_FILENAMES_DICT.items():
            if tex_filename:
                full_tex_path = os.path.join(TEXTURE_FOLDER, tex_filename)
                if not os.path.exists(full_tex_path):
                    dummy_texture = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
                    if 'normal' in tex_name:
                        dummy_texture[:, :, 0] = 128
                        dummy_texture[:, :, 1] = 128
                        dummy_texture[:, :, 2] = 255
                    elif 'base_color' in tex_name:
                        dummy_texture[:, :, 0] = 150
                        dummy_texture[:, :, 1] = 100
                        dummy_texture[:, :, 2] = 50
                    iio.imwrite(full_tex_path, dummy_texture)
                    print(f"Dummy texture '{tex_filename}' created for testing.")
        print("-" * 50 + "\n")

    for model_name, model_path in model_info.items():
        print(f"\n--- Processing model: {model_name} ---")

        output_dir_model = f"render_out/{model_name}"
        heatmap_png_dir = os.path.join(output_dir_model, "heatmaps_png")
        heatmap_png_name = f"heatmap_{model_name}.png"
        model_obj_base = os.path.join(output_dir_model, f"mountain_from_heatmap_{model_name}")
        vis_dir = os.path.join(output_dir_model, "3d_visualizations")
        gif_name = f"3d_rotation_{model_name}.gif"

        heatmap_array, orig_w, orig_h = generate_heatmap_array_and_save_png_from_model(
            model_path=model_path,
            input_image_path=INPUT_IMAGE_PATH,
            model_input_size=MODEL_INPUT_SIZE,
            transform_pipeline=inference_transform,
            output_png_dir=heatmap_png_dir,
            output_png_filename=heatmap_png_name
        )

        if heatmap_array is not None:
            print(f"Generando modelo 3D y GIF para {model_name}...")
            generate_3d_model_and_gif_from_heatmap(
                heatmap_array=heatmap_array,
                output_base_name=model_obj_base,
                scale_factor=SCALE_FACTOR_HEIGHT,
                resolution_scale=RESOLUTION_SCALE_MODEL,
                texture_paths_dict=TEXTURE_FILENAMES_DICT,
                save_visualization=True,
                vis_output_dir=vis_dir,
                gif_filename=gif_name
            )
        else:
            print(f"Heatmap generation failed for {model_name}. Skipping 3D generation.")

    print("\nProcess completed for all models.")