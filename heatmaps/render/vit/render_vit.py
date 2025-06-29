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
import torch.nn.functional as F
import timm

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
    'vit_mse': 'ViT/MSE2/model_mse2.pth',
    'vit_l1': 'ViT/L12/model_l12.pth',
    'vit_bce': 'ViT/BCE/model_bcelogits.pth',
}

# 2. ViT MODEL DEFINITION
class ViT(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True, input_size=(224, 224)):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained)
        self.input_size = input_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2), nn.ReLU(),
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
        
        H_feat, W_feat = self.input_size[0] // self.encoder.patch_embed.patch_size[0], \
                         self.input_size[1] // self.encoder.patch_embed.patch_size[1]
        x = x.view(B, H_feat, W_feat, -1).permute(0, 3, 1, 2)
        
        return self.decoder(x)

# 3. Inference Transformation
inference_transform = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Function to Generate Heatmap Array AND Save PNG
def generate_heatmap_array_and_save_png_from_model(model_path, input_image_path, model_input_size,
                                                  transform_pipeline,
                                                  output_overlay_png_dir, output_overlay_png_filename,
                                                  output_pure_heatmap_png_dir, output_pure_heatmap_png_filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = ViT(input_size=model_input_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        print(f"Model '{model.__class__.__name__}' loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model not found at '{model_path}'. Please verify the path.")
        return None, None, None
    except Exception as e:
        print(f"Error loading the model or its state_dict: {e}")
        print("Ensure that the ViT class definition is correct and matches the saved model.")
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

    inverted_heatmap_for_save = 255 - normalized_heatmap_for_save

    final_heatmap_image_for_save = cv2.resize(
        inverted_heatmap_for_save,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR
    )

    os.makedirs(output_pure_heatmap_png_dir, exist_ok=True)
    pure_heatmap_png_path = os.path.join(output_pure_heatmap_png_dir, output_pure_heatmap_png_filename)
    Image.fromarray(final_heatmap_image_for_save, mode='L').save(pure_heatmap_png_path)
    print(f"Pure grayscale heatmap (inverted colors) saved to: {pure_heatmap_png_path}")

    original_image_cv = cv2.imread(input_image_path)
    if original_image_cv is None:
        print(f"Error: Could not load original image with OpenCV at '{input_image_path}'.")
        return None, None, None

    heatmap_color = cv2.applyColorMap(final_heatmap_image_for_save, cv2.COLORMAP_JET)
    heatmap_overlay = cv2.addWeighted(original_image_cv, 0.6, heatmap_color, 0.4, 0)

    os.makedirs(output_overlay_png_dir, exist_ok=True)
    output_overlay_png_path = os.path.join(output_overlay_png_dir, output_overlay_png_filename)
    iio.imwrite(output_overlay_png_path, heatmap_overlay)
    print(f"Heatmap (inverted colors) overlaid on original image and saved to: {output_overlay_png_path}")

    return heatmap_np, original_width, original_height

# 5. Function to generate 3D model and save static image
def generate_3d_model_and_save_static_image_from_heatmap(heatmap_array, output_base_name, scale_factor, resolution_scale,
                                                        texture_paths_dict, save_visualization=True, vis_output_dir=None, static_image_filename='3d_static_view.png'):
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

    x = np.linspace(width - 1, 0, width)
    y = np.linspace(0, height - 1, height)

    X, Y = np.meshgrid(x, y)

    scale_xy = 1.0 / max(width, height) * 50
    X_scaled = X * scale_xy - (width * scale_xy / 2)
    Y_scaled = Y * scale_xy - (height * scale_xy / 2)

    Z = heatmap_normalized * scale_factor

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("3D Mountain Generated from Heatmap")
    
    ax.grid(True, linestyle='-', linewidth=0.8, color='darkgrey')
    
    ax.tick_params(axis='x', which='major', pad=10)
    ax.tick_params(axis='y', which='major', pad=10)
    ax.tick_params(axis='z', which='major', pad=10)

    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])
    ax.set_zlim([0, 25])
    
    ax.set_xticks(np.arange(-25, 26, 10))
    ax.set_yticks(np.arange(-25, 26, 10))
    ax.set_zticks(np.arange(0, 26, 2.5))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')

    desired_angle = 90
    ax.plot_surface(X_scaled, Y_scaled, Z, cmap='terrain', rstride=1, cstride=1, antialiased=False)
    ax.view_init(elev=30, azim=desired_angle)
    plt.tight_layout()

    if save_visualization and vis_output_dir:
        os.makedirs(vis_output_dir, exist_ok=True)
        static_image_path = os.path.join(vis_output_dir, static_image_filename)
        
        plt.savefig(static_image_path, bbox_inches='tight')
        print(f"Static 3D visualization saved to: {static_image_path}")

    plt.close(fig)

if __name__ == "__main__":
    needs_dummy_data = False
    for model_key, model_path in model_info.items():
        if not os.path.exists(model_path):
            needs_dummy_data = True
            print(f"Dummy data needed: Model '{model_key}' not found at '{model_path}'")
            break
    if not needs_dummy_data and not os.path.exists(INPUT_IMAGE_PATH):
        needs_dummy_data = True
        print(f"Dummy data needed: Input image not found at '{INPUT_IMAGE_PATH}'")

    if needs_dummy_data:
        print("\n--- CREATING DUMMY MODELS AND IMAGE FOR TESTING ---")
        print("!!! WARNING: Please replace paths in 'model_info' with your actual trained model paths.")
        print(f"And '{INPUT_IMAGE_PATH}' with the path to your actual input image.")

        for model_key, model_path_dummy in model_info.items():
            os.makedirs(os.path.dirname(model_path_dummy) or '.', exist_ok=True)
            if not os.path.exists(model_path_dummy):
                dummy_model = ViT(input_size=MODEL_INPUT_SIZE)
                torch.save(dummy_model.state_dict(), model_path_dummy)
                print(f"Dummy ViT model '{model_key}' created at '{model_path_dummy}'.")

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

        output_dir_model = f"render/render_out/{model_name}"
        
        heatmap_overlay_png_dir = os.path.join(output_dir_model, "heatmaps_overlay_png")
        heatmap_overlay_png_name = f"heatmap_overlay_{model_name}.png"
        
        pure_heatmap_png_dir = os.path.join(output_dir_model, "heatmaps_pure_grayscale_png")
        pure_heatmap_png_name = f"heatmap_pure_grayscale_{model_name}.png"

        model_obj_base = os.path.join(output_dir_model, f"mountain_from_heatmap_{model_name}")
        vis_dir = os.path.join(output_dir_model, "3d_visualizations")
        static_image_name = f"3d_static_view_{model_name}_270deg.png" 

        heatmap_array, orig_w, orig_h = generate_heatmap_array_and_save_png_from_model(
            model_path=model_path,
            input_image_path=INPUT_IMAGE_PATH,
            model_input_size=MODEL_INPUT_SIZE,
            transform_pipeline=inference_transform,
            output_overlay_png_dir=heatmap_overlay_png_dir,
            output_overlay_png_filename=heatmap_overlay_png_name,
            output_pure_heatmap_png_dir=pure_heatmap_png_dir,
            output_pure_heatmap_png_filename=pure_heatmap_png_name
        )

        if heatmap_array is not None:
            print(f"Generando modelo 3D y guardando imagen estática para {model_name}...")
            generate_3d_model_and_save_static_image_from_heatmap(
                heatmap_array=heatmap_array,
                output_base_name=model_obj_base,
                scale_factor=SCALE_FACTOR_HEIGHT,
                resolution_scale=RESOLUTION_SCALE_MODEL,
                texture_paths_dict=TEXTURE_FILENAMES_DICT,
                save_visualization=True,
                vis_output_dir=vis_dir,
                static_image_filename=static_image_name
            )
        else:
            print(f"Heatmap generation failed for {model_name}. Skipping 3D generation.")

    print("\nProceso completo para todos los modelos.")