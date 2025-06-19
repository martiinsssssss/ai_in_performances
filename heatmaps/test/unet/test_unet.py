import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import json
import random
from datetime import datetime

class JHU_CrowdPP_TestDataset(Dataset):
    def __init__(self, images_dir, heatmaps_dir, input_size=(224, 224), transform=None):
        self.image_paths = []
        self.heatmap_paths = []
        self.transform = transform
        self.input_size = input_size
        self.original_image_paths = [] 

        if not os.path.exists(images_dir) or not os.path.exists(heatmaps_dir):
            raise FileNotFoundError(
                f"Directories not found: {images_dir} and {heatmaps_dir}. Please check your JHU-Crowd++ dataset structure."
            )

        image_filenames = sorted(os.listdir(images_dir))
        for img_filename in image_filenames:
            if img_filename.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(images_dir, img_filename)

                heatmap_filename = img_filename.replace(".jpg", ".npy").replace(".png", ".npy").replace(".jpeg", ".npy")
                heatmap_path = os.path.join(heatmaps_dir, heatmap_filename)

                if os.path.exists(heatmap_path):
                    self.image_paths.append(img_path)
                    self.heatmap_paths.append(heatmap_path)
                    self.original_image_paths.append(img_path)
                else:
                    print(f"Warning: Heatmap file not found for {img_filename}. Skipping.")

        if not self.image_paths:
            raise ValueError("No image-heatmap pairs found in the specified directory.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        heatmap_path = self.heatmap_paths[idx]
        original_img_path = self.original_image_paths[idx]

        original_image_pil = Image.open(original_img_path).convert("RGB")
        
        resize_only_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])
        original_image_tensor_for_display = resize_only_transform(original_image_pil)


        image_for_model = Image.open(img_path).convert("RGB")

        heatmap = np.load(heatmap_path)
        heatmap = np.clip(heatmap, 0, None)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).convert("L")
        heatmap_img = heatmap_img.resize(self.input_size, resample=Image.BILINEAR)

        if self.transform:
            image_for_model = self.transform(image_for_model)
        
        heatmap_tensor = transforms.ToTensor()(heatmap_img)

        return original_image_tensor_for_display, image_for_model, heatmap_tensor, img_path 

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


def calculate_heatmap_mae(predicted_heatmaps, ground_truth_heatmaps):
    """
    Calculates the Pixel-wise Mean Absolute Error (MAE) between heatmaps.
    Args:
        predicted_heatmaps (torch.Tensor): Predicted heatmaps (B, 1, H, W), expected in the range [0,1].
        ground_truth_heatmaps (torch.Tensor): Ground truth heatmaps (B, 1, H, W), expected in the range [0,1].
    Returns:
        float: The mean MAE across all pixels and batch samples.
    """
    return torch.mean(torch.abs(predicted_heatmaps - ground_truth_heatmaps)).item()

def test_model(model, dataloader, device, model_loss_type='mse', visualization_dir=None, num_visualizations=3):
    """
    Tests a heatmap generation model and saves example visualizations, including the one most similar to the ground truth.
    """
    model.eval()
    model.to(device)

    total_heatmap_mae = 0
    num_batches = 0
    

    collected_visualizations = {
        'best': {'mae': float('inf'), 'data': None}, 
        'random': [] 
    }
    saved_img_paths_for_viz = set() 

    if visualization_dir and not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    print(f"Testing model on {len(dataloader.dataset)} images...")
    with torch.no_grad():
        for i, (original_images_tensor_for_display, images_for_model, ground_truth_heatmaps, img_paths) in enumerate(tqdm(dataloader)):
            images_for_model = images_for_model.to(device)
            ground_truth_heatmaps = ground_truth_heatmaps.to(device)

            model_output = model(images_for_model)
            predicted_heatmaps = None

            if model_loss_type == 'bcewithlogits':
                predicted_heatmaps = torch.sigmoid(model_output)
            elif model_loss_type == 'bceloss':
                predicted_heatmaps = torch.sigmoid(model_output)
            elif model_loss_type in ['l1', 'mse', 'smoothl1']:
                predicted_heatmaps = torch.clamp(model_output, 0, 1)
            else:
                raise ValueError(f"Unknown loss type: {model_loss_type}. Supported types: 'l1', 'mse', 'smoothl1', 'bce', 'bcewithlogits'.")

            # Calculate MAE for each image in the batch
            batch_individual_mae = torch.abs(predicted_heatmaps - ground_truth_heatmaps).mean(dim=[1, 2, 3]).cpu()
            
            # Update overall MAE
            total_heatmap_mae += batch_individual_mae.sum().item()
            num_batches += batch_individual_mae.size(0)

            if visualization_dir:
                for j in range(images_for_model.size(0)):
                    current_img_mae = batch_individual_mae[j].item()
                    current_img_path = img_paths[j]
                    
                    viz_data_tuple = (
                        original_images_tensor_for_display[j].permute(1, 2, 0).cpu().numpy(),
                        ground_truth_heatmaps[j][0].cpu().numpy(),
                        predicted_heatmaps[j][0].cpu().numpy(),
                        os.path.splitext(os.path.basename(current_img_path))[0]
                    )

                    if current_img_mae < collected_visualizations['best']['mae']:
                        collected_visualizations['best']['mae'] = current_img_mae
                        collected_visualizations['best']['data'] = viz_data_tuple
                        
                    if len(collected_visualizations['random']) < num_visualizations and \
                       current_img_path not in saved_img_paths_for_viz:
                        
                        collected_visualizations['random'].append({
                            'mae': current_img_mae,
                            'data': viz_data_tuple,
                            'path': current_img_path
                        })
                        saved_img_paths_for_viz.add(current_img_path)
    
    if visualization_dir:
        if collected_visualizations['best']['data']:
            original_img_np, gt_np, pred_np, img_filename_base = collected_visualizations['best']['data']
            
            pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
            gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original_img_np)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            axes[1].imshow(original_img_np)
            axes[1].imshow(gt_np, cmap='jet', alpha=0.6)
            axes[1].set_title("Ground Truth Heatmap (Best MAE)")
            axes[1].axis('off')
            axes[2].imshow(original_img_np)
            axes[2].imshow(pred_np, cmap='jet', alpha=0.6)
            axes[2].set_title("Predicted Heatmap (Best MAE)")
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            plot_filename = os.path.join(visualization_dir, f"example_{img_filename_base}_best_mae.png")
            plt.savefig(plot_filename)
            plt.close(fig)
            print(f"  Saved best MAE visualization to {plot_filename}")


        random_viz_data = random.sample(collected_visualizations['random'], min(num_visualizations, len(collected_visualizations['random'])))

        for k, viz_item in enumerate(random_viz_data):
            original_img_np, gt_np, pred_np, img_filename_base = viz_item['data']
            
            pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
            gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original_img_np)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            axes[1].imshow(original_img_np)
            axes[1].imshow(gt_np, cmap='jet', alpha=0.6)
            axes[1].set_title("Ground Truth Heatmap")
            axes[1].axis('off')
            axes[2].imshow(original_img_np)
            axes[2].imshow(pred_np, cmap='jet', alpha=0.6)
            axes[2].set_title("Predicted Heatmap")
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            plot_filename = os.path.join(visualization_dir, f"example_{img_filename_base}_random_{k+1}.png")
            plt.savefig(plot_filename)
            plt.close(fig)
            print(f"  Saved random visualization {k+1} to {plot_filename}")

    mean_heatmap_mae = total_heatmap_mae / num_batches
    return {"heatmap_mae": mean_heatmap_mae}

def save_metrics_plot(metrics_df, output_plot_path):
    """
    Generates and saves a 3D bar chart of Heatmap MAE for all models.
    """
    if metrics_df.empty:
        print("No metrics to plot for overall progression.")
        return

    plt.style.use('seaborn-v0_8-darkgrid')

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    #Data for the bars
    xpos = np.arange(len(metrics_df))
    ypos = np.zeros(len(metrics_df))
    zpos = np.zeros(len(metrics_df))
    
    dx = np.ones(len(metrics_df)) * 0.8
    dy = np.ones(len(metrics_df)) * 0.8
    dz = metrics_df['Heatmap MAE'].values

    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_df)))

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
    ax.set_xticks(xpos + dx/2)
    ax.set_xticklabels(metrics_df['Model Name'], rotation=45, ha='right', fontsize=10)

    ax.set_xlabel('Model Name', fontsize=12, weight='bold')
    ax.set_ylabel('')
    ax.set_zlabel('Heatmap MAE', fontsize=12, weight='bold')
    ax.set_title('3D Heatmap MAE Comparison Across UNet Models', fontsize=16, weight='bold')

    ax.view_init(elev=20, azim=-45)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"\n3D MAE comparison plot saved to {output_plot_path}")


if __name__ == "__main__":
    JHU_CROWDPP_ROOT = 'jhu_crowd'
    TEST_IMAGES_DIR = os.path.join(JHU_CROWDPP_ROOT, "test", "images")
    TEST_HEATMAPS_DIR = os.path.join(JHU_CROWDPP_ROOT, "test", "heatmaps")
    MODEL_INPUT_SIZE = (224, 224)

    #MODEL LOADING
    MODEL_CONFIGS = {
        'UNet_MSELoss_Model': {
            'path': 'outputs/UNet/MSE/model_mse.pth',
            'loss_type': 'mse',
            'base_visualizations_path': 'model_test/UNet/visualizations',
        },
        'UNet_L1Loss_Model': {
            'path': 'outputs/UNet/L1/model_l1.pth',
            'loss_type': 'l1',
            'base_visualizations_path': 'model_test/UNet/visualizations',
        },
        'UNet_BCEWithLogitsLoss_Model': {
            'path': 'outputs/UNet/BCE/model_bcelogits.pth',
            'loss_type': 'bcewithlogits',
            'base_visualizations_path': 'model_test/UNet/visualizations',
        },
    }

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading JHU-Crowd++ test dataset...")
    try:
        test_dataset = JHU_CrowdPP_TestDataset(
            images_dir=TEST_IMAGES_DIR,
            heatmaps_dir=TEST_HEATMAPS_DIR,
            input_size=MODEL_INPUT_SIZE,
            transform=test_transform
        )
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        print(f"Test dataset loaded successfully with {len(test_dataset)} images.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading dataset: {e}")
        print("Please double-check that 'JHU_CROWDPP_ROOT', 'TEST_IMAGES_DIR', and 'TEST_HEATMAPS_DIR' are correctly configured and accessible.")
        exit()

    all_model_metrics = {}
    metrics_df = pd.DataFrame(columns=['Model Name', 'Heatmap MAE'])

    for model_name, config in MODEL_CONFIGS.items():
        model_path = config['path']
        loss_type = config['loss_type']
        base_visualizations_path = config.get('base_visualizations_path')

        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_viz_output_dir = os.path.join(base_visualizations_path, loss_type, current_timestamp)
        os.makedirs(model_viz_output_dir, exist_ok=True)

        print(f"\n--- Testing: {model_name} (Loss Type: {loss_type.upper()}) ---")
        print(f"Saving visualizations to: {model_viz_output_dir}")

        try:
            model = UNet(n_channels=3, n_classes=1)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Model weights loaded successfully from: {model_path}")

            metrics = test_model(
                model=model,
                dataloader=test_dataloader,
                device=DEVICE,
                model_loss_type=loss_type,
                visualization_dir=model_viz_output_dir, 
                num_visualizations=3
            )
            all_model_metrics[model_name] = metrics

            new_row = pd.DataFrame([{'Model Name': model_name, 'Heatmap MAE': metrics['heatmap_mae']}])
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

            print(f"Results for {model_name}:")
            print(f"  Heatmap MAE: {metrics['heatmap_mae']:.6f}")

        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. Skipping {model_name}.")
        except Exception as e:
            print(f"An unexpected error occurred while loading or testing {model_name}: {e}")

    print("\n--- Summary of All Model Metrics ---")
    print(metrics_df)

    overall_output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_Overall")
    overall_metrics_output_dir = os.path.join('model_test', 'UNet', 'overall_results', overall_output_timestamp)
    os.makedirs(overall_metrics_output_dir, exist_ok=True)
    overall_csv_path = os.path.join(overall_metrics_output_dir, 'unet_all_models_heatmap_evaluation_metrics.csv')
    metrics_df.to_csv(overall_csv_path, index=False)
    print(f"\nOverall heatmap evaluation metrics saved to {overall_csv_path}")

    metrics_plot_output_dir = os.path.join('model_test', 'UNet', 'overall_results', overall_output_timestamp) 
    metrics_plot_filename = os.path.join(metrics_plot_output_dir, 'unet_all_models_mae_3d_bar_chart.png')
    
    if not metrics_df.empty:
        save_metrics_plot(metrics_df, metrics_plot_filename)
    else:
        print("No metrics to plot. DataFrame is empty.")