# Interactive Performances: Real-Time AI Systems Shaping Audience and Artist Experiences

In this repository, you'll find all the source code for my Degree's Final Project, focused on using real-time AI systems to enhance live performances from both the artist’s and the audience’s perspectives.

---

## 🧪 Environment Set-up

Make sure Python is installed on your machine. This project was developed using **Python 3.8.10**.

### 1. Create and Activate the Virtual Environment

```bash
python3.8 -m venv tfg_env
source tfg_env/bin/activate  # On Windows: tfg_env\Scripts\activate
```

### 2. Install Dependencies

Install all necessary packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 🔥 Heatmaps Module

<div align="center">
  <img src="assets/blender_pipeline.png" alt="System's Pipeline" width="700">
</div>

### 📂 Dataset Set-up

This project uses the **JHU-Crowd++** dataset from Kaggle.

1. Download the dataset from:  
   https://www.kaggle.com/datasets/hoangxuanviet/jhu-crowd/data  
2. Run the preprocessing script:  
   `heatmaps/data_preprocessing.py`

This step generates the required heatmaps since they're not included in the original dataset.

> ⚠️ **Important:** Make sure to configure the correct input/output paths in `data_preprocessing.py`.

---

### 🛠️ Blender Set-up

Rendering is handled via Blender. Download the latest version here:  
https://www.blender.org/

---

### 🧠 Model Training

- Training scripts are located in the `train/` directory.
- Hyperparameters can be edited at the top of each script.
- CNN-trained models are saved in `models/cnn/`.
- **UNet** and **ViT** weights must be downloaded from their respective subfolders.

> ⚠️ **Note:** Some model files may include download links; ensure they're placed correctly before running.

<br>

![UNet BCE with Logits training results](assets/training_bce_unet.png "UNet BCE with Logits training results")  
*UNet model results using BCE with Logits Loss.*

---

### 🖼️ Inference & Visualization

Below is an example using a concert image. From left to right:
- **Original Image**
- **Overlayed Heatmap**
- **Grayscale Heatmap**
- **3D Plot**
- **Blender Render**

<table style="width:100%; border: none;">
  <tr>
    <td style="border: none; text-align:center;">
      <img src="assets/stand.jpg" alt="Original image" width="180">
    </td>
    <td style="border: none; text-align:center;">
      <img src="assets/heatmap_overlay_unet_bce.png" alt="Overlayed Heatmap" width="180">
    </td>
    <td style="border: none; text-align:center;">
      <img src="assets/heatmap_pure_grayscale_unet_bce.png" alt="Grayscale Heatmap" width="180">
    </td>
    <td style="border: none; text-align:center;">
      <img src="assets/3d_rotation_unet_bce.gif" alt="3D Plot" width="180">
    </td>
    <td style="border: none; text-align:center;">
      <img src="assets/stand_render.png" alt="Render" width="200">
    </td>
  </tr>
</table>

#### Manual Rendering (Basic)

To produce the render manually:
1. Generate the grayscale heatmap and 3D plot from `render/model_name/`.
2. Open `render/montaña_render.blend` in Blender.
3. Load the grayscale heatmap manually in the **Material Properties** panel.

---

### ⚙️ Automated Pipeline

Inside the `heatmaps/auto/` directory, you'll find an automated version of the rendering pipeline.

Just run:

```bash
python generate.py
```

The script:
- Loads the model of your choice
- Generates the heatmap
- Automatically connects it to the `.blend` file

#### Recommended Folder Structure:

```
├── 📁 heatmaps/
├── 📄 model_xxx.pth      # Model weights
├── 🖼️ stand.jpg          # Original image
├── ⛰️ montaña_render.blend
├── 🐍 blender_update_displacement.py
├── 🐍 generate.py
├── 🐍 model_cnn.py
├── 🐍 model_unet.py
├── 🐍 model_vit.py
└── 🧰 utils.py
```

> ⚠️ **Reminder:** Double-check all file paths before execution.

---

## 🎵 Music Interaction Module

<div align="center">
  <img src="assets/touchdesigner_pipeline.png" alt="System's Pipeline" width="700">
</div>

### 🧩 TouchDesigner Set-up

This module uses **TouchDesigner** for real-time audio-visual feedback.  
Download it here:  
https://derivative.ca/download

---

### 🖐️ Software Execution

1. Open `music/note_tracker.72.toe` in TouchDesigner and press **Play**.
2. Run `music/hand_tracker.py` to activate real-time hand tracking using MediaPipe.

<br>

#### Live Interface Preview:
<div align="center">
  <img src="assets/touchdesigner_interaction.png" alt="TouchDesigner Interaction" width="200">
</div>

---

### 🧪 Testing and Analysis

- For standalone testing, run:
  ```bash
  python music/mediapipe_test.py
  ```
  Set the duration of the session as needed.

- After testing, run:
  ```bash
  python music/analyze_results.py
  ```
  This will help analyze captured gesture data and system response.
