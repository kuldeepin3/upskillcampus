# 🌱 Crop & Weed Detection using YOLOv8

This repository contains a machine learning project designed to solve a critical agricultural business problem: selective pesticide application. By using object detection to distinguish between **crops** and **weeds**, farmers can target weeds precisely, thereby reducing chemical usage, cutting operational costs, protecting crops and soil health, and minimizing environmental impact.

---

## 📌 Business Problem & Intuition

Farmers traditionally spray herbicides uniformly across entire fields. This standard approach has several drawbacks:
* **High Costs:** Massive volumes of pesticides are wasted on areas without weeds.
* **Environmental & Soil Damage:** Runoff pollutes water systems and degrades soil quality.
* **Crop Damage:** Chemical exposure can stress or harm the actual crops.

**The Solution:** Using **YOLO (You Only Look Once)**, this project processes agricultural field images in a single forward pass to identify and localize weeds and crops in real time. This enables precise, spot-spraying systems.

For more details on the business motivation, see [intution.txt](file:///K:/data/Cd/ML/UpSkill/Crop%20&%20Weed%20Detection%20using%20YOLO/intution.txt).

---

## 📂 Project Structure

* **[app.py](file:///K:/data/Cd/ML/UpSkill/Crop%20&%20Weed%20Detection%20using%20YOLO/app.py)**: The main interactive Web UI built with Streamlit.
* **[Notebook/Crop_Weed_Detection_YOLO.ipynb](file:///K:/data/Cd/ML/UpSkill/Crop%20&%20Weed%20Detection%20using%20YOLO/Notebook/Crop_Weed_Detection_YOLO.ipynb)**: Jupyter Notebook covering data exploration, training setup, and validation checks.
* **[Data/data.yaml](file:///K:/data/Cd/ML/UpSkill/Crop%20&%20Weed%20Detection%20using%20YOLO/Data/data.yaml)**: YOLOv8 configuration file specifying class names (`crop` and `weed`) and folder structures.
* **Notebook/runs/detect/train-2/weights/best.pt**: The trained model weights generated during training.
* **[requirments.txt](file:///K:/data/Cd/ML/UpSkill/Crop%20&%20Weed%20Detection%20using%20YOLO/requirments.txt)**: List of dependencies required to run the project.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system. 

### 2. Installation
Install the project dependencies:
```bash
pip install -r requirments.txt
```

> [!NOTE]
> Key requirements include `streamlit`, `ultralytics` (YOLO), `opencv-python`, `torch`, `matplotlib`, and `pillow`.

### 3. Run the Web Application
Launch the Streamlit dashboard:
```bash
python -m streamlit run app.py
```
After running, open `http://localhost:8501` in your web browser.

---

## 🛠️ Model Training & Dataset Details

The model was trained using the **YOLOv8 Nano (yolov8n)** architecture which provides an optimal balance of latency and accuracy for real-time edge devices (such as smart tractors/drones).

* **Dataset Size:** 1,300 images with corresponding annotation labels.
* **Classes:** 
  1. `crop` (Label 0)
  2. `weed` (Label 1)
* **Training Settings:**
  - Input Image Size (`imgsz`): 512x512
  - Epochs: 20
  - Batch Size: Auto

### Training Code Snippet
```python
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model on custom agriculture data
model.train(
    data="../Data/data.yaml",
    epochs=20,
    imgsz=512
)
```

---

## 📊 Performance Metrics

After 20 training epochs, the model achieved the following performance on the validation set:

| Metric | Value |
| :--- | :---: |
| **Precision** | `84.07%` |
| **Recall** | `72.78%` |
| **mAP50** | `82.66%` |
| **mAP50-95** | `53.14%` |

Training curves, batch predictions, and confusion matrices are located in the `Notebook/runs/detect/train-2/` directory.

---

## 🖥️ Streamlit App Features

The Streamlit web application allows users to interactively test the model:
1. **Upload Image:** Supports `.jpg`, `.jpeg`, `.png`, and `.webp` images.
2. **Interactive Bounding Boxes:** Displays the original image alongside the model predictions with labeled boxes.
3. **Detection Summary:** Lists detected targets, showing the classified label (`crop`/`weed`) and confidence scores.
