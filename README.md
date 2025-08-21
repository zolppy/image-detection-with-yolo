# YOLOv8 Fruit Detection Project

This project demonstrates how to train a YOLOv8 object detection model to detect various types of fruits. The process involves loading a pre-trained YOLOv8 model, training it on a custom dataset of fruits, evaluating its performance, and running inference on sample images.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation](#3-evaluation)
  - [4. Inference](#4-inference)
- [Results](#results)
- [Dependencies](#dependencies)

## Project Overview

The goal of this project is to build and train a real-time fruit detection system using the YOLOv8 architecture. We use a publicly available fruits dataset and fine-tune a pre-trained YOLOv8 model to accurately identify and locate fruits in images.

The project notebook (`main.ipynb`) covers the following key steps:
1.  **Environment Setup**: Cloning the necessary dataset and installing the `ultralytics` library.
2.  **Model Loading**: Loading a pre-trained YOLOv8s model.
3.  **Training**: Fine-tuning the model on the custom fruits dataset for 20 epochs.
4.  **Validation**: Evaluating the model's performance on the validation set.
5.  **Inference**: Running predictions on single and multiple images to see the model in action.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine or cloud environment (like Google Colab).

### Prerequisites

- Python 3.8 or later
- `pip` for package installation
- `git` for cloning the repository

### Installation

1.  **Clone the Fruits Detection Dataset:**
    The dataset is provided in a public GitHub repository. Clone it to your local machine.
    ```bash
    git clone [https://github.com/lightly-ai/dataset_fruits_detection.git](https://github.com/lightly-ai/dataset_fruits_detection.git)
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd dataset_fruits_detection
    ```

3.  **Install the required Python library:**
    This project primarily relies on the `ultralytics` package, which contains the YOLOv8 implementation.
    ```bash
    pip install ultralytics
    ```
    Other dependencies like `pandas`, `numpy`, `matplotlib`, and `opencv-python` are generally installed alongside `ultralytics` or are standard in data science environments. If not, you can install them via pip:
    ```bash
    pip install pandas numpy matplotlib opencv-python
    ```

## Usage

The `main.ipynb` notebook provides a step-by-step guide. Here is a summary of the workflow:

### 1. Data Preparation
The dataset is expected to be in the YOLO format, with a `data.yaml` file that specifies the paths to training and validation sets, as well as the class names. This is already configured in the cloned repository.

### 2. Model Training
A pre-trained YOLOv8s model is loaded and then trained on the fruit dataset. The training configuration is set as follows:
-   **Epochs**: 20
-   **Image Size**: 640x640 pixels
-   **Batch Size**: 16
-   **Pretrained**: True

You can start the training process by running the corresponding cell in the notebook:
```python
from ultralytics import YOLO

# Load a pretrained YOLOv8s model
model = YOLO("yolov8s.pt")

# Train the model
results = model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    pretrained=True
)
```

### 3. Evaluation
After training, the model's performance is evaluated on the validation set to check metrics like mean Average Precision (mAP).
```python
# Evaluate the trained model
metrics = model.val()
```

### 4. Inference
You can use the trained model to make predictions on new images. The notebook provides examples for running inference on a single image and a batch of images. The results (images with bounding boxes) are saved automatically.

```python
# Run inference on a sample image
sample_image = "valid/images/0_0_640.jpg"
preds = model.predict(sample_image, save=True)
```

## Results

The training process will save the best model weights in a `runs/detect/train/weights/` directory. The inference results, including images with bounding boxes drawn on them, will be saved in `runs/detect/predict/`.

The notebook visualizes the predictions using `matplotlib` to display the output directly.

## Dependencies

-   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
-   [PyTorch](https://pytorch.org/)
-   OpenCV-Python
-   Matplotlib
-   NumPy
-   Pandas
