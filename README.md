# Skin Lesion Classification with ResNet18

The objective of my machine learning project was to develop a Flask web application designed for image recognition, specifically tailored for combat and contact sports. The application allows users to upload images of skin bumps or infections, and the system then labels them as malicious (potentially harmful) or not. This functionality is particularly valuable for sports like wrestling and Brazilian Jiu-Jitsu, where skin infections are common.

While I couldn't find a dataset specifically focused on skin infections related to these sports, I discovered the ISIC (International Skin Imaging Collaboration) website. The ISIC repository contains over 1.2 million images of various skin lesions, classified as malignant or benign. These images are sourced from 3D total body photographs and are primarily used for identifying skin cancers. I leveraged this dataset for training my model, adapting it to meet the needs of a preliminary check for skin cancer detection.


## Project Overview

The goal of this project is to build a machine learning model that can accurately classify skin lesions, aiding in early detection of skin cancer. The model is trained on images from the ISIC archive.

### 1. Clone the Repository

```bash
git clone https://github.com/player-w1/OMANNA-SAD.git
cd <your-repo-directory>
```

### 2. Set Up Virtual Environment

```
.\env\Scripts\Activate.ps1  # Windows
source env/bin/activate     # macOS/Linux
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Download and Preprocess Data

In the image-sourcing-script folder, set your path variables to save the images downloaded ISIC API

```
python isic-api.py
```


