# Skin Cancer Lesion Classification with ResNet18 Convolutional Neural Network

The objective of my machine learning project was to develop a Flask web application designed for image recognition, specifically tailored for combat and contact sports. The application allows users to upload images of skin bumps or infections, and the system then labels them as malicious (potentially harmful) or not. This functionality is particularly valuable for sports like wrestling and Brazilian Jiu-Jitsu, where skin infections are common.

While I couldn't find a dataset specifically focused on skin infections related to these sports, I discovered the ISIC (International Skin Imaging Collaboration) website. The [ISIC Archive](https://www.isic-archive.com/) contains over 1.2 million images of various skin lesions, classified as malignant or benign. These images are sourced from 3D total body photographs and are primarily used for identifying skin cancers. I leveraged this dataset for training my model, adapting it to meet the needs of a preliminary check for skin cancer detection.


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

#### Download the Images 

In the **image-sourcing-script folder** edit the script to set your path variables to save the images downloaded from the ISIC API

```
python isic-api.py
```

#### Pre-Process Images

The script **preprocess_images.py** applies transformations to the images before they are consumed by the model 

Edit the script to set your path variables & run 

```
python preprocess_images.py
```

## 5. Verify Cuda availability 

Skip this part if you are not using GPU acceleration. 

I ran into some issues initially when I loaded **torch.cuda** & was still experiencing 100% CPU usage 

Long story short make sure that:  

CUDA is properly installed and added to the system PATH & that you have PyTorch with CUDA Support (in requirements.txt already)

Run the cuda_check.py to verify that the GPU device is detected below is the expected output: 

```
CUDA available: True
CUDA device name: <GPU Brand & Model>
```


## 6. Train the Model

Run the training script to train the ResNet18 model.

```
python resnet18-training.py
```

Here is the Output:

![terminal-output-training](https://github.com/user-attachments/assets/86d55d39-6654-4900-bc66-6a2ae298a84e)


### Visualisation with MatPlot 

Below are the 2 graphs that are generated afterwards:

![91 5-0 240valid-](https://github.com/user-attachments/assets/894eb904-474c-45d9-b4e7-3bd4cd9749b1)

Training and Validation Loss Plot:

**X-axis:** Epochs (number of training cycles)  
**Y-axis:** Loss values  
**Training Loss Line:** Shows how the model's error on the training data decreases over time  
**Validation Loss Line:** Shows how the model's error on the validation data changes over time  
**Goal:** Both lines should decrease. If validation loss increases while training loss decreases, it indicates overfitting  


![accuracy validation-](https://github.com/user-attachments/assets/38fd8570-b43d-426b-892a-733403ed081e)

Validation Accuracy Plot:

**X-axis:** Epochs  
**Y-axis:** Accuracy percentage  
**Validation Accuracy Line:** Shows how the model's accuracy on the validation data changes over the epochs
**Goal:** The accuracy should increase and stabilize at a high value.


## Results 

Validation Loss: Achieved a validation loss of 0.2389 and an accuracy of 91.5%.

## Model Details

- **Architecture**: ResNet18  
  Why?Upon a bit of research I found that this CNN architecture helps eliminate the vanishing gradient problem as they can become very small during backpropagation.
   
- **Optimizations**: Data augmentation, dropout for regularization, weight decay in the optimizer. (I have commented the optimizations in resnet18-training.py)

## Future Work
Further tuning of hyperparameters.  
Addition of more bigger dataset for broader classification capabilities.  


## Acknowledgements
[ISIC Archive](https://www.isic-archive.com/) for the dataset  
PyTorch for the deep learning framework.  



## License 

MIT License

Copyright © 2024 Joseph Moubarak

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.






