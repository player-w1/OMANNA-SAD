import os


try:  #Had problems with Numpy (was using Numpy 2.0, some dependancies require Numpy version<2.0)
    import numpy as np
    print("Numpy version:", np.__version__)
except ImportError as e:
    print("Numpy is not available:", e)
    exit(1)

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

BENIGN_PATH = r"C:\Users\Joseph Moubarak\Desktop\OMANNA SAD\dataset\benign"
MALIGNANT_PATH = r"C:\Users\Joseph Moubarak\Desktop\OMANNA SAD\dataset\malignant"
PREPROCESSED_PATH = r"C:\Users\Joseph Moubarak\Desktop\OMANNA SAD\dataset\preprocessed"

#Mkdir for the preprocessed  datasets
os.makedirs(os.path.join(PREPROCESSED_PATH, 'benign'), exist_ok=True)
os.makedirs(os.path.join(PREPROCESSED_PATH, 'malignant'), exist_ok=True)

#Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(30),  
    transforms.ToTensor(),  # Converts images to tensors for PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# fix the tensor object cannot be saved error
to_pil = transforms.ToPILImage()

def preprocess_images(input_path, output_path, transform):
    for img_name in tqdm(os.listdir(input_path)):
        img_path = os.path.join(input_path, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            img = to_pil(img)  # Convert tensor back to PIL
            output_img_path = os.path.join(output_path, img_name)
            img.save(output_img_path)
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")

print("Preprocessing benign images...")
preprocess_images(BENIGN_PATH, os.path.join(PREPROCESSED_PATH, 'benign'), transform)

print("Preprocessing malignant images...")
preprocess_images(MALIGNANT_PATH, os.path.join(PREPROCESSED_PATH, 'malignant'), transform)
