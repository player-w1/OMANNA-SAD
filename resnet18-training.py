import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

#Define preprocessed paths
PREPROCESSED_PATH = r"C:\Users\Joseph Moubarak\Desktop\OMANNA SAD\dataset\preprocessed"

#Define image transformations 
#optimisation1 added transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Data augmentation: Random horizontal flip
    transforms.RandomRotation(30),  # Data augmentation: Random rotation
    transforms.RandomResizedCrop(224),  # Data augmentation: Random resized crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Load datasets 
dataset = datasets.ImageFolder(PREPROCESSED_PATH, transform=transform)

#Split the dataset in 2 training & validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#data loaders with the split and sampling 32 images before the weights are updated
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#Load ResNet18
model = resnet18(pretrained=True)

#Output finallayer to match the 2 classifications (malignant&benign) to be changed later for precise diagnostics
num_classes = len(dataset.classes)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Optimization2 Add dropout prevent model becoming too reliant on certain neurons) Goal:Reduce overfitting
    nn.Linear(model.fc.in_features, num_classes)
)

#Use cuda module 2 send load on GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  #was not sure if Cuda module was working constant high cpu load when running with spikes on GPU
model.to(device)


# Loss function and optimizer (weight & biases)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) #optimization3 Keeping weights small 4 simpler patterns goal: reduce overfitting

#Training loop (1 epoch = full iteration of dataset)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Validate loss
model.eval()
val_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%")

# Save the model
torch.save(model.state_dict(), "skin_lesion_model.pth")

