import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os

# --- 1. MODEL ARCHITECTURE (DEEPER/WIDER - MUST MATCH simple_nn.py) ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 1st Conv Block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 2nd Conv Block
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 3rd Conv Block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully Connected Layers (Input size: 128 * 8 * 8)
        self.fc1 = nn.Linear(128 * 8 * 8, 512) 
        self.fc2 = nn.Linear(512, 10) 
        
    def forward(self, x):
        # 1st Conv + BN + ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 2nd Conv + BN + ReLU + Pool
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # 3rd Conv + BN + ReLU + Pool
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Flattening: 128 * 8 * 8 = 8192 features
        x = torch.flatten(x, 1) 
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. CONFIGURATION AND LOADING ---
if __name__ == '__main__':
    # Use CPU for loading/testing unless you specifically configure for GPU/MPS
    device = torch.device("cpu") 

    # Define the path to your saved model weights
    # Note: Adjust './cifar_net.pth' if your file is in a different location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PATH = os.path.join(BASE_DIR, 'cifar_net.pth')

    # Load the model structure and weights
    model = Net().to(device)
    
    # Load the state dictionary from the saved file
    try:
        model.load_state_dict(torch.load(PATH, map_location=device))
        model.eval() # Set the model to evaluation mode
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit() # Exit if loading fails

    # --- 3. DATA LOADING AND TRANSFORMATION (for testing) ---
    # NOTE: The test transform should NOT include augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 64
    num_workers = 2

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- 4. ACCURACY CALCULATION ---
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct // total
    print(f'Accuracy of the saved network on the 10000 test images: {accuracy} %')