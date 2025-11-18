import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# --- 1. CONFIGURATION AND DEVICE SETUP ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# --- 2. DATA LOADING AND AUGMENTATION ---
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64 # Increased batch size for faster training
num_workers = 2 

# Load data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# --- 3. MODEL ARCHITECTURE (DEEPER/WIDER) ---
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
        
        # Fully Connected Layers (Input size: 128 channels * 4x4 image patch)
        self.fc1 = nn.Linear(128 * 8 * 8, 512) 
        self.fc2 = nn.Linear(512, 10) 
        
    def forward(self, x):
        # 1st Conv + BN + ReLU
        x = F.relu(self.bn1(self.conv1(x))) # 32x32
        
        # 2nd Conv + BN + ReLU + Pool
        x = F.relu(self.bn2(self.conv2(x))) # 32x32
        x = self.pool(x) # 16x16
        
        # 3rd Conv + BN + ReLU + Pool
        x = F.relu(self.bn3(self.conv3(x))) # 16x16
        x = self.pool(x) # 8x8
        
        # Flattening: 128 * 8 * 8 = 8192 features
        x = torch.flatten(x, 1) 
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net().to(device)


# --- 4. LOSS FUNCTION AND OPTIMIZER ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# --- 5. TRAINING LOOP ---
if __name__ == '__main__':
    num_epochs = 20 # Increased epochs for deeper model
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):  
        net.train() # Set model to training mode
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # Print more frequently due to larger batch size
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training.')

    # --- 6. SAVE MODEL ---
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Model saved to {PATH}')

    # --- 7. FINAL TEST AND ACCURACY CHECK ---
    net.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the new network on the 10000 test images: {100 * correct // total} %')