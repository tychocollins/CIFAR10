import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# 1. Device Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} for testing.")


# 2. Define the CNN Architecture (MUST MATCH simple_nn.py)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x


# 3. Main Execution Block for Loading and Testing
if __name__ == '__main__':
    print("\n--- Testing Saved Model ---")
    
    # Load the Model
    PATH = './cifar_net.pth'
    model = Net().to(device)
    # Load the trained weights from the .pth file
    try:
        model.load_state_dict(torch.load(PATH, map_location=device))
        model.eval() # Set model to evaluation mode
        print(f"Successfully loaded trained model from {PATH}")
    except FileNotFoundError:
        print(f"Error: Model file {PATH} not found. Did you run simple_nn.py first?")
        exit()

    # Data Setup (Needed to check accuracy)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

    # Calculate Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Model loaded from file achieved an accuracy of: {100 * correct // total} %')

    # 4. Individual Prediction Demo (Inference)
    
    # Get one batch of images and labels from the test set
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Move the images to the device the model is on (MPS)
    images = images.to(device)
    
    # Run the single batch through the model
    with torch.no_grad():
        outputs = model(images)
    
    # Get the predicted class index (highest score)
    _, predicted = torch.max(outputs, 1)

    # CIFAR-10 classes tuple from the training file
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Print the results for the first 4 images in the batch
    print("\n--- Individual Prediction ---")
    print(f'True Labels:    {[classes[labels[j]] for j in range(4)]}')
    print(f'Model Prediction: {[classes[predicted[j]] for j in range(4)]}')