import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# 1. DEFINE THE MODEL (MUST MATCH simple_nn.py)
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

# 2. LOAD THE MODEL AND EVALUATE
@st.cache_resource # Caches the model so it only loads once
def load_and_evaluate_model():
    # Setup
    device = torch.device("cpu") # Use CPU for deployment simplicity
    PATH = './cifar_net.pth'
    
    # Load Model
    model = Net().to(device)
    try:
        # Load weights saved on your MPS device, mapped to CPU for deployment
        model.load_state_dict(torch.load(PATH, map_location=device))
        model.eval()
    except FileNotFoundError:
        st.error("Error: Model weights (cifar_net.pth) not found! Run simple_nn.py first.")
        return None, None

    # Load and Evaluate Test Data (for accuracy display)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
    
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
    return model, accuracy

# 3. STREAMLIT APPLICATION INTERFACE
if __name__ == '__main__':
    st.title("CIFAR-10 Image Classifier Demo 🧠")
    
    model, accuracy = load_and_evaluate_model()
    
    if model:
        st.markdown(f"**Loaded Model Accuracy:** <span style='font-size: 24px; color: green;'>{accuracy}%</span>", unsafe_allow_html=True)
        st.write("This model was trained on Apple Silicon (MPS) and is loaded via Streamlit.")