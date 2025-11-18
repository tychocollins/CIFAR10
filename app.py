import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# --- 1. MODEL DEFINITION (MUST MATCH THE 83% ACCURACY ARCHITECTURE) ---
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
        
        # Flattening
        x = torch.flatten(x, 1) 
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. MODEL LOADING AND ACCURACY CALCULATION ---
@st.cache_resource
def load_and_evaluate_model():
    device = torch.device("cpu") 
    
    # Absolute path for cloud deployment
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PATH = os.path.join(BASE_DIR, 'cifar_net.pth')
    
    model = Net().to(device)
    
    # Load the new weights
    try:
        model.load_state_dict(torch.load(PATH, map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"Error loading model weights (cifar_net.pth): {e}")
        return None, 0

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Accuracy calculation logic
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

# --- 3. USER INTERFACE AND PREDICTION LOGIC ---
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")
    st.title("CIFAR-10 Image Classifier Demo 🧠")
    
    # This line now loads the correct weights file
    model, accuracy = load_and_evaluate_model()
    
    if model:
        st.markdown(f"**Loaded Model Accuracy (Test Set):** <span style='font-size: 24px; color: #4CAF50;'>{accuracy}%</span>", unsafe_allow_html=True)
        st.write("This CNN model was trained on Apple Silicon (MPS) and is loaded via Streamlit.")
        
        st.markdown("---")
        st.subheader("Try It Out: Image Prediction")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            
            if st.button('Classify Image'):
                
                # Preprocessing
                transform = transforms.Compose(
                    [transforms.Resize((32, 32)),  
                     transforms.ToTensor(),         
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                
                input_tensor = transform(image)
                input_batch = input_tensor.unsqueeze(0) 
                
                # Inference
                with torch.no_grad():
                    output = model(input_batch)
                
                _, predicted_idx = torch.max(output, 1)
                predicted_class = CLASSES[predicted_idx.item()]
                
                # Display Result
                st.success(f"### Prediction: {predicted_class.upper()}")
                
                probabilities = F.softmax(output, dim=1)[0] * 100
                top_prob, top_catid = torch.topk(probabilities, 3)
                
                st.caption("Top 3 Confidences:")
                for i in range(top_prob.size(0)):
                    st.write(f"- **{CLASSES[top_catid[i]]}**: {top_prob[i].item():.2f}%")