import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# --- 1. MODEL DEFINITION (Must match simple_nn.py) ---
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
        x = F.relu(x.fc1(x))
        x = F.relu(x.fc2(x))
        x = x.fc3(x) 
        return x

# --- 2. MODEL LOADING AND ACCURACY CALCULATION ---
# @st.cache_resource is critical for loading the model only once
@st.cache_resource
def load_and_evaluate_model():
    # Setup paths and device
    device = torch.device("cpu") 
    
    # Use the absolute path logic we fixed earlier for cloud deployment
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PATH = os.path.join(BASE_DIR, 'cifar_net.pth')
    
    # Load Model
    model = Net().to(device)
    try:
        model.load_state_dict(torch.load(PATH, map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, 0

    # Accuracy Calculation (used for the header display)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Note: CIFAR10 data is downloaded to a temporary location on the cloud
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
    st.title("CIFAR-10 Image Classifier Demo 🧠")
    
    # Load model and display performance metrics
    model, accuracy = load_and_evaluate_model()
    
    if model:
        st.markdown(f"**Loaded Model Accuracy (Test Set):** <span style='font-size: 24px; color: #4CAF50;'>{accuracy}%</span>", unsafe_allow_html=True)
        st.write("This model was trained on Apple Silicon (MPS) and is loaded via Streamlit.")
        
        st.markdown("---")
        st.subheader("Try It Out: Image Prediction")
        
        # File uploader widget
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            
            # Button to trigger prediction
            if st.button('Classify Image'):
                
                # 1. Preprocessing (MUST match training transform)
                transform = transforms.Compose(
                    [transforms.Resize((32, 32)),  # Resize image to 32x32 pixels
                     transforms.ToTensor(),         # Convert to PyTorch tensor
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                
                # Apply transform to the PIL image
                input_tensor = transform(image)
                # Add a batch dimension (1 image)
                input_batch = input_tensor.unsqueeze(0) 
                
                # 2. Inference (Prediction)
                with torch.no_grad():
                    output = model(input_batch)
                
                # Get the predicted class index (highest score)
                _, predicted_idx = torch.max(output, 1)
                predicted_class = CLASSES[predicted_idx.item()]
                
                # 3. Display Result
                st.markdown(f"### Model Prediction: **{predicted_class.upper()}**")
                
                # Optional: Display confidence scores
                probabilities = F.softmax(output, dim=1)[0] * 100
                top_prob, top_catid = torch.topk(probabilities, 3)
                
                st.caption("Top 3 Confidences:")
                for i in range(top_prob.size(0)):
                    st.write(f"- {CLASSES[top_catid[i]]}: {top_prob[i].item():.2f}%")