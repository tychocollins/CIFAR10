# 🧠 PyTorch Deep Learning Image Classifier (CIFAR-10)

This project implements a high-performance **Convolutional Neural Network (CNN)** to classify images from the CIFAR-10 dataset. The entire machine learning pipeline—from model training and optimization to public web deployment—is complete.

---

**🚀 Live Demo**

You can interact with the live model and test its performance by uploading your own images (cars, dogs, planes, etc.):

🔗 **[cifar101.streamlit.app](https://cifar101.streamlit.app)** ✨

---

**✨ Project Highlights & Performance**

This model achieved a significant performance boost through architectural engineering and optimization:

| Metric | Detail |
| :--- | :--- |
| **Final Test Accuracy** | **83%** (Massive jump from the initial 60% baseline) |
| **Deployment Platform** | Streamlit Cloud (Fully self-hosted web app) |
| **Technology Stack** | PyTorch, Streamlit, Python 3.11 |
| **Optimization Techniques** | Data Augmentation, Batch Normalization, Adam Optimizer |
| **Model Architecture** | Custom 3-Layer Deep CNN |

---

**🏗️ Technical Architecture (V2: High Accuracy)**

The final architecture that achieved 83% accuracy uses modern deep learning components, significantly enhancing the model's ability to extract complex features:

**Model Components (`Net` Class)**

* **Deep Convolutional Structure:** Three sequential convolutional blocks (`conv1`, `conv2`, `conv3`) with increased channel depth (32 -> 64 -> 128) to extract richer features.
* **Batch Normalization:** `nn.BatchNorm2d` layers were implemented after each convolution to **stabilize training** and accelerate convergence.
* **Activation & Pooling:** ReLU activation functions and Max Pooling layers were used to introduce non-linearity and downsample feature maps.
* **Fully Connected Layers:** Two fully connected layers (`fc1`, `fc2`) map the 8,192 features from the convolutional stack down to the 10 final classes.

**Training Optimizations**

The training script (`simple_nn.py`) was optimized using:

* **Optimizer:** Switched from basic SGD to the **Adam optimizer** for faster and more stable convergence.
* **Data Augmentation:** Implemented `transforms.RandomCrop` and `transforms.RandomHorizontalFlip` to artificially expand the training data, making the model more **robust** to variations in input images.

---

**💻 Setup and Run Locally**

### Prerequisites

* Python 3.11+
* Git LFS (needed for the large `cifar_net.pth` file)
* Anaconda/Miniconda (recommended for environment management)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tychocollins/CIFAR10.git](https://github.com/tychocollins/CIFAR10.git)
    cd CIFAR10
    ```

2.  **Create and activate the environment:**
    ```bash
    conda create -n pytorch_env python=3.11
    conda activate pytorch_env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Run the Live App Locally

To test the application locally without the internet, ensure you have the `cifar_net.pth` file downloaded:

```bash
streamlit run app.py
