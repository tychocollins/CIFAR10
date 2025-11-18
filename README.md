**🧠 PyTorch Deep Learning Image Classifier (CIFAR-10)**

This project implements a high-performance Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The entire machine learning pipeline—from model training and optimization to public web deployment—is complete.

**🚀 Live Demo**

You can interact with the live model and test its performance by uploading your own images (cars, dogs, planes, etc.):
🔗 cifar101.streamlit.app✨ 


Project Highlights & PerformanceThis model achieved a significant performance boost through architectural engineering and optimization:MetricDetailFinal Test Accuracy83% (Massive jump from the initial 60% baseline)Deployment PlatformStreamlit Cloud (Fully self-hosted web app)Technology StackPyTorch, Streamlit, Python 3.11Optimization TechniquesData Augmentation, Batch Normalization, Adam OptimizerModel ArchitectureCustom 3-Layer Deep CNN🏗️ 


Technical Architecture (V2: High Accuracy)The final architecture that achieved 83% accuracy uses modern deep learning components, significantly enhancing the model's ability to extract complex features:Model Components (Net Class)Deep Convolutional Structure: Three sequential convolutional blocks (conv1, conv2, conv3) with increased channel depth (32 -> 64 -> 128) to extract richer features.Batch Normalization: nn.BatchNorm2d layers were implemented after each convolution to stabilize training and accelerate convergence.Activation & Pooling: ReLU activation functions and Max Pooling layers were used to introduce non-linearity and downsample feature maps.Fully Connected Layers: Two fully connected layers (fc1, fc2) map the 8,192 features from the convolutional stack down to the 10 final classes.Training OptimizationsThe training script (simple_nn.py) was optimized using:Optimizer: Switched from basic SGD to the Adam optimizer for faster and more stable convergence.

Data Augmentation: Implemented transforms.RandomCrop and transforms.RandomHorizontalFlip to artificially expand the training data, making the model more robust to variations in input images.💻 Setup and Run LocallyPrerequisitesPython 3.11+Git LFS (needed for the large cifar_net.pth file)Anaconda/Miniconda (recommended for environment management)InstallationClone the repository:Bashgit clone https://github.com/tychocollins/CIFAR10.gitcdCIFAR10

Create and activate the environment:Bashconda create -n pytorch_env python=3.11
conda activate pytorch_env

Install dependencies:(You will need to ensure your requirements.txt lists PyTorch, torchvision, and streamlit)Bashpip install -r requirements.txt
Run the Live App LocallyTo test the application locally without the internet, ensure you have the cifar_net.pth file downloaded:Bashstreamlit run app.py

🏆 Deployment & Debugging (MLOps)The deployment phase required successfully overcoming several technical obstacles, showcasing robust troubleshooting skills:Model/Architecture Synchronization: Resolved persistent RuntimeError by ensuring the Net class was identical in simple_nn.py, test_classifier.py, and the deployed app.py.Git LFS Handling: Successfully managed the large 162 MB model weights file (cifar_net.pth) using Git LFS to bypass standard GitHub size limits.Environment Configuration: Debugged ModuleNotFoundError by manually configuring the Streamlit Cloud environment to support the required PyTorch version.Git Merge Resolution: Successfully resolved complex "non-fast-forward" and "unfinished merge" errors to synchronize local and remote repositories during the final deployment phase.

