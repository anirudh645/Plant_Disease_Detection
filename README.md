# 🌱 Plant Disease Detection System

A deep learning-based web application that uses Convolutional Neural Networks (CNN) to detect and classify plant diseases from leaf images. The system can identify 38 different plant disease categories across various crops including fruits and vegetables.

## 🚀 Live Demo
**Try the app now:** [Plant Disease Detection - Live Demo](https://anirudh645-plant-disease-detection-front-end-file-qcmzmh.streamlit.app/)

## 📋 Table of Contents
- [Live Demo](#live-demo)
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Model Performance](#model-performance)
- [Contributing](#contributing)

## 🔍 Overview

This project implements a CNN-based plant disease detection system that can classify plant leaf images into 38 different categories, including both diseased and healthy plant classifications. The system is deployed as a user-friendly web interface using Streamlit and is available online for immediate use.

## ✨ Features

- **🌐 Live Web Application**: Available online with no installation required
- **Multi-class Classification**: Identifies 38 different plant disease categories
- **User-friendly Interface**: Clean and intuitive Streamlit web application
- **Real-time Prediction**: Upload and get instant disease classification
- **Image Preprocessing**: Automatic image resizing and normalization
- **Model Caching**: Efficient model loading using Streamlit caching
- **Mobile Responsive**: Works on desktop and mobile devices
- **Comprehensive Coverage**: Supports multiple crops including:
  - Fruits: Apple, Grape, Orange, Peach, Cherry, Blueberry
  - Vegetables: Tomato, Potato, Pepper, Corn, Squash
  - Others: Strawberry, Raspberry, Soybean

## 📊 Dataset

- **Source**: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data?select=New+Plant+Diseases+Dataset%28Augmented%29)
- **Classes**: 38 different plant disease categories
- **Training Split**: 80% training, 20% validation
- **Image Size**: 224×224×3 (RGB)
- **Data Augmentation**: Rotation, normalization, and other preprocessing techniques

## 🏗️ Model Architecture

- **Algorithm**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input Shape**: (224, 224, 3)
- **Output**: 38-class classification
- **Model File**: `CNN_plantdiseases_model.keras`

### Image Preprocessing Pipeline:
1. Resize to 224×224 pixels
2. Convert BGR to RGB color space
3. Normalize pixel values (0-1 range)
4. Reshape for model input

## 🚀 Installation

### Quick Start - Use Online App
**No installation required!** Simply visit the [live demo](https://anirudh645-plant-disease-detection-front-end-file-qcmzmh.streamlit.app/) to start using the application immediately.

### Local Installation (Optional)

#### Prerequisites
- Python 3.7+
- pip package manager

#### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Plant_Disease_Detection/Deploy
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Download Model (if using Git LFS)
The model file is tracked with Git LFS. Ensure you have Git LFS installed:
```bash
git lfs pull
```

## 💻 Usage

### Online Usage (Recommended)
1. Visit [Plant Disease Detection App](https://anirudh645-plant-disease-detection-front-end-file-qcmzmh.streamlit.app/)
2. Navigate through the application using the sidebar
3. Upload a plant leaf image on the "Disease Prediction" page
4. Get instant disease classification results

### Local Usage
```bash
streamlit run front_end_file.py
```

### Using the Application
1. **Home Page**: View model information and system details
2. **Disease Prediction**: 
   - Upload a plant leaf image (JPG, JPEG, PNG)
   - Click "Show Image" to preview
   - Click "Predict" to get disease classification

### Supported Image Formats
- JPG/JPEG
- PNG

## 🌐 Deployment

The application is deployed on **Streamlit Community Cloud** and is accessible worldwide at:
[https://anirudh645-plant-disease-detection-front-end-file-qcmzmh.streamlit.app/](https://anirudh645-plant-disease-detection-front-end-file-qcmzmh.streamlit.app/)

### Deployment Features:
- **24/7 Availability**: Always accessible online
- **Automatic Updates**: Synced with repository changes
- **Scalable Infrastructure**: Handles multiple concurrent users
- **HTTPS Security**: Secure data transmission

## 📁 Project Structure

```
Plant_Disease_Detection/
├── Deploy/
│   ├── front_end_file.py          # Main Streamlit application
│   ├── CNN_plantdiseases_model.keras  # Trained CNN model
│   ├── requirements.txt           # Python dependencies
│   ├── .gitattributes            # Git LFS configuration
│   ├── images/                   # Sample images
│   └── download.jpeg             # Sample image
├── Archive.zip                   # Dataset archive
└── plant_disease_week_1_checkpoint.py  # Training notebook
```

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, PIL
- **Web Framework**: Streamlit
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Deployment**: Streamlit Community Cloud
- **Model Storage**: Git LFS for large files

## 📈 Model Performance

- **Training Data**: 80% of augmented dataset
- **Validation Data**: 20% of dataset
- **Image Augmentation**: Rotation, rescaling, and other transformations
- **Batch Size**: 164
- **Input Resolution**: 224×224 pixels

### Supported Plant Categories:
- **Apple**: Scab, Black rot, Cedar apple rust, Healthy
- **Tomato**: 8 different disease classifications + Healthy
- **Grape**: Black rot, Esca, Leaf blight, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Corn**: Multiple rust and blight classifications + Healthy
- **And many more...**

## ⚠️ Important Notes

- The model predictions may not be highly accurate due to limited training
- Ensure uploaded images are clear and well-lit for better results
- The system works best with leaf images similar to the training dataset
- For best results, use images with good lighting and minimal background noise

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Links

- **Live Demo**: [Plant Disease Detection App](https://anirudh645-plant-disease-detection-front-end-file-qcmzmh.streamlit.app/)
- [Dataset Source](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Community Cloud](https://streamlit.io/cloud)

---

*For any questions or issues, please open an issue in the repository or try the [live demo](https://anirudh645-plant-disease-detection-front-end-file-qcmzmh.streamlit.app/) first.*