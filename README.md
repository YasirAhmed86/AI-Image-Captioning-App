# AI-Image-Captioning-App
This project is a Streamlit-based web application that generates AI-powered captions for uploaded images. It utilizes ViT-GPT2 for image captioning and ResNet-50 for optional feature extraction.

**Features**
Upload an image (JPG, PNG, JPEG)
Generate AI-generated captions using a Vision-Transformer and GPT-2 model
Extract image features using ResNet-50
User-friendly Streamlit interface
Tech Stack

PyTorch (torch, torchvision)
Hugging Face Transformers (transformers)
Streamlit (for UI)
PIL (for image processing)
NumPy

Installation
Clone the Repository
git clone https://github.com/your-repo/image-captioning.git
cd image-captioning

Install Dependencies
pip install torch torchvision transformers streamlit pillow numpy

Run the Application
streamlit run project.py

#How It Works
Upload an image via the Streamlit interface
Feature extraction (optional) using ResNet-50
Generate captions with ViT-GPT2
View the AI-generated caption on the UI

**Model Details**
ResNet-50: Used for feature extraction (removes the last classification layer)
ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning): Pre-trained model for image captioning

**Example Output**
Input Image	Generated Caption
"A cat sitting on a wooden table looking at the camera."

**Future Enhancements**
Support for real-time webcam uploads
Multiple caption generation options
Multilingual captions

**Acknowledgments**
Hugging Face for their pre-trained ViT-GPT2 model
PyTorch for deep learning support
