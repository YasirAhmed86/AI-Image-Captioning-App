import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import streamlit as st
import os

# Import the needed classes from transformers
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Load pre-trained ResNet-50 model (only needed if you want to do something with these features)
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last classification layer
resnet.eval()  # Set model to evaluation mode

# Load the ViT-GPT2 model for image captioning
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def extract_image_features(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        features = resnet(image).squeeze().numpy()  # Convert to numpy array
    return features.flatten()  # Flatten for input to captioning model if needed

def generate_caption(image_path):
    # Open image
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

# Streamlit UI
st.title("AI Image Captioning")
st.write("Upload an image and get an AI-generated caption.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_path = "temp.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Uploaded Image", use_column_width=True)

    # Optional: extract features if you need them
    st.write("🔍 Extracting features using ResNet-50...")
    features = extract_image_features(image_path)

    if st.button("Generate Caption"):
        # Correct function name
        caption = generate_caption(image_path)
        st.success(f"📝 Generated Caption: {caption}")
