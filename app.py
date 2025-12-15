import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, DEVICE

st.set_page_config(page_title="ArchClassifier")

@st.cache_resource
def load_model():
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    
    try:
        model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    except FileNotFoundError:
        st.error("'models/best_model.pth' файлы табылмады! Алдымен train.py іске қосыңыз.")
        return None
        
    model = model.to(DEVICE)
    model.eval()
    return model

def process_image(image):
    transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image_np = np.array(image)
    augmented = transform(image=image_np)["image"]
    return augmented.unsqueeze(0)

st.title("Архитектуралық стильдер классификаторы")
st.write("Ғимарат суретін жүктеңіз және модель оның архитектуралық стилін анықтайды.")

model = load_model()

uploaded_file = st.file_uploader("Суретті жүктеңіз...", type=["jpg", "jpeg", "png", "webp", "avif"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Жүктелген сурет')
    
    if st.button('Архитектуралық стильді тану'):
        with st.spinner('Талдау...'):
            img_tensor = process_image(image).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            conf, predicted_idx = torch.max(probs, 0)
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            
            st.success(f"Нәтиже: **{predicted_class}** ({conf.item()*100:.1f}%)")
            
            st.write("---")
            st.write("Санат бойынша ықтималдық:")
            probs_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
            st.bar_chart(probs_dict)