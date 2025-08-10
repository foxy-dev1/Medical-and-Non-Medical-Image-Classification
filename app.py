import streamlit as st
import fitz
import requests
import os
import requests.compat
import torchvision.models as models
import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from io import BytesIO
from playwright.sync_api import sync_playwright
import time
from urllib.parse import urljoin


st.title("Medical vs Non Medical Image Classification")
st.header("Upload pdf or Give URL")

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
     try:
          vgg_model = models.vgg16(weights=None)
          in_features = vgg_model.classifier[-1].in_features
          vgg_model.classifier[-1] = nn.Linear(in_features,5)
          vgg_model.load_state_dict(torch.load("best_model (2).pth"))
          vgg_model.eval()
          return vgg_model
     except Exception as e:
          st.error(f"Error loading model: {e}")
          return None
    
if "images" not in st.session_state:
     st.session_state["images"] = set()

if "model" not in st.session_state:
     st.session_state["model"] = load_model().to(device)

vgg_model = st.session_state["model"]

input_pdf = st.file_uploader("choose pdf file",type="pdf")
input_url = st.text_input("Enter url")


def transform_image(img_bytes):
    test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],
                                                    [0.229,0.224,0.225])]        
                        )
    try:
        if isinstance(img_bytes, bytes):
            image = Image.open(BytesIO(img_bytes))
        else:
            image = Image.open(img_bytes) 
        
        image = test_transform(image)
        return image
    
    except Exception as e:
        st.error(f"Error Transforming image: {e}")
        return None
    



def get_images_from_pdf(binary_data):

    doc = fitz.open(stream=binary_data,filetype="pdf")
    no_images = 0
    for i in range(len(doc)):
        page = doc[i]
        images = page.get_images(full=True)
        if images != []:
             no_images+= len(images)
        for img in images:
            xref = img[0]
            img_data = fitz.Pixmap(doc,xref)
            png_bytes = img_data.tobytes("png")
            st.session_state["images"].add(png_bytes)
    st.write(f"found {no_images} images")



def get_images_from_url(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url,wait_until="domcontentloaded",timeout=30000)

        scroll_height = page.evaluate("() => document.body.scrollHeight")
        current_position = 0
        while current_position < scroll_height:
            page.mouse.wheel(0, 1000)
            time.sleep(1)
            current_position += 1000
            scroll_height = page.evaluate("() => document.body.scrollHeight")


        image_urls = []
        for img in page.query_selector_all("img"):
            src = img.get_attribute("src") or img.get_attribute("data-src")
            if src and src not in image_urls:
                full_url = urljoin(url,src)
                if full_url not in image_urls:
                    image_urls.append(full_url)

        for image_url in image_urls:
            if image_url.lower().endswith(".svg"):
                continue

            try:
                file_name = os.path.basename(image_url)

                res = requests.get(image_url,timeout=6)
                if res.status_code == 200 and "image" in res.headers.get("Content-Type",""):
                    st.session_state["images"].add(res.content)
              

            except Exception as e:
                print(f"Error loading {image_url}: {e}")

        browser.close()

classes = ['Medical_Image_type_MRI',
            'Medical_Image_type_Xray',
            'Medical_Image_type_ctscan',
            'Medical_Image_type_ultrasound',
            'Non_Medical_Image']


classes_output = ["Medical Image (Possible type MRI)",
                  "Medical Image (Possible type Xray)",
                  "Medical Image (Possible type CT-Scan)",
                  "Medical Image (Possible type Ultrasound)",
                  "Non Medical Image"]


def main():
    if input_pdf and input_url:
        st.write("Use Either pdf or url only at a time")

    elif input_pdf:
            try:
                binary_data = input_pdf.getvalue()
                get_images_from_pdf(binary_data)
                if st.session_state["images"]:
                    for img_data in st.session_state["images"]:
                        col1,col2 = st.columns([1,1])
                        with col1:
                            st.image(img_data,width=300)
                        transformed_image = transform_image(img_data)
                        if device == "cuda":
                            transformed_image = transformed_image.unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = vgg_model(transformed_image)
        
                        _,pred = torch.max(output,1)
                        probs = F.softmax(output, dim=1)
                        confidence = probs[0][pred.item()].item()

                        with col2:
                            st.write(f"{classes_output[pred.item()]} , Confidence -> {confidence:.2f}")

                    st.session_state["images"].clear()
                else:
                        st.write("Failed to Predict on Images  -> images not found")
            except Exception as e:
                print(f"Error Predicting Images error-> {e}")
            
    elif input_url:
            try:
                get_images_from_url(input_url)
                
                if st.session_state["images"]:
                    for img_data in st.session_state["images"]:
                        col1,col2 = st.columns([1,1])
                        with col1:
                            st.image(img_data,width=300)
                        
                        transformed_image = transform_image(img_data)
                        if device == "cuda":
                            transformed_image = transformed_image.unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            output = vgg_model(transformed_image)
                            
                        _,pred = torch.max(output,1)
                        probs = F.softmax(output, dim=1)
                        confidence = probs[0][pred.item()].item()

                        with col2:
                            st.write(f"{classes_output[pred.item()]} , Confidence -> {confidence:.2f}")

                    st.session_state["images"].clear()

                else:
                    st.write("Failed to Predict on Images  -> images not found")

            except Exception as e:
                print(f"Error Predicting Images error-> {e}")
             
            


if __name__ == "__main__":
    with st.spinner():
        main()