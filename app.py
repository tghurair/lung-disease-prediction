import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# Page config
st.set_page_config(
    page_title="Lung Disease X-Ray Analyzer 🫁",
    page_icon="🔬",
    layout="wide"
)

# Constants
CLASS_NAMES = [
    'Corona Virus Disease',
    'Viral Pneumonia',
    'Tuberculosis',
    'Bacterial Pneumonia',
    'Normal'
]

# Disease descriptions for streaming
DISEASE_DESCRIPTIONS = {
    'Corona Virus Disease': """
    🦠 COVID-19 Analysis:
    • Typical signs include ground-glass opacities
    • Often bilateral and peripheral distribution
    • May show consolidation in severe cases
    """,
    'Viral Pneumonia': """
    🦠 Viral Pneumonia Analysis:
    • Interstitial patterns visible
    • Often shows bilateral involvement
    • Less dense than bacterial pneumonia
    """,
    'Tuberculosis': """
    🔬 Tuberculosis Analysis:
    • Upper lobe involvement common
    • Cavitary lesions may be present
    • Possible lymph node enlargement
    """,
    'Bacterial Pneumonia': """
    🦠 Bacterial Pneumonia Analysis:
    • Dense consolidation visible
    • Often unilateral
    • Sharp margins typical
    """,
    'Normal': """
    ✅ Normal Chest X-Ray Analysis:
    • Clear lung fields
    • Normal heart size
    • No significant abnormalities
    """
}

class LungDiseasePredictor:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = models.mobilenet_v3_small(pretrained=False)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, len(CLASS_NAMES))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image or file path")
            
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            pred_prob, pred_class = torch.max(probabilities, 1)
            
        return {
            'class': CLASS_NAMES[pred_class.item()],
            'probability': pred_prob.item(),
            'all_probabilities': {
                class_name: prob.item() 
                for class_name, prob in zip(CLASS_NAMES, probabilities[0])
            }
        }

def plot_prediction_probabilities(prediction):
    probs = prediction['all_probabilities']
    fig, ax = plt.subplots(figsize=(10, 4))
    
    bars = ax.barh(list(probs.keys()), list(probs.values()))
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Probabilities')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{width:.2%}',
                ha='left',
                va='center')
    
    plt.tight_layout()
    return fig

def create_overview_tab():
    st.title("🫁 Lung Disease X-Ray Analysis System")
    
    # Introduction
    st.header("🔬 About This Tool")
    st.write("""
    Welcome to our advanced Lung Disease Analysis System. This AI-powered tool assists in analyzing chest X-rays 
    for various lung conditions, providing rapid and accurate preliminary assessments.
    """)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📊 Global Impact")
        st.write("""
        - 4 million pneumonia cases annually
        - 1.5 million TB cases detected
        - COVID-19 affecting millions
        """)
    with col2:
        st.subheader("🎯 Detection Rate")
        st.write("""
        - 92% accuracy in testing
        - Real-time analysis
        - Instant results
        """)
    with col3:
        st.subheader("💡 AI Benefits")
        st.write("""
        - 24/7 availability
        - Consistent analysis
        - Rapid screening
        """)
    
    # How it works
    st.header("🔄 How It Works")
    cols = st.columns(4)
    cols[0].subheader("1. Upload X-Ray 📤")
    cols[1].subheader("2. AI Processing 🔄")
    cols[2].subheader("3. Analysis 🔍")
    cols[3].subheader("4. Results 📊")
    
    # Medical Resources
    st.header("📚 Medical Resources")
    st.write("""
    - [WHO - Pneumonia Information](https://www.who.int/health-topics/pneumonia)
    - [CDC - COVID-19 Resources](https://www.cdc.gov/coronavirus/2019-ncov/index.html)
    - [WHO - Tuberculosis Guidelines](https://www.who.int/health-topics/tuberculosis)
    """)

def create_prediction_tab():
    st.title("🔍 X-Ray Analysis")
    
    # Initialize session state for selected image
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    
    # Sample images section
    st.header("📸 Sample Images")
    sample_cols = st.columns(5)
    
    # Define image paths with correct extensions
    image_paths = {
        'Bacterial Pneumonia': 'sample_images/bacterial_pneumonia.jpeg',
        'Corona Virus Disease': 'sample_images/corona_virus_disease.png',
        'Normal': 'sample_images/normal.jpeg',
        'Tuberculosis': 'sample_images/tuberculosis.jpg',
        'Viral Pneumonia': 'sample_images/viral_pneumonia.jpeg'
    }
    
    # Set consistent size for all images
    target_size = (224, 224)  # Standard size for medical images
    
    for i, (col, class_name) in enumerate(zip(sample_cols, CLASS_NAMES)):
        col.subheader(class_name)
        try:
            # Open and resize image
            img = Image.open(image_paths[class_name]).convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            # Display image and add a button below it
            col.image(img, use_column_width=True)
            if col.button('Select', key=f'btn_{class_name}'):
                st.session_state.selected_image = image_paths[class_name]
                st.rerun()
    
        except Exception as e:
            col.write(f"Error loading image: {e}")
    
    st.divider()
    
    # Upload and prediction section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
        
        # Use either uploaded file or selected sample image
        image_to_analyze = None
        if uploaded_file:
            image_to_analyze = Image.open(uploaded_file).convert('RGB')
            st.image(image_to_analyze, caption="Uploaded X-Ray", use_column_width=True)
        elif st.session_state.selected_image:
            image_to_analyze = Image.open(st.session_state.selected_image).convert('RGB')
            st.image(image_to_analyze, caption="Selected Sample X-Ray", use_column_width=True)
        
        # Create two columns for buttons
        button_col1, button_col2 = st.columns([1, 1])
        
        # Clear Selection button (only show if sample image is selected)
        if st.session_state.selected_image:
            if button_col1.button('❌ Clear Selection', use_container_width=True):
                st.session_state.selected_image = None
                st.rerun()
        
        # Analyze button
        if image_to_analyze:
            if button_col2.button("🔍 Analyze Image", type="primary", use_container_width=True):
                try:
                    predictor = LungDiseasePredictor('model/lung-disease-predictor.pth')
                    
                    with st.spinner('Analyzing image...'):
                        prediction = predictor.predict(image_to_analyze)
                    
                    with col2:
                        st.header("📊 Analysis Results")
                        st.subheader(f"Prediction: {prediction['class']}")
                        st.metric("Confidence", f"{prediction['probability']:.2%}")
                        
                        st.pyplot(plot_prediction_probabilities(prediction))
                        
                        st.header("📋 Detailed Analysis")
                        # Display the description with proper formatting
                        description = DISEASE_DESCRIPTIONS[prediction['class']]
                        for line in description.split('\n'):
                            if line.strip():  # Only display non-empty lines
                                st.write(line.strip())
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    
    # Medical disclaimer
    st.divider()
    st.caption("""
    ⚕️ Medical Disclaimer: This tool is for educational purposes only and should not be used for medical diagnosis. 
    Please consult with healthcare professionals for medical advice.
    """)

def main():
    tab1, tab2 = st.tabs(["📋 Overview", "🔍 Prediction"])
    
    with tab1:
        create_overview_tab()
    
    with tab2:
        create_prediction_tab()

if __name__ == "__main__":
    main()