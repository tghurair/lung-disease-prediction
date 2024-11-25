
# Lung Disease X-Ray Analyzer 🫁

An AI-powered web application for analyzing chest X-rays and detecting various lung conditions using deep learning.

## Features

- Real-time analysis of chest X-ray images
- Detection of 5 conditions:
  - COVID-19
  - Viral Pneumonia  
  - Tuberculosis
  - Bacterial Pneumonia
  - Normal chest X-rays
- Interactive web interface with sample images
- Detailed analysis results with confidence scores
- Medical descriptions for each condition

## Technology Stack 🔧

- PyTorch - Deep learning framework
- MobileNetV3 - Neural network architecture
- Streamlit - Web interface
- Python - Programming language

## Installation

1. Clone the repository:
```
bash
git clone https://github.com/yourusername/lung-disease.git
cd lung-disease
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the Streamlit app:
```
streamlit run app.py
```

## Usage

1. Open the web app in your browser: [http://localhost:8501](http://localhost:8501)
2. Upload a chest X-ray image
3. Get instant analysis and results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer ⚕️

This tool is for educational and research purposes only. It should not be used for medical diagnosis. Always consult healthcare professionals for medical advice.

## Acknowledgments

- Dataset: [Lung Disease Dataset on Kaggle](https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types)
- PyTorch team for the MobileNetV3 implementation

