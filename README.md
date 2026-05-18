[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20270366-blue)](https://doi.org/10.5281/zenodo.20270366)

# Disease Classification Using CNN and Feature Fusion

A comprehensive machine learning project for classifying skin lesions using Convolutional Neural Networks (CNN) combined with metadata feature fusion. This system provides an interactive web application for medical professionals to diagnose skin conditions by uploading images and answering relevant medical questions.

## 🎯 Project Overview

This project implements a hybrid approach to skin disease classification by combining:
- **Deep Learning (CNN)**: For image-based feature extraction and classification
- **Ensemble Learning**: Multiple classifiers (XGBoost, LightGBM, Random Forest) for improved accuracy
- **Feature Fusion**: Integration of image features with patient metadata for enhanced predictions

The system can classify six types of skin conditions:
- **ACK** - Actinic Keratosis
- **BCC** - Basal Cell Carcinoma
- **MEL** - Melanoma
- **NEV** - Nevus (Mole)
- **SCC** - Squamous Cell Carcinoma
- **SEK** - Seborrheic Keratosis

## ✨ Features

- **Interactive Web Interface**: User-friendly Streamlit application for easy interaction
- **Multi-Step Questionnaire**: Comprehensive patient information collection
- **Image Upload**: Support for PNG, JPG, and JPEG formats
- **Real-time Predictions**: Instant disease classification with confidence scores
- **Patient Profile Management**: Stores and displays patient information
- **Progress Tracking**: Visual progress bar for questionnaire completion
- **Ensemble Predictions**: Uses multiple ML models for robust classification

## 🛠️ Technologies Used

- **Python**
- **TensorFlow/Keras**: Deep learning framework for CNN models
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning utilities and preprocessing
- **XGBoost**: Gradient boosting classifier
- **LightGBM**: Light gradient boosting machine
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **OpenCV**: Image processing
- **Pickle/Joblib**: Model serialization

## 📋 Prerequisites

Before running this application, ensure you have:
- Python 3.7 or higher
- pip (Python package installer)
- All required model files (included in the repository)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Moham3d-3ssam/Disease-Classification-Using-CNN-and-Feature-Fusion.git
   cd Disease-Classification-Using-CNN-and-Feature-Fusion
   ```

2. **Install required dependencies**
   ```bash
   pip install streamlit tensorflow pandas numpy opencv-python scikit-learn xgboost lightgbm joblib
   ```

   Or create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install streamlit tensorflow pandas numpy opencv-python scikit-learn xgboost lightgbm joblib
   ```

## 💻 Usage

1. **Start the Streamlit application**
   ```bash
   streamlit run Web_App.py
   ```

2. **Open your web browser**
   - The application will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

3. **Use the application**
   - Upload a skin lesion image
   - Answer the medical questionnaire (21 questions about patient history and lesion characteristics)
   - Click "Submit" to get the diagnosis
   - View the prediction results with confidence score

## 📁 Project Structure

```
Disease-Classification-Using-CNN-and-Feature-Fusion/
├── Web_App.py                  # Main Streamlit application
├── metadata.csv                # Training data metadata
├── cnn_tf_model.keras         # CNN TensorFlow model
├── feature_extractor.keras    # Feature extraction model
├── final_best_model.keras     # Best performing model
├── final_model.keras          # Final trained model
├── final_tf_model.keras       # Final TensorFlow model
├── ensemble_clf.pki           # Ensemble classifier (pickle)
├── xgb_clf.pki               # XGBoost classifier
├── lgb_clf.pki               # LightGBM classifier
├── rf_clf.pki                # Random Forest classifier
├── sc.pkl                    # StandardScaler for feature normalization
├── orderLabels.pkl           # Label encoding mapping
├── x_test_ann.pkl            # Test data for ANN
└── README.md                 # Project documentation
```

## 🔬 Model Architecture

### CNN Component
- Input: 128x128 RGB images
- Architecture: Convolutional layers for automatic feature extraction
- Output: Feature vector for fusion

### Metadata Component
- Patient demographics: age, gender, race
- Medical history: skin cancer history, cancer history, smoking, drinking
- Lesion characteristics: diameter, region, elevation, changes, symptoms
- Environmental factors: sewage system, piped water access
- Region encoding: One-hot encoding for 13 body regions

### Ensemble Component
- Combines CNN features with metadata features
- Uses multiple classifiers (XGBoost, LightGBM, Random Forest)
- Voting mechanism for final prediction
- Confidence score based on classifier agreement

## 📊 Input Features

### Patient Information (Demographics)
- Name, Age, Gender, Race

### Medical History
- Smoking status
- Alcohol consumption
- Personal skin cancer history
- Family cancer history
- Pesticide exposure
- Previous biopsy

### Lesion Characteristics
- Body region (13 possible locations)
- Two diameter measurements
- Elevation status
- Growth pattern
- Symptoms: pain, bleeding, itching
- Recent changes in appearance

### Environmental Factors
- Access to sewage system
- Access to piped water

## 🎯 Disease Classifications

| Code | Disease Name | Description |
|------|-------------|-------------|
| ACK | Actinic Keratosis | Pre-cancerous skin growth caused by sun damage |
| BCC | Basal Cell Carcinoma | Most common type of skin cancer |
| MEL | Melanoma | Most dangerous type of skin cancer |
| NEV | Nevus | Common mole, usually benign |
| SCC | Squamous Cell Carcinoma | Second most common skin cancer |
| SEK | Seborrheic Keratosis | Non-cancerous skin growth |

## 🔒 Important Notes

- This system is designed as a **decision support tool** for medical professionals
- It should **not replace** professional medical diagnosis
- Always consult with qualified healthcare providers for medical decisions
- The model's predictions are based on training data and may not cover all cases

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👥 Authors

- **Moham3d-3ssam** - [GitHub Profile](https://github.com/Moham3d-3ssam)

## 🙏 Acknowledgments

- Thanks to the medical community for providing guidelines on skin lesion classification
- Dataset contributors and researchers in dermatology AI
- Open-source community for the amazing tools and libraries

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue in the GitHub repository.

---

**Disclaimer**: This tool is for educational and research purposes. Always seek professional medical advice for health concerns.
