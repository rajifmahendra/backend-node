# CVD Risk Calculator Web Application

A web-based cardiovascular disease risk assessment tool built with Streamlit and machine learning.

## Features

- Interactive web form for inputting patient data
- Real-time CVD risk calculation using Gradient Boosting algorithm
- Risk categorization (Low, Moderate, High)
- Professional medical-style interface
- Responsive design that works on desktop and mobile

## Requirements

- Python 3.7 or higher
- Required packages (see requirements.txt)

## Installation & Setup

1. **Install Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure the dataset file is present**:
   - Make sure `CVDRisk_final_ML.csv` is in the same directory as `cvd_risk_app.py`

## Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run cvd_risk_app.py
   ```

2. **Open in browser**:
   - The app will automatically open in your default browser
   - If not, go to `http://localhost:8501`

## How to Use

1. **Fill in Basic Information**:
   - Gender (Male/Female)
   - Age (18-100 years)
   - Ethnicity
   - BMI
   - Systolic Blood Pressure
   - Cholesterol/HDL Ratio

2. **Select Risk Factors**:
   - Smoking status
   - Medical conditions (diabetes, kidney disease, etc.)
   - Family history
   - Current medications

3. **Calculate Risk**:
   - Click "Calculate CVD Risk" button
   - View your risk probability and category
   - Read recommendations

## Risk Categories

- **Low Risk** (< 30%): Maintain healthy lifestyle
- **Moderate Risk** (30-70%): Consult healthcare provider
- **High Risk** (> 70%): Immediate medical attention recommended

## Model Information

- **Algorithm**: Gradient Boosting Classifier
- **Features**: 20 risk factors including demographics, medical history, and lifestyle factors
- **Training**: Trained on cardiovascular disease dataset with feature selection optimization

## Disclaimer

⚠️ **Important**: This tool is for educational and research purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## Files Structure

```
lomba/
├── cvd_risk_app.py          # Main Streamlit application
├── CVDRisk_final_ML.csv     # Dataset file
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── Bismillah.ipynb        # Original analysis notebook
└── GradientBoosting.ipynb # Gradient boosting focused notebook
```

## Troubleshooting

1. **"Dataset file not found" error**:
   - Ensure `CVDRisk_final_ML.csv` is in the same directory as the app

2. **Package import errors**:
   - Run `pip install -r requirements.txt` to install all dependencies

3. **Port already in use**:
   - Use `streamlit run cvd_risk_app.py --server.port 8502` to use a different port

## Technical Details

The application uses:
- **Streamlit** for the web interface
- **scikit-learn** for machine learning model
- **pandas** for data manipulation
- **numpy** for numerical computations

The model is trained in real-time when the app starts, using the same preprocessing and training pipeline from the original notebooks.