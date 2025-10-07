import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Rs ABCD",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .risk-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .low-risk {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .moderate-risk {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
        color: #333333;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        color: #212529;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .disclaimer-box h4 {
        color: #856404 !important;
        margin-bottom: 0.5rem;
    }
    .disclaimer-box p {
        color: #856404 !important;
        margin: 0;
    }
    /* Ensure all text is visible with proper contrast */
    .stMarkdown p {
        color: #333333 !important;
    }
    .stMarkdown h4 {
        color: #333333 !important;
    }
    /* Fix for metrics display */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    [data-testid="metric-container"] > div {
        color: #000000 !important;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #000000 !important;
        font-weight: bold;
    }
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: #555555 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare the model
@st.cache_data
def load_data():
    """Load the CVD risk dataset"""
    try:
        data = pd.read_csv('CVDRisk_final_ML.csv')
        return data
    except FileNotFoundError:
        st.error("Dataset file 'CVDRisk_final_ML.csv' not found. Please ensure the file is in the same directory.")
        return None

@st.cache_resource
def train_model():
    """Train the Gradient Boosting model with PSO-selected features"""
    data = load_data()
    if data is None:
        return None, None, None, None, None
    
    # Remove participant ID
    data = data.drop(columns=['Participant_ID'])
    
    # Define variables
    target_variable = 'CVD_Outcome'
    categorical_variables = [
        'DMT1', 'Ethnicity', 'Migraine', 'Sex', 'Smoking_Status_Qrisk3_1', 
        'CKD', 'AF', 'SLE', 'DMT2', 'Severe_mental_illness', 'RA', 
        '1st_degree_relatives', 'BP_med', 'psychotic_med', 
        'Erectile_dysfunction_med', 'steroid_med'
    ]
    numeric_variables = [
        'Age_at_recruitment', 'BMI_1', 'SBP_automated_reading', 'Cholesterol/HDL_ratio'
    ]
    
    # Encode categorical variables
    le_dict = {}
    for col in categorical_variables:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le
    
    # Encode target variable
    le_target = LabelEncoder()
    data[target_variable] = le_target.fit_transform(data[target_variable])
    
    # Split features and target
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numeric variables
    scaler = StandardScaler()
    X_train[numeric_variables] = scaler.fit_transform(X_train[numeric_variables])
    X_test[numeric_variables] = scaler.transform(X_test[numeric_variables])
    
    # Train Gradient Boosting model
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    
    # Get predictions for evaluation
    y_pred = gb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return gb_model, scaler, le_dict, le_target, accuracy

def get_risk_interpretation(probability):
    """Interpret the risk probability"""
    if probability < 0.3:
        return "Low Risk", "low-risk"
    elif probability < 0.7:
        return "Moderate Risk", "moderate-risk"
    else:
        return "High Risk", "high-risk"

def main():
    # Header
    st.markdown('<h1 class="main-header">Sistem ABCD</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Rumah Sakit ABCD</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, le_dict, le_target, accuracy = train_model()
    
    if model is None:
        st.error("Unable to load the model. Please check the dataset file.")
        return
    
    # Display model performance
    st.sidebar.markdown("### Model Information")
    st.sidebar.info(f"**Model**: Gradient Boosting Classifier\n**Accuracy**: {accuracy:.1%}")
    
    # Create two columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="sub-header">Basic Information</h3>', unsafe_allow_html=True)
        
        # Gender
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        # Age
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
        
        # Ethnicity/Race
        ethnicity_options = ["white or not stated", "Black caribbean", "Indian", "Pakistani", "Black african", "Other ethnic group"]
        ethnicity = st.selectbox("Ethnicity", ethnicity_options)
        
        # BMI
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        
        # Systolic Blood Pressure
        sbp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)
        
        # Cholesterol/HDL ratio
        chol_hdl_ratio = st.number_input("Cholesterol/HDL Ratio", min_value=1.0, max_value=15.0, value=4.0, step=0.1)
    
    with col2:
        st.markdown('<h3 class="sub-header">Risk Factors</h3>', unsafe_allow_html=True)
        
        # Smoking Status
        smoking_options = ["non-smoker", "ex-smoker", "light smoker", "moderate smoker", "heavy smoker"]
        smoking = st.selectbox("Smoking Status", smoking_options)
        
        # Medical Conditions
        st.markdown("**Medical Conditions:**")
        diabetes_t1 = st.checkbox("Type 1 Diabetes (DMT1)")
        diabetes_t2 = st.checkbox("Type 2 Diabetes (DMT2)")
        ckd = st.checkbox("Chronic Kidney Disease (CKD)")
        af = st.checkbox("Atrial Fibrillation (AF)")
        migraine = st.checkbox("Migraine")
        sle = st.checkbox("Systemic Lupus Erythematosus (SLE)")
        mental_illness = st.checkbox("Severe Mental Illness")
        ra = st.checkbox("Rheumatoid Arthritis (RA)")
        
        # Family History & Medications
        st.markdown("**Family History & Medications:**")
        family_history = st.checkbox("1st Degree Relatives with CVD")
        bp_med = st.checkbox("Blood Pressure Medication")
        psychotic_med = st.checkbox("Psychotic Medication")
        ed_med = st.checkbox("Erectile Dysfunction Medication")
        steroid_med = st.checkbox("Steroid Medication")
    
    # Calculate Risk Button
    if st.button("Calculate CVD Risk", type="primary", use_container_width=True):
        try:
            # Prepare input data
            input_data = {
                'Age_at_recruitment': age,
                'Sex': gender,
                'BMI_1': bmi,
                'SBP_automated_reading': sbp,
                'Cholesterol/HDL_ratio': chol_hdl_ratio,
                'Smoking_Status_Qrisk3_1': smoking,
                'Ethnicity': ethnicity,
                'CKD': 'yes' if ckd else 'no',
                'AF': 'yes' if af else 'no',
                'Migraine': 'yes' if migraine else 'no',
                'SLE': 'yes' if sle else 'no',
                'DMT1': 'yes' if diabetes_t1 else 'no',
                'DMT2': 'yes' if diabetes_t2 else 'no',
                'Severe_mental_illness': 'yes' if mental_illness else 'no',
                'RA': 'yes' if ra else 'no',
                '1st_degree_relatives': 'yes' if family_history else 'no',
                'BP_med': 'yes' if bp_med else 'no',
                'psychotic_med': 'yes' if psychotic_med else 'no',
                'Erectile_dysfunction_med': 'yes' if ed_med else 'no',
                'steroid_med': 'yes' if steroid_med else 'no'
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            categorical_variables = [
                'DMT1', 'Ethnicity', 'Migraine', 'Sex', 'Smoking_Status_Qrisk3_1', 
                'CKD', 'AF', 'SLE', 'DMT2', 'Severe_mental_illness', 'RA', 
                '1st_degree_relatives', 'BP_med', 'psychotic_med', 
                'Erectile_dysfunction_med', 'steroid_med'
            ]
            
            for col in categorical_variables:
                if col in le_dict:
                    try:
                        input_df[col] = le_dict[col].transform(input_df[col])
                    except ValueError:
                        # Handle unseen categories
                        input_df[col] = 0
            
            # Scale numeric variables
            numeric_variables = ['Age_at_recruitment', 'BMI_1', 'SBP_automated_reading', 'Cholesterol/HDL_ratio']
            input_df[numeric_variables] = scaler.transform(input_df[numeric_variables])
            
            # Make prediction
            risk_probability = model.predict_proba(input_df)[0][1]
            risk_category, risk_class = get_risk_interpretation(risk_probability)
            
            # Display results
            st.markdown("---")
            st.markdown('<h3 class="sub-header">Risk Assessment Results</h3>', unsafe_allow_html=True)
            
            # Risk probability
            st.markdown(f"""
            <div class="risk-box {risk_class}">
                <h2>{risk_category}</h2>
                <p>Risk Probability: {risk_probability:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            st.progress(risk_probability)
            
            # Risk interpretation
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{risk_probability:.1%}")
            with col2:
                st.metric("Risk Level", risk_category)
            with col3:
                if risk_probability < 0.3:
                    st.metric("Recommendation", "Maintain healthy lifestyle")
                elif risk_probability < 0.7:
                    st.metric("Recommendation", "Consult healthcare provider")
                else:
                    st.metric("Recommendation", "Immediate medical attention")
            
            # Additional information
            st.markdown("""
            <div class="disclaimer-box">
                <h4>⚠️ Important Disclaimer</h4>
                <p>This tool is for educational purposes only and should not replace professional medical advice. 
                Please consult with a healthcare provider for proper cardiovascular risk assessment and treatment recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during calculation: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Developed By ABCD Team</p>
        <p>2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()