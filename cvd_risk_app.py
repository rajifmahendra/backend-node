import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Prediksi Risiko Penyakit Jantung Koroner 10 Tahun ke Depan",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling similar to the HTML reference
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        background: linear-gradient(180deg, #0f1724 0%, #081223 100%);
        color: #ffffff !important;
    }
    
    /* Main container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Disclaimer box */
    .disclaimer-box {
        background-color: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .disclaimer-text {
        color: #000000 !important;
        font-weight: 500;
        margin: 0;
    }
    
    .disclaimer-note {
        color: #000000 !important;
        font-style: italic;
        font-size: 0.875rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Form container */
    .form-container {
        background: white;
        border-radius: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(71, 85, 105, 0.1);
        padding: 2rem;
        color: #1e293b;
    }
    
    /* Header styles */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e293b;
        line-height: 1.2;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #f43f5e;
        font-style: italic;
        font-weight: 500;
        font-size: 1rem;
        letter-spacing: 0.025em;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff !important;
        margin: 1.5rem 0 0.75rem 0;
        border-bottom: 1px solid #fee2e2;
        padding-bottom: 0.25rem;
    }
    
    .section-subtitle {
        color: #cbd5e1 !important;
        font-style: italic;
        font-size: 0.875rem;
        margin-bottom: 1rem;
    }
    
    /* Form elements */
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    
    /* Labels */
    .stSelectbox > label,
    .stNumberInput > label,
    .stTextInput > label {
        color: #ffffff !important;
        font-weight: 500 !important;
        margin-bottom: 0.25rem !important;
    }
    
    /* Make all text visible */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        color: #ffffff !important;
    }
    
    /* Section headers should be white */
    .section-header {
        color: #ffffff !important;
    }
    
    .section-subtitle {
        color: #cbd5e1 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: #dc2626 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background: #b91c1c !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Results container */
    .results-container {
        background: white;
        border-radius: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(71, 85, 105, 0.1);
        padding: 2rem;
        color: #1e293b;
        margin-top: 2rem;
    }
    
    /* Results section headers should be white */
    .results-container .section-header {
        color: #ffffff !important;
    }
    
    .results-container .section-subtitle {
        color: #cbd5e1 !important;
    }
    
    .risk-result-box {
        background-color: #fef2f2;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .risk-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #dc2626;
        margin-bottom: 0.5rem;
    }
    
    .risk-subtitle {
        font-size: 0.875rem;
        font-style: italic;
        color: #6b7280;
        margin-bottom: 1rem;
    }
    
    .risk-percentage {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        background-color: #dc2626;
        padding: 1rem 2rem;
        border-radius: 8px;
        display: inline-block;
    }
    
    /* Risk interpretation table */
    .risk-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1.5rem;
        table-layout: fixed;
    }
    
    .risk-table th {
        padding: 0.75rem;
        text-align: center;
        font-weight: 600;
        color: white;
        vertical-align: middle;
        width: 25%;
    }
    
    .risk-table td {
        padding: 0.75rem;
        text-align: justify;
        border: 1px solid #e5e7eb;
        vertical-align: top;
        width: 25%;
        background-color: #fff;
        color: #000;
    }
    
    .low-risk-bg { background-color: #10b981; }
    .borderline-risk-bg { background-color: #fbbf24; }
    .intermediate-risk-bg { background-color: #f87171; }
    .high-risk-bg { background-color: #dc2626; }
    
    .highlighted-risk {
        border: 2px solid #000 !important;
        font-size: 1.125rem !important;
    }
    
    .highlighted-recommendation {
        border: 2px solid #000 !important;
        background-color: #fef3c7 !important;
        font-size: 1.125rem !important;
        color: #000 !important;
    }
    
    /* Patient summary */
    .summary-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .summary-label {
        font-weight: 500;
        color: #374151;
    }
    
    .summary-value {
        color: #1f2937;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <p class="disclaimer-text">⚠️ Catatan: Model ini harus diisi oleh tenaga medis. Prediksi hanya berlaku untuk individu tanpa penyakit jantung koroner yang sudah ada dan yang tidak rutin mengonsumsi statin.</p>
        <p class="disclaimer-note">⚠️ Note: This model should be completed by medical professionals. The predictions are only valid for individuals without existing coronary heart disease and individuals who are not taking statins regularly.</p>
    </div>
    """, unsafe_allow_html=True)

    # Main form container
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="main-title">Prediksi Risiko Penyakit Jantung Koroner 10 Tahun ke Depan</h1>
        <p class="subtitle">Coronary Heart Disease 10-Year Risk Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, le_dict, le_target, accuracy = train_model()
    
    if model is None:
        st.error("Tidak dapat memuat model. Periksa file dataset.")
        return
    
    # Sidebar info
    st.sidebar.info(f"**Model**: Gradient Boosting\n**Akurasi**: {accuracy:.1%}")
    
    # Section 1: Data Pribadi
    st.markdown('<h3 class="section-header">Data Pribadi</h3>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Personal Information</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Usia (Age)
        age = st.number_input("Usia *", min_value=25, max_value=80, value=None, placeholder="Masukkan usia (25-80)", help="Age")
        
    with col2:
        # Jenis Kelamin (Sex)
        gender = st.selectbox("Jenis Kelamin *", ["Pilih / Select"] + ["Male", "Female"], 
                             format_func=lambda x: x if x == "Pilih / Select" else ("Laki-laki" if x == "Male" else "Perempuan"),
                             help="Sex")
    
    # Section 2: Pengukuran Klinis
    st.markdown('<h3 class="section-header">Pengukuran Klinis</h3>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Clinical Measurements</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Tinggi Badan (Height in cm)
        height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=None, placeholder="Contoh: 170",
                                help="Height in cm — used to calculate BMI")
        
    with col2:
        # Berat Badan (Weight in kg)
        weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=None, placeholder="Contoh: 70",
                                help="Weight in kg — used to calculate BMI")
        
    with col3:
        # BMI (auto-calculated)
        if height and weight and height > 0 and weight > 0:
            bmi = weight / ((height/100) ** 2)
            st.number_input("BMI (kg/m²)", value=round(bmi, 1), disabled=True, 
                           help="Automatically calculated")
        else:
            bmi = None
            st.number_input("BMI (kg/m²)", value=None, disabled=True, placeholder="Otomatis",
                           help="Automatically calculated")
    
    # Second row
    col1, col2 = st.columns(2)
    
    with col1:
        # Rasio Kolesterol / HDL
        chol_hdl_ratio = st.number_input("Rasio Kolesterol / HDL", min_value=1.0, max_value=15.0, value=None, step=0.1, placeholder="Contoh: 4.2",
                                        help="Cholesterol/HDL ratio")
        
    with col2:
        # Tekanan Sistolik
        sbp = st.number_input("Tekanan Sistolik (mmHg)", min_value=80, max_value=250, value=None, placeholder="Contoh: 120",
                             help="Systolic blood pressure")
    
    # Section 3: Riwayat Klinis & Pengobatan
    st.markdown('<h3 class="section-header">Riwayat Klinis & Pengobatan</h3>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Clinical history & treatments</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pengobatan Tekanan Darah
        bp_med_options = ["Unknown", "No", "Yes"]
        bp_med = st.selectbox("Pengobatan Tekanan Darah", bp_med_options, index=0,
                             help="Blood pressure lowering agent")
        
        # Penggunaan Antipsikotik (Atypical)
        psychotic_med_options = ["Unknown", "No", "Yes"] 
        psychotic_med = st.selectbox("Penggunaan Antipsikotik (Atypical)", psychotic_med_options, index=0,
                                    help="Atypical antipsychotic medication use")
        
        # Pengobatan Disfungsi Ereksi
        ed_med_options = ["Unknown", "No", "Yes"]
        ed_med = st.selectbox("Pengobatan Disfungsi Ereksi", ed_med_options, index=0,
                             help="Diagnosis/treatment for erectile dysfunction")
        
    with col2:
        # Migrain
        migraine_options = ["Unknown", "No", "Yes"]
        migraine = st.selectbox("Migrain", migraine_options, index=0,
                               help="Migraine (ICD10-G43)")
        
        # Penggunaan Steroid Tablet Rutin
        steroid_med_options = ["Unknown", "No", "Yes"]
        steroid_med = st.selectbox("Penggunaan Steroid Tablet Rutin", steroid_med_options, index=0,
                                  help="Regular steroid tablet use")
        
        # Riwayat Keluarga (Angina / Serangan Jantung)
        family_history_options = ["Unknown", "No", "Yes"]
        family_history = st.selectbox("Riwayat Keluarga (Angina / Serangan Jantung)", family_history_options, index=0,
                                     help="1st-degree relatives with angina/heart attack before age 60")
    
    # Section 4: Kondisi Medis
    st.markdown('<h3 class="section-header">Kondisi Medis</h3>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Medical conditions</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Diabetes Tipe 2
        diabetes_t2_options = ["Unknown", "No", "Yes"]
        diabetes_t2 = st.selectbox("Diabetes Tipe 2", diabetes_t2_options, index=0,
                                  help="Type 2 Diabetes Mellitus (ICD10-E11)")
        
        # Fibrilasi Atrium (AF)
        af_options = ["Unknown", "No", "Yes"]
        af = st.selectbox("Fibrilasi Atrium (AF)", af_options, index=0,
                         help="Atrial fibrillation (ICD10-I48)")
        
        # Penyakit Mental Berat
        mental_illness_options = ["Unknown", "No", "Yes"]
        mental_illness = st.selectbox("Penyakit Mental Berat", mental_illness_options, index=0,
                                     help="Severe mental illness (ICD10-F20/F31/F33)")
        
    with col2:
        # Gagal Ginjal Kronis (CKD)
        ckd_options = ["Unknown", "No", "Yes"]
        ckd = st.selectbox("Gagal Ginjal Kronis (CKD)", ckd_options, index=0,
                          help="Chronic kidney disease (ICD10-N18)")
        
        # Arthritis Rematoid
        ra_options = ["Unknown", "No", "Yes"]
        ra = st.selectbox("Arthritis Rematoid", ra_options, index=0,
                         help="Rheumatoid arthritis (ICD10-M05/M06)")
    
    # Section 5: Gaya Hidup
    st.markdown('<h3 class="section-header">Gaya Hidup</h3>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Lifestyle</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Merokok
        smoking_options = ["Unknown", "non-smoker", "ex-smoker", "light smoker", "moderate smoker", "heavy smoker"]
        smoking = st.selectbox("Merokok", smoking_options, index=0,
                              help="Smoking status")
        
    with col2:
        # Konsumsi Alkohol
        alcohol_options = ["Unknown", "No", "Yes"]
        alcohol = st.selectbox("Konsumsi Alkohol", alcohol_options, index=0,
                              help="Alcohol consumption")
    
    # Add missing variables for compatibility
    diabetes_t1 = "no"  # Not in the HTML form, set default
    sle = "no"  # Not in the HTML form, set default
    ethnicity = "white or not stated"  # Set default ethnicity
    
    # Calculate Risk Button
    st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
    if st.button("Hitung Risiko / Calculate Risk", type="primary", use_container_width=True):
        # Validate required fields
        missing_fields = []
        if not age:
            missing_fields.append("Usia")
        if gender == "Pilih / Select":
            missing_fields.append("Jenis Kelamin")
        if not height:
            missing_fields.append("Tinggi Badan")
        if not weight:
            missing_fields.append("Berat Badan")
        if not chol_hdl_ratio:
            missing_fields.append("Rasio Kolesterol/HDL")
        if not sbp:
            missing_fields.append("Tekanan Sistolik")
            
        if missing_fields:
            st.error(f"❌ Harap isi field berikut: {', '.join(missing_fields)}")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return
            
        try:
            # Convert dropdown values to yes/no format
            def convert_to_yesno(value):
                if value == "Yes":
                    return "yes"
                elif value == "No":
                    return "no"
                else:
                    return "no"  # Default for "Unknown"
            
            # Prepare input data
            input_data = {
                'Age_at_recruitment': age,
                'Sex': gender,
                'BMI_1': bmi,
                'SBP_automated_reading': sbp,
                'Cholesterol/HDL_ratio': chol_hdl_ratio,
                'Smoking_Status_Qrisk3_1': smoking if smoking != "Unknown" else "non-smoker",
                'Ethnicity': ethnicity,
                'CKD': convert_to_yesno(ckd),
                'AF': convert_to_yesno(af),
                'Migraine': convert_to_yesno(migraine),
                'SLE': sle,  # Default value
                'DMT1': diabetes_t1,  # Default value
                'DMT2': convert_to_yesno(diabetes_t2),
                'Severe_mental_illness': convert_to_yesno(mental_illness),
                'RA': convert_to_yesno(ra),
                '1st_degree_relatives': convert_to_yesno(family_history),
                'BP_med': convert_to_yesno(bp_med),
                'psychotic_med': convert_to_yesno(psychotic_med),
                'Erectile_dysfunction_med': convert_to_yesno(ed_med),
                'steroid_med': convert_to_yesno(steroid_med)
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
            
            # Close form container and create results container
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            
            # Results header with logo placeholder
            st.markdown("""
            <div class="header-container">
                <div style="text-align: center; margin-bottom: 1rem;">
                    <div style="width: 60px; height: 60px; background: #dc2626; border-radius: 8px; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                        <span style="color: white; font-size: 1.5rem; font-weight: bold;">❤️</span>
                    </div>
                </div>
                <h1 style="text-align: center; font-size: 1.75rem; font-weight: 600; color: #ffffff; margin: 0;">
                    Hasil Prediksi Risiko Penyakit Jantung Koroner dalam 10 Tahun ke Depan
                </h1>
                <p style="text-align: center; font-style: italic; color: #f43f5e; font-size: 1rem; margin: 0.5rem 0 0 0;">
                    10-Year Risk of Coronary Heart Disease Result
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Evaluasi Risiko section
            st.markdown('<h3 class="section-header">Evaluasi Risiko</h3>', unsafe_allow_html=True)
            st.markdown('<p class="section-subtitle">Risk Evaluation</p>', unsafe_allow_html=True)
            
            # Current date and time
            current_time = datetime.datetime.now()
            st.markdown(f"""
            <p style="color: #ffffff; font-size: 0.875rem; margin-bottom: 1.5rem;">
                <strong>Dihasilkan pada:</strong> {current_time.strftime('%d %b %Y, %H.%M')}<br>
                <em>Generated at: {current_time.strftime('%b %d %Y, %H:%M AM')}</em>
            </p>
            """, unsafe_allow_html=True)
            
            # Risk result box
            risk_percentage = risk_probability * 100
            st.markdown(f"""
            <div class="risk-result-box">
                <div class="risk-title">Prediksi Risiko Penyakit Jantung Koroner dalam 10 Tahun ke depan</div>
                <div class="risk-subtitle">10-year risk of Coronary heart disease</div>
                <div class="risk-percentage">{risk_percentage:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretasi Risiko section
            st.markdown('<h3 class="section-header">Interpretasi Risiko</h3>', unsafe_allow_html=True)
            st.markdown('<p class="section-subtitle">Risk Interpretation</p>', unsafe_allow_html=True)
            
            # Recommendation text
            st.markdown("""
            <div style="background: #64748b; color: white; padding: 0.75rem; text-align: center; font-size: 0.875rem; margin-bottom: 1rem;">
                Rekomendasi Statin berdasarkan risiko ASCVD ditujukan untuk pasien usia 40-75 tahun dengan LDL-C 70 hingga < 190 mg/dL<br>
                (1.8 hingga < 4.9 mmol/L) tanpa diabetes melitus<br><br>
                <em>Statin recommendations based on ASCVD risk is intended for patients age 40-75 years with LDL-C 70 to < 190 mg/dL<br>
                (1.8 to < 4.9 mmol/L) without diabetes mellitus</em>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk categories and recommendations
            def get_risk_category_index(prob):
                if prob < 0.05: return 0  # Low Risk (<5%)
                elif prob < 0.075: return 1  # Borderline Risk (5% to <7.5%)
                elif prob < 0.20: return 2  # Intermediate Risk (≥7.5% to <20%)
                else: return 3  # High Risk (≥20%)
            
            risk_index = get_risk_category_index(risk_probability)
            
            # Create risk table exactly like HTML reference
            st.markdown("""
            <table style="width: 100%; border-collapse: collapse; margin-top: 1rem; table-layout: fixed;">
                <thead>
                    <tr>""", unsafe_allow_html=True)
            
            # Header row with proper highlighting
            headers = [
                ("Low Risk<br>(&lt;5%)", "#10b981"),
                ("Borderline Risk<br>(5% to &lt;7.5%)", "#fbbf24"), 
                ("Intermediate Risk<br>(≥7.5% to &lt;20%)", "#f87171"),
                ("High Risk<br>(≥20%)", "#dc2626")
            ]
            
            for i, (header, bg_color) in enumerate(headers):
                border_style = "border: 2px solid #000;" if i == risk_index else "border: 1px solid #fff;"
                font_size = "font-size: 1.125rem;" if i == risk_index else "font-size: 1rem;"
                st.markdown(f'''
                    <th style="background-color: {bg_color}; color: white; padding: 0.75rem; 
                               text-align: center; font-weight: 600; vertical-align: middle; 
                               width: 25%; {border_style} {font_size}">
                        {header}
                    </th>''', unsafe_allow_html=True)
            
            st.markdown("""
                    </tr>
                </thead>
                <tbody>
                    <tr>""", unsafe_allow_html=True)
            
            # Risk recommendations
            recommendations = [
                "Tekankan gaya hidup sehat untuk mengurangi faktor risiko (class I)<br><br><em>Emphasize healthy lifestyle factors to reduce risk factors (class I)</em>",
                "Pertimbangkan statin intensitas sedang jika ada faktor risiko tambahan (class IIb)<br><br><em>Consider moderate-intensity statin if risk-enhancing factors* are present (class IIb)</em>",
                "Pertimbangkan statin intensitas sedang jika ada faktor risiko tambahan (class I)<br><br><em>Consider moderate-intensity statin if risk-enhancing factors* are present (class I)</em>",
                "Pertimbangkan statin intensitas tinggi (class I)<br><br><em>Consider high-intensity statin (class I)</em>"
            ]
            
            for i, rec in enumerate(recommendations):
                if i == risk_index:
                    st.markdown(f'''
                    <td style="padding: 0.75rem; text-align: justify; vertical-align: top;
                               border: 2px solid #000; background-color: #fef3c7; 
                               font-size: 1.125rem; width: 25%; color: #000;">
                        {rec}
                    </td>''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <td style="padding: 0.75rem; text-align: justify; vertical-align: top;
                               border: 1px solid #e5e7eb; background-color: #fff;
                               width: 25%; color: #000;">
                        {rec}
                    </td>''', unsafe_allow_html=True)
            
            st.markdown("""
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)
            
            # Patient Summary
            st.markdown('<h3 class="section-header">Ringkasan Pasien</h3>', unsafe_allow_html=True)
            st.markdown('<p class="section-subtitle">Patient Summary</p>', unsafe_allow_html=True)
            
            # Create patient summary table
            summary_data = [
                ("Usia (Age)", f"{age}"),
                ("Jenis Kelamin (Sex)", "male" if gender == "Male" else "female"),
                ("Tinggi Badan (Height in cm)", f"{height}" if 'height' in locals() else "null"),
                ("Berat Badan (Weight in kg)", f"{weight}" if 'weight' in locals() else "null"),
                ("BMI (kg/m²)", f"{bmi:.1f}" if bmi else "null"),
                ("Rasio Kolesterol / HDL (Cholesterol/HDL ratio)", f"{chol_hdl_ratio}" if chol_hdl_ratio else "null"),
                ("Tekanan Sistolik (Systolic blood pressure)", f"{sbp}" if sbp else "null"),
                ("Pengobatan Tekanan Darah (Blood pressure lowering agent)", bp_med),
                ("Migrain (Migraine)", migraine),
                ("Penggunaan Antipsikotik (Atypical antipsychotic use)", psychotic_med),
                ("Penggunaan Steroid Tablet Rutin (Regular steroid tablet use)", steroid_med),
                ("Pengobatan Disfungsi Ereksi (Erectile dysfunction treatment)", ed_med),
                ("Riwayat Keluarga (Family history angina/heart attack)", family_history),
                ("Diabetes Tipe 2 (Type 2 Diabetes Mellitus)", diabetes_t2),
                ("Gagal Ginjal Kronis (Chronic kidney disease)", ckd),
                ("Fibrilasi Atrium (Atrial fibrillation)", af),
                ("Arthritis Rematoid (Rheumatoid arthritis)", ra),
                ("Penyakit Mental Berat (Severe mental illness)", mental_illness),
                ("Merokok (Smoking status)", smoking),
                ("Konsumsi Alkohol (Alcohol consumption)", alcohol if 'alcohol' in locals() else "Unknown")
            ]
            
            for i, (label, value) in enumerate(summary_data):
                bg_color = "#f9fafb" if i % 2 == 0 else "white"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; 
                     border-bottom: 1px solid #e5e7eb; background-color: {bg_color};">
                    <span style="font-weight: 500; color: #374151;">{label}</span>
                    <span style="color: #1f2937;">{value}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Close results container
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat perhitungan: {str(e)}")
    else:
        # Close form container if no calculation
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close button div
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem 1rem; margin-top: 2rem;">
        <p style="margin: 0; font-weight: 500;">Developed By ABCD Team</p>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem;">© 2025 - Sistem CVD Risk Calculator</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()