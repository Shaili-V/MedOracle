import streamlit as st
from streamlit_tags import st_tags
import pandas as pd
import joblib
import numpy as np



# Load dataset useed for lookup
@st.cache_data
def load_disease_data():
    '''Load the disease information dataset from a CSV file.'''
    return pd.read_csv('data/disease_info.csv')
disease_data = load_disease_data()

# Load the trained model
@st.cache_resource
def load_model():
    '''Load the pre-trained Random Forest model from a joblib file.'''
    return joblib.load('random_forest_model.joblib')

model = load_model()


# Helper function to get disease information
def get_disease_info(disease_name):
    '''Look up and return the description, medication, severity, and contagiousness
      for a given disease name from the disease_data DataFrame.'''
    row = disease_data[disease_data['Disease'].str.lower() == disease_name.lower()]
    if not row.empty:
        return {
        'description' : row.iloc[0]["Description"],
        'medication' : row.iloc[0]["Medication"],
        'severity': row.iloc[0]["Severity"],
        'contagious': row.iloc[0]["Contagious"] }
    else:
        return None
    
# Callback function to handle button clicks
def make_callback(disease_name):
    def callback(d=disease_name):
        st.session_state.selected_disease = d
        st.session_state.show_disease_info = True
    return callback

def go_back():
    st.session_state.show_disease_info = False
    st.session_state.selected_disease = None

# Set page configuration
st.set_page_config(
    page_title="MedOracle: Symptom-Based Disease Predictor",
    page_icon="🩺",
    layout="wide"
)
    
# Load symptom names from the CSV file into a list
@st.cache_data
def load_symptoms():
    '''Load the list of symptoms from the SympScan dataset CSV file.'''
    return pd.read_csv('data/SympScan.csv').columns[1:].tolist()
symptoms = load_symptoms()

# UI (title and autocomplete descriptin)
st.title("MedOracle: Symptom-Based Disease Predictor")
st.markdown("Describe your symptoms using the autocomplete below.")
# Autocomplete input for symptoms using streamlit-tags
selected_symptoms = st_tags(
    label = "Enter Symptoms:",
    text = "Type and press enter",
    value = [],
    suggestions = symptoms,
    maxtags = 30,
    key = "symptom_input"
)

# Initialize session state keys if they don't exist
if 'show_disease_info' not in st.session_state:
    st.session_state.show_disease_info = False
if 'selected_disease' not in st.session_state:
    st.session_state.selected_disease = None
# If in normal prediction mode, show top 3 diseases
if not st.session_state.show_disease_info:
    # Predict button
    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("Please enter at least one symptom.")
        else:
            with st.spinner("Predicting diseases..."):
                # Create the binary input vector
                input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
                # Predict probabilities for all disease classes
                probabilities = model.predict_proba([input_vector])[0]
                # Get top 3 predictions and their probabilities
                top_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]
                top_diseases = [(model.classes_[i], probabilities[i]) for i in top_indices]
                st.session_state.top_diseases = top_diseases
    top_diseases = st.session_state.get("top_diseases",[])
    if top_diseases:
        st.subheader("Top 3 Predicted Diseases:")
        cols = st.columns(3) # Create 3 columns
        for i, (disease, prob) in enumerate(top_diseases):
            with cols[i]:
                st.markdown(f"<u><b>{disease}</b></u>", unsafe_allow_html=True)
                st.markdown(f"Confidence: {prob:.2%}")
                if st.button("View Info", key=f"view_{i}", on_click=make_callback(disease)):
                    st.session_state.selected_disease = disease
                    st.session_state.show_disease_info = True
          #  st.button(f"{disease} -- Confidence: ({prob:.2%})", on_click=make_callback(disease), key=disease)
else: # If in disease info mode, show details for the selected disease
    disease_name = st.session_state.selected_disease
    info = get_disease_info(disease_name)
    if info:
        with st.spinner("Loading disease information..."):
            #title
            st.header(disease_name)

            # Color- cded severity badge
            severity = info['severity'].lower()
            severity_colors = {
                "mild": "🟢 Mild-- Self-care",
                "moderate": "🟡 Moderate-- Routine visit",
                "urgent": "🟠 Urgent-- Same-day visit",
                "emergency": "🔴 Emergency-- Go to ER"
            }
            severity_tag = severity_colors.get(severity, severity.title())

            # Contagiousness badge
            contagious_icon = "🟢 Not Contagious"
            if info['contagious'].lower() == "yes":
                contagious_icon = "⚠️ Contagious"
            elif info['contagious'].lower() == "sometimes":
                contagious_icon = "🟡 Sometimes Contagious"

            # display info
            st.markdown(f"**📝 Description:** {info['description']}")
            st.markdown(f"**💊 Medication:** {info['medication']}")
            st.markdown(f"**🚨 Severity:** {severity_tag}")
            st.markdown(f"**🦠 Contagiousness:** {contagious_icon}")
    else:
        st.error("No information found for this disease.")
    
    if st.button("Back to Predictions"):
        with st.spinner("Returning to predictions..."):
            go_back()


# Footer
st.markdown("---")
st.markdown("Created by Shaili Vemuri")

# Side bar
with st.sidebar:
    st.title("MedOracle: Symptom-Based Disease Predictor")
    st.markdown("Welcome! This AI tool predicts possible diseases based on symptoms.")

    with st.expander("How to Use"):
        st.markdown("Enter your symptoms in the input box below. Use the autocomplete feature (tab button) to select symptoms from the list.")
        st.markdown("Click the 'Predict Disease' button to see the top 3 predicted diseases.")
        st.markdown("You can click on a disease name to see more details about it, including description, medication, severity, and contagiousness.")
        st.markdown("The model is trained on a dataset of symptoms and diseases using a Random Forest classifier.")
        st.markdown("The dataset includes 96,0000 datapoints with 237 different symptoms and 101 different diseases.")
    
    with st.expander("🔍 View All Available Symptoms"):
        st.markdown("\n".join([f"- {s}" for s in sorted(symptoms)]))  

    with st.expander("Triage Severity Key"):
        st.markdown("🔴 **Emergency** — Immediate medical attention needed")
        st.markdown("🟠 **Urgent** — Seek care soon")
        st.markdown("🟡 **Moderate** — Monitor or consult a doctor")
        st.markdown("🟢 **Mild** — Home care usually sufficient")
        
    with st.expander("More Info"):
        st.markdown("**Dataset Source:** Kaggle SympScan Dataset(https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease. Used for all medical data and model training.")
        st.markdown("**Disclaimer:** This tool is for educational purposes only, not medical advice. For any health concerns, please consult a licensed medical professional.")
        st.markdown("For more information, visit https://github.com/Shaili-V/MedOracle")
    
    