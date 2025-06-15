# MedOracle: AI-Powered Symptom-Based Disease Predictor
This is a Streamlit-based AI web application that predicts diseases based on user-input symptoms using a machine learning model trained on a healthcare dataset.

---

## Try It Online
- [Launch MedOracle on Streamlit Cloud](#) *(link coming soon)*

## Features
- **Symptom Autocomplete Input**
    - Users select from 237 predefined symptoms using a convenient autofill search feature

- **AI-Powered Predictions**
    - Outputs the top 3 most likely diseases with confidence levels using a trained Random Forest model

- **Detailed Disease Information**
    - For each of the 3 predicted diseases, users can view:
        - Description
        - Common medications and treatment
        - Severitly Level (color coded with triage key)
        - Contagiousness status

- **Severity Color Coding**
    - Emergency (🔴), Urgent (🟠), Moderate (🟡), Mild (🟢)

- **Fully Responsive Interface**
    - Polished and effective UI utilizes Streamlit's titles, columns, buttons

- **Sidebar Instructions & Medical Disclaimer**
    - Enhances usability and transparency for users by explaining the application and attaching an alphabetized list of the available symptoms

---

## 📊 How It Works

1. **Symptom Selection**
   - Users select relevant symptoms via autocomplete input
2. **Prediction Engine**
   - A binary vector of symptoms is passed into a trained Random Forest model
3. **Top 3 Predictions**
   - Returned in order of probability with clickable info cards
4. **Disease Info Cards**
   - Show description, medication, severity, and contagiousness

---

## Tools Used
- **Python 3.11:** – Main programming language
- **Streamlit:** – Frontend web app framework
- **Scikit-learn:** – Machine learning model training
- **Pandas & Numpy:** – Data preprocessing and manipulation
- **VS Code:** – Development environment
- **Git & GitHub:** – Version control and collaboration
- **Streamlit Cloud:** – Hosting and deployment platform

---

## Model Details
- Algorithm: RandomForestClassifier
- Dataset: 96,000+ patient records, 237 symptoms, 101 diseases
- Accuracy:
  - Training: 94.5%
  - Testing: 86.8%
  - F1 Score: ~0.87
  - Tuned with: n_estimators=50, min_samples_split=5, min_samples_leaf=2

--- 

## Goals & Inspiration 
- Build a real-world AI tool that blends healthcare and machine learning that aims to increase accessbility to medical information by helping individuals understand the meaning of their symptoms
- Imporve skills in AI, data science, Python, and user interface design
- Learn to deploy a working prject to GitHub and Streamlit Cloud


---

## File Structure
```project-root/
├── data/
│ ├── disease_info.csv # Lookup table with descriptions, medications, severity, contagiousness
│ └── SympScan.csv # Cleaned dataset of symptoms and diseases
├── .gitignore # Specifies files/folders to ignore in version control
├── app.py # Main Streamlit application file
├── model.py # Code for training and tuning the Random Forest model
├── random_forest_model.joblib # Saved trained model for prediction
├── requirements.txt # List of dependencies for running the app
└── README.md # Project documentation
```

---

## Acknowledgments
- Used SympScan.csv and disease_info.csv (combined from multiple  datasets) sourced from Kaggle SympScan Dataset(https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease)




