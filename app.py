import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Diabetes AI Prediction",
    page_icon="🩺",
    layout="wide"
)
st.markdown(
"""
<style>

.main-title{
font-size:40px;
font-weight:bold;
text-align:center;
}

.card{
padding:20px;
border-radius:15px;
background-color:#f5f5f5;
}

</style>
""",
unsafe_allow_html=True
)

# =========================
# SESSION STATE
# =========================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "risk_result" not in st.session_state:
    st.session_state.risk_result = ""

if "confidence" not in st.session_state:
    st.session_state.confidence = 0

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# =========================
# LOAD MODEL
# =========================

model = pickle.load(
    open("diabetes_xgb_model.pkl","rb")
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🩺 Diabetes AI System")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🩺 Prediction",
        "📊 Model Performance",
        "📈 Feature Importance",
        "📊 Patient Analysis"
    ]
)

# =========================
# HOME PAGE
# =========================

if page == "🏠 Home":

    st.markdown(
    """
    <h1 style='text-align:center;'>
    🩺 Explainable AI Diabetes Risk Prediction System
    </h1>
    """,
    unsafe_allow_html=True
    )


    st.markdown(
    """
    This system combines:

    ✔ Machine Learning  
    ✔ Explainable AI (SHAP)  
    ✔ Patient Risk Prediction  
    ✔ Power BI Analytics Dashboard  
    ✔ Automated PDF Reporting  

    """
    )


    col1, col2, col3 = st.columns(3)


    with col1:
        st.metric(
            "Model",
            "XGBoost"
        )


    with col2:
        st.metric(
            "Accuracy",
            "97.5%"
        )


    with col3:
        st.metric(
            "Explainability",
            "SHAP"
        )



    st.info(
    """
    Purpose:

    To predict diabetes risk levels and provide
    understandable explanations behind AI decisions.

    """
    )


    st.warning(
    """
    Disclaimer:
    This application is developed for educational and
    research purposes. It is not a medical diagnosis tool.
    """
    )

# =========================
# PREDICTION PAGE
# =========================

if page == "🩺 Prediction":
    st.title(
        "Patient Diabetes Risk Prediction"
    )

    col1,col2 = st.columns(2)

    with col1:
        Age = st.number_input(
            "Age",
            1,
            120,
            30
        )

        BMI = st.number_input(
            "BMI",
            10.0,
            70.0,
            25.0
        )

        Fasting_Blood_Glucose = st.number_input(
            "Fasting Blood Glucose",
            50.0,
            400.0,
            100.0
        )

        HbA1c = st.number_input(
            "HbA1c",
            3.0,
            20.0,
            5.5
        )

    with col2:
        Waist_Circumference = st.number_input(
            "Waist Circumference",
            20.0,
            200.0,
            90.0
        )

        Blood_Pressure_Systolic = st.number_input(
            "Blood Pressure Systolic",
            70.0,
            250.0,
            120.0
        )

        Blood_Pressure_Diastolic = st.number_input(
            "Blood Pressure Diastolic",
            40.0,
            150.0,
            80.0
        )

        Cholesterol_LDL = st.number_input(
            "Cholesterol LDL"
        )

    Cholesterol_HDL = st.number_input(
        "Cholesterol HDL"
    )

    GGT = st.number_input(
        "GGT"
    )

    Serum_Urate = st.number_input(
        "Serum Urate"
    )

    Smoking_Status = st.selectbox(
        "Smoking Status",
        [
            "Never",
            "Former",
            "Current"
        ]
    )

    Physical_Activity_Level = st.selectbox(
        "Physical Activity Level",
        [
            "Low",
            "Moderate",
            "High"
        ]
    )

    Family_History_of_Diabetes = st.selectbox(
        "Family History of Diabetes",
        [
            0,
            1
        ]
    )

    data = pd.DataFrame(
    {
    "Age":[Age],
    "BMI":[BMI],
    "Fasting_Blood_Glucose":[Fasting_Blood_Glucose],
    "HbA1c":[HbA1c],
    "Waist_Circumference":[Waist_Circumference],
    "Blood_Pressure_Systolic":[Blood_Pressure_Systolic],
    "Blood_Pressure_Diastolic":[Blood_Pressure_Diastolic],
    "Cholesterol_LDL":[Cholesterol_LDL],
    "Cholesterol_HDL":[Cholesterol_HDL],
    "GGT":[GGT],
    "Serum_Urate":[Serum_Urate],
    "Smoking_Status":[Smoking_Status],
    "Physical_Activity_Level":[Physical_Activity_Level],
    "Family_History_of_Diabetes":[Family_History_of_Diabetes]
    }
    )

    if st.button("Predict"):

        st.session_state.prediction = model.predict(data)

        prediction = st.session_state.prediction

        probability = model.predict_proba(data)

        confidence = max(probability[0])*100

        risk_mapping = {
            0:"Low Diabetes Risk",
            1:"Moderate Diabetes Risk",
            2:"High Diabetes Risk"
        }

        st.session_state.risk_result = (
            risk_mapping[prediction[0]]
        )

        st.session_state.confidence = confidence

        st.session_state.prediction_done = True

    if st.session_state.prediction_done:

        st.success(
            f"Prediction: {st.session_state.risk_result}"
        )

        st.info(
            f"Confidence: {st.session_state.confidence:.2f}%"
        )

        # Gauge
        fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=st.session_state.confidence,
            title={
                "text":"Prediction Confidence"
            },
            gauge={
                "axis":{
                    "range":[0,100]
                }
            }
        )
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        # REPORT TABLE

        st.subheader(
            "Patient Report"
        )

        report = pd.DataFrame(
        {
        "Parameter":
        [
        "Age",
        "BMI",
        "Glucose",
        "HbA1c",
        "Blood Pressure",
        "Smoking",
        "Activity"
        ],
        "Value":
        [
        str(Age),
        str(BMI),
        str(Fasting_Blood_Glucose),
        str(HbA1c),
        str(
        f"{Blood_Pressure_Systolic}/{Blood_Pressure_Diastolic}"
        ),
        Smoking_Status,
        Physical_Activity_Level
        ]
        }
        )
        st.dataframe(
            report,
            width="stretch"
        )

        # SHAP
        transformed = (
            model
            .named_steps["preprocessor"]
            .transform(data)
        )

        explainer = shap.TreeExplainer(
            model.named_steps["classifier"]
        )

        shap_values = explainer(
            transformed
        )

        feature_names = (
            model
            .named_steps["preprocessor"]
            .get_feature_names_out()
        )

        feature_names = [
        x.replace("remainder__","")
        .replace("cat__","")
        for x in feature_names
        ]

        predicted_class = int(
            st.session_state.prediction[0]
        )

        explanation = shap.Explanation(
        values=
        shap_values.values[0,:,predicted_class],
        base_values=
        shap_values.base_values[0,predicted_class],
        data=
        transformed[0],
        feature_names=
        feature_names
        )

        st.subheader(
            "Why this prediction?"
        )

        fig = plt.figure(
            figsize=(8,6)
        )

        shap.plots.waterfall(
            explanation,
            show=False
        )

        st.pyplot(fig)
        if st.button("Generate Report"):
            file="Diabetes_Report.pdf"
            doc = SimpleDocTemplate(file)
            styles=getSampleStyleSheet()
            text=f"""
            Diabetes AI Prediction Report
            Risk:
            {st.session_state.risk_result}
            Confidence:
            {st.session_state.confidence:.2f}%

            Age:
            {Age}

            BMI:
            {BMI}
            """
            doc.build(
            [
            Paragraph(
            text,
            styles["Normal"]
            )
            ]
            )
            with open(file,"rb") as f:

                st.download_button(

                "Download PDF",
                f,
                file_name=file
                )

# =========================
# MODEL PAGE
# =========================

if page=="📊 Model Performance":

    st.title(
        "Model Performance"
    )

    st.metric(
        "Accuracy",
        "97.5%"
    )

    st.write(
    """
    Algorithm:
    XGBoost Classifier

    Precision: 0.97

    Recall: 0.97

    F1 Score: 0.97

    """
    )

# =========================
# FEATURE IMPORTANCE
# =========================

if page=="📈 Feature Importance":

    st.title(
        "Important Diabetes Factors"
    )
    importance=pd.DataFrame(
    {
    "Feature":
    [
    "LDL",
    "GGT",
    "Waist",
    "Blood Pressure",
    "Age"
    ],
    "Importance":
    [
    0.13,
    0.12,
    0.12,
    0.10,
    0.09
    ]
    }
    )
    st.bar_chart(
        importance.set_index("Feature")
    )

if page=="📊 Patient Analysis":

    st.title(
    "Diabetes Patient Analysis Dashboard"
    )


    st.write(
    "Power BI dashboard analysis"
    )
    st.subheader(
    "Analysis Of Different Age Group Patients"
    )


    st.write(
    """
    The Power BI dashboard performs exploratory analysis
    of diabetes patients based on age groups and health factors.
    """
    )

    st.image(
    "Dashboard_Page_1.png"
    )
    st.image(
    "Dashboard_Page_2.png"
    )