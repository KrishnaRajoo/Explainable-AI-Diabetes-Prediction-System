import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)
# =========================
# LOAD CSS
# =========================
def load_css():
    with open("styles/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()



# =========================
# SESSION STATE
# =========================

# =========================
# SPLASH SCREEN
# =========================

if "show_splash" not in st.session_state:
    st.session_state.show_splash = True

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
# SPLASH SCREEN
# =========================

if st.session_state.show_splash:

    st.markdown(
        """
        <style>
        .title{
            text-align:center;
            color:#00ff88;
            font-size:48px;
            font-weight:800;
        }

        .subtitle{
            text-align:center;
            color:white;
            font-size:20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='title'>🩺</div>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Explainable AI Diabetes Prediction System</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI Powered Healthcare Intelligence</div>", unsafe_allow_html=True)

    progress = st.progress(0)

    status = st.empty()

    messages = [
        "Loading Machine Learning Model...",
        "Initializing XGBoost...",
        "Preparing Explainable AI...",
        "Loading Dashboard...",
        "Launching Application..."
    ]

    for i in range(100):
        progress.progress(i + 1)

        if i < 20:
            status.info(messages[0])
        elif i < 40:
            status.info(messages[1])
        elif i < 60:
            status.info(messages[2])
        elif i < 80:
            status.info(messages[3])
        else:
            status.success(messages[4])

        time.sleep(0.03)

    st.session_state.show_splash = False
    st.rerun()

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🩺 Diabetes Prediction System")

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

    st.markdown("""
    <div class="hero">

    <div class="glass">
        <h1 style="text-align:center;color:#00ff88;">
            🩺 Explainable AI Diabetes Risk Prediction System
        </h1>
    </div>

    <p class="subtitle">
    Advanced Healthcare Intelligence Platform
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">

    <h4>

    This application predicts diabetes risk using
    Explainable Artificial Intelligence.

    The prediction is performed using an
    XGBoost Classifier while SHAP explains
    why the model made a particular decision.

    </h4>

    <h3>This system combines:</h3>

    <h4>✔ Machine Learning</h4>
    <h4>✔ Explainable AI (SHAP)</h4>
    <h4>✔ Patient Risk Prediction</h4>
    <h4>✔ Power BI Analytics Dashboard</h4>
    <h4>✔ Automated PDF Reporting</h4>

    </div>
    """, unsafe_allow_html=True)



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



    st.markdown("""
    <div class="glass-card">

    <h2>Purpose</h2>

    To predict diabetes risk levels using
    Explainable Artificial Intelligence.

    </div>
    """, unsafe_allow_html=True)


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
    st.markdown("""
    <div class="glass">
        <h1 style="text-align:center;color:#00ff88;">
            🩺 Diabetes Risk Prediction
        </h1>
        <p style="text-align:center;font-size:18px;">
            Enter the patient's health information below to predict
            the diabetes risk using Explainable Artificial Intelligence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="glass">
    <h2>👤 Personal Information</h2>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:

        Age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

        BMI = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=70.0,
            value=25.0
        )

        Waist_Circumference = st.number_input(
            "Waist Circumference",
            min_value=20.0,
            max_value=200.0,
            value=90.0
        )

    with c2:

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

    st.write("")

    # ======================================
    # LABORATORY TESTS
    # ======================================

    st.markdown("""
    <div class="glass">
    <h2>🧪 Laboratory Tests</h2>
    </div>
    """, unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:

        Fasting_Blood_Glucose = st.number_input(
            "Fasting Blood Glucose",
            min_value=50.0,
            max_value=400.0,
            value=100.0
        )

        HbA1c = st.number_input(
            "HbA1c",
            min_value=3.0,
            max_value=20.0,
            value=5.5
        )

        Cholesterol_LDL = st.number_input(
            "Cholesterol LDL",
            min_value=10.0,
            max_value=300.0,
            value=100.0
        )

        Cholesterol_HDL = st.number_input(
            "Cholesterol HDL",
            min_value=10.0,
            max_value=150.0,
            value=50.0
        )

    with c4:

        Blood_Pressure_Systolic = st.number_input(
            "Blood Pressure Systolic",
            min_value=70.0,
            max_value=250.0,
            value=120.0
        )

        Blood_Pressure_Diastolic = st.number_input(
            "Blood Pressure Diastolic",
            min_value=40.0,
            max_value=150.0,
            value=80.0
        )

        GGT = st.number_input(
            "GGT",
            min_value=1.0,
            max_value=500.0,
            value=30.0
        )

        Serum_Urate = st.number_input(
            "Serum Urate",
            min_value=1.0,
            max_value=20.0,
            value=5.0
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

    if st.button(
        "🧠 Predict Diabetes Risk",
        use_container_width=True,
        type="primary"
        ):
        prediction = model.predict(data)

        probability = model.predict_proba(data)

        st.session_state.probability = probability

        confidence = float(max(probability[0]) * 100)
        st.session_state.prediction = model.predict(data)


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

        st.markdown("---")

        st.markdown(
            "<h1 style='text-align:center;'>Prediction Result</h1>",
            unsafe_allow_html=True
        )

        risk = st.session_state.risk_result

        confidence = st.session_state.confidence

    # -----------------------------------
    # Risk Card
    # -----------------------------------

        if risk == "Low Diabetes Risk":

            color = "#00ff88"
            icon = "🟢"

        elif risk == "Moderate Diabetes Risk":

            color = "#ffb300"
            icon = "🟠"

        else:

            color = "#ff1744"
            icon = "🔴"

        st.markdown(
            f"""
            <div style="
                background:{color}20;
                border-left:8px solid {color};
                border-radius:18px;
                padding:25px;
                margin-top:15px;
                margin-bottom:25px;
            ">

            <h2>{icon} {risk}</h2>

            <h3>Prediction Confidence : {confidence:.2f}%</h3>

            </div>
            """,
            unsafe_allow_html=True
        )

 # -----------------------------------
    # Confidence Gauge
    # -----------------------------------

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,

                number={
                    "suffix":"%"
                },

                title={
                    "text":"AI Confidence Score"
                },

                gauge={

                    "axis":{
                        "range":[0,100]
                    },

                    "bar":{
                        "color":"#00ff88"
                    },

                    "steps":[

                        {
                            "range":[0,40],
                            "color":"#8B0000"
                        },

                        {
                            "range":[40,70],
                            "color":"orange"
                        },

                        {
                            "range":[70,100],
                            "color":"green"
                        }

                    ]

                }

            )
        )

        gauge.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(
            gauge,
            width="stretch"
        )
    # -----------------------------------
    # Probability Distribution
    # -----------------------------------

        st.subheader("Prediction Probability")

        probability_df = pd.DataFrame({

            "Risk Level":[
                "Low",
                "Moderate",
                "High"
            ],

            "Probability": st.session_state.probability[0]

        })


        fig = px.bar(
            probability_df,
            x="Risk Level",
            y="Probability",
            text="Probability",
            color="Risk Level",
            color_discrete_map={
                "Low": "#00ff88",
                "Moderate": "#00d46d",
                "High": "#00b45d"
            }
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            xaxis_title="",
            yaxis_title="Probability (%)"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        st.plotly_chart(fig, width="stretch")

    # -----------------------------------
    # Patient Summary
    # -----------------------------------

        st.subheader("Patient Summary")

        c1, c2 = st.columns(2)

        with c1:

            st.metric("Age", f"{Age} Years")

            st.metric("BMI", BMI)

            st.metric("HbA1c", HbA1c)

            st.metric(
                "Glucose",
                Fasting_Blood_Glucose
            )

        with c2:

            st.metric(
                "Blood Pressure",
                f"{Blood_Pressure_Systolic}/{Blood_Pressure_Diastolic}"
            )

            st.metric(
                "Smoking",
                Smoking_Status
            )

            st.metric(
                "Activity",
                Physical_Activity_Level
            )

            st.metric(
                "Family History",
                Family_History_of_Diabetes
            )


    # -----------------------------------
    # Clinical Indicators
    # -----------------------------------

        st.subheader("Clinical Indicators")

        status_col1, status_col2, status_col3 = st.columns(3)

        with status_col1:

            if BMI < 18.5:
                st.info("BMI: Underweight")
            elif BMI < 25:
                st.success("BMI: Normal")
            elif BMI < 30:
                st.warning("BMI: Overweight")
            else:
                st.error("BMI: Obese")

        with status_col2:

            if HbA1c < 5.7:
                st.success("HbA1c: Normal")
            elif HbA1c < 6.5:
                st.warning("HbA1c: Prediabetes")
            else:
                st.error("HbA1c: Diabetes Range")

        with status_col3:

            if Fasting_Blood_Glucose < 100:
                st.success("Glucose: Normal")
            elif Fasting_Blood_Glucose < 126:
                st.warning("Glucose: Prediabetes")
            else:
                st.error("Glucose: High")

    # =====================================================
    # EXPLAINABLE AI (SHAP)
    # =====================================================

        st.markdown("---")

        st.markdown("""
        <div class="glass">
        <h2 style="text-align:center;color:#00ff88;">
        🧠 Explainable AI Dashboard
        </h2>
        <p style="text-align:center;">
        Understand why the AI predicted this diabetes risk.
        </p>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # Transform Input
    # -----------------------------

        transformed_data = (
            model
            .named_steps["preprocessor"]
            .transform(data)
        )

    # -----------------------------
    # SHAP Explainer
    # -----------------------------

        explainer = shap.TreeExplainer(
            model.named_steps["classifier"]
        )

        shap_values = explainer(
            transformed_data
        )

    # -----------------------------
    # Feature Names
    # -----------------------------

        feature_names = (
            model
            .named_steps["preprocessor"]
            .get_feature_names_out()
        )

        feature_names = [
            x.replace("remainder__", "")
             .replace("cat__", "")
            for x in feature_names
        ]

        predicted_class = int(
            st.session_state.prediction[0]
        )

    # -----------------------------
    # SHAP Explanation
    # -----------------------------

        explanation = shap.Explanation(

            values=shap_values.values[
                0,
                :,
                predicted_class
            ],

            base_values=shap_values.base_values[
                0,
                predicted_class
            ],

            data=transformed_data[0],

            feature_names=feature_names

        )

    # -----------------------------
    # Waterfall Plot
    # -----------------------------

        st.subheader("SHAP Waterfall Explanation")

        fig = plt.figure(figsize=(10,6))

        shap.plots.waterfall(
            explanation,
            show=False
        )

        st.pyplot(fig)

        plt.close(fig)

    # =====================================================
    # TOP CONTRIBUTING FEATURES
    # =====================================================

        st.subheader("Top AI Contributing Features")

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Impact": abs(explanation.values)

        })

        importance = importance.sort_values(
            "Impact",
            ascending=False
        )

        top5 = importance.head(5)

        fig = px.bar(
            top5,
            x="Feature",
            y="Impact",
            text="Impact",
            color="Impact",
            color_continuous_scale=["#00ff88", "#00b45d", "#007a3d"]
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis_title="Top AI Contributing Features",
            yaxis_title="Impact Score",
            coloraxis_showscale=False
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            marker_line_color="#00ff88",
            marker_line_width=1.5
        )

        st.plotly_chart(fig, width="stretch")

# =====================================================
# FEATURE CONTRIBUTION TABLE
# =====================================================

        st.subheader("Feature Contribution Details")

        display_df = top5.copy()

        display_df["Impact"] = (
            display_df["Impact"]
            .round(4)
        )

        st.dataframe(
            display_df,
            use_container_width=True
        )

# =====================================================
# AI CLINICAL INSIGHTS
# =====================================================

        st.markdown("""
        <div class="glass">
        <h2 style="color:#00ff88;">
💡         AI Clinical Insights
        </h2>
        </div>
        """, unsafe_allow_html=True)

        insights = []

# -----------------------------
# HbA1c
# -----------------------------

        if HbA1c >= 6.5:
            insights.append(
                "🔴 HbA1c is in the diabetic range."
            )
        elif HbA1c >= 5.7:
            insights.append(
                "🟠 HbA1c indicates prediabetes."
            )
        else:
            insights.append(
                "🟢 HbA1c is within the normal range."
            )

# -----------------------------
# Glucose
# -----------------------------

        if Fasting_Blood_Glucose >= 126:
            insights.append(
                "🔴 Fasting blood glucose is significantly elevated."
            )
        elif Fasting_Blood_Glucose >= 100:
            insights.append(
                "🟠 Blood glucose is higher than normal."
            )
        else:
            insights.append(
                "🟢 Blood glucose is normal."
            )

# -----------------------------
# BMI
# -----------------------------

        if BMI >= 30:
            insights.append(
                "🔴 BMI indicates obesity, which increases diabetes risk."
            )
        elif BMI >= 25:
            insights.append(
                "🟠 BMI indicates overweight."
            )
        else:
            insights.append(
                "🟢 BMI is within the healthy range."
            )

# -----------------------------
# Blood Pressure
# -----------------------------

        if Blood_Pressure_Systolic >= 140:
            insights.append(
                "🔴 Elevated systolic blood pressure detected."
            )

# -----------------------------
# Activity
# -----------------------------

        if Physical_Activity_Level == "Low":
            insights.append(
                "🟠 Increasing physical activity may reduce future risk."
            )

# -----------------------------
# Smoking
# -----------------------------

        if Smoking_Status == "Current":
            insights.append(
                "🔴 Smoking is a significant health risk factor."
            )

# -----------------------------
# Family History
# -----------------------------

        if Family_History_of_Diabetes == 1:
            insights.append(
                "🟠 Family history increases susceptibility to diabetes."
            )

# =====================================================
# DISPLAY INSIGHTS
# =====================================================

        for item in insights:
            st.info(item)

# =====================================================
# GENERAL RECOMMENDATIONS
# =====================================================

        st.markdown("""
        <div class="glass">
        <h2 style="color:#00ff88;">
        🩺 Lifestyle Recommendations
        </h2>
        </div>
        """, unsafe_allow_html=True)

        recommendations = [

            "Maintain a balanced diet rich in vegetables and whole grains.",

            "Exercise for at least 150 minutes per week.",

            "Monitor blood glucose regularly if at risk.",

            "Maintain a healthy body weight.",

            "Avoid smoking and limit alcohol consumption.",

            "Consult a healthcare professional for medical advice."

        ]

        for rec in recommendations:
            st.success(rec)

        if st.button("Generate Report"):

            file = "Diabetes_Report.pdf"

            doc = SimpleDocTemplate(file)

            styles = getSampleStyleSheet()

            story = []

            title = Paragraph(
                "<b><font size=18>Explainable AI Diabetes Prediction Report</font></b>",
                styles["Title"]
            )

            story.append(title)

            story.append(Paragraph("<br/>", styles["Normal"]))

            story.append(
                Paragraph("<b>Prediction Result</b>", styles["Heading2"])
            )

            story.append(
                Paragraph(
                    f"<b>Predicted Risk:</b> {st.session_state.risk_result}",
                    styles["Normal"]
                )
            )

            story.append(
                Paragraph(
                    f"<b>Prediction Confidence:</b> {st.session_state.confidence:.2f}%",
                    styles["Normal"]
                )
            )

            story.append(Paragraph("<br/>", styles["Normal"]))

            story.append(
                Paragraph("<b>Patient Information</b>", styles["Heading2"])
            )

            patient_details = [

                ("Age", Age),
                ("BMI", BMI),
                ("Fasting Blood Glucose", Fasting_Blood_Glucose),
                ("HbA1c", HbA1c),
                ("Waist Circumference", Waist_Circumference),
                ("Blood Pressure Systolic", Blood_Pressure_Systolic),
                ("Blood Pressure Diastolic", Blood_Pressure_Diastolic),
                ("Cholesterol LDL", Cholesterol_LDL),
                ("Cholesterol HDL", Cholesterol_HDL),
                ("GGT", GGT),
                ("Serum Urate", Serum_Urate),
                ("Smoking Status", Smoking_Status),
                ("Physical Activity Level", Physical_Activity_Level),
                ("Family History of Diabetes", Family_History_of_Diabetes)

            ]

            for parameter, value in patient_details:

                story.append(
                    Paragraph(
                        f"<b>{parameter}:</b> {value}",
                        styles["Normal"]
                    )
                )

            story.append(Paragraph("<br/>", styles["Normal"]))

            story.append(
                Paragraph("<b>AI Interpretation</b>", styles["Heading2"])
            )

            story.append(
                Paragraph(
                    "This prediction was generated using an XGBoost Machine Learning model and explained using SHAP (Explainable Artificial Intelligence). "
                    "The confidence score represents the model's certainty for this prediction.",
                    styles["Normal"]
                )
            )

            story.append(Paragraph("<br/>", styles["Normal"]))

            story.append(
                Paragraph(
                    "<b>Disclaimer:</b> This report is intended for educational and research purposes only and should not be considered a medical diagnosis. Please consult a qualified healthcare professional for clinical decisions.",
                    styles["Normal"]
                )
            )

            doc.build(story)

            with open(file, "rb") as pdf:

                st.download_button(
                    "📄 Download PDF Report",
                    pdf,
                    file_name=file,
                    mime="application/pdf"
                )
    st.markdown("""
    <div class="footer">

    © 2026 Explainable AI Diabetes Prediction System

    Developed by Krishna Rajoo

    </div>
    """, unsafe_allow_html=True)

# =========================
# MODEL PAGE
# =========================

if page=="📊 Model Performance":

    st.markdown("""
    <div class="glass">
        <h1 style="text-align:center;color:#00ff88;">
            📊 Model Performance
        </h1>
        <p style="text-align:center;font-size:18px;">
            Check the model accuracy, precision, recall and f1 score
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.metric(
        "Accuracy",
        "97.5%"
    )
    st.title(
        "Algorithm:XGBoost Classifier"
    )
    st.metric(
        "Precision",
        "0.97"
    )
    st.metric(
        "Recall",
        "0.97"
    )
    st.metric(
        "F1 Score",
        "0.97%"
    )

    st.markdown("""
    <div class="footer">

    ©Explainable AI Diabetes Prediction System

    Developed by Krishna Rajoo

    </div>
    """, unsafe_allow_html=True)

# =========================
# FEATURE IMPORTANCE
# =========================

if page=="📈 Feature Importance":
    st.markdown("""
    <div class="glass">
        <h1 style="text-align:center;color:#00ff88;">
            📈Important Diabetes factor
        </h1>
        <p style="text-align:center;font-size:18px;">
            Check the important factors that contribute to prediction 
        </p>
    </div>
    """, unsafe_allow_html=True)
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

    fig = px.bar(
        importance,
        x="Feature",
        y="Importance",
        text="Importance",
        color="Importance",
        color_continuous_scale=[
            "#00ff88",
            "#00d46d",
            "#00b45d"
        ]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="white",
            size=14
        ),
        xaxis_title="",
        yaxis_title="Importance Score",
        coloraxis_showscale=False
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        marker_line_color="#00ff88",
        marker_line_width=1.5
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )
    top5 = importance.head(5)
    top5_sorted = top5.sort_values("Importance", ascending=True)
    fig = px.pie(
        top5_sorted,
        names="Feature",
        values="Importance",
        hole=0.6,
        color_discrete_sequence=["#00ff88", "#00d46d", "#00b45d", "#008f4c", "#005f33"]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title="Feature Contribution Distribution"
    )

    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=top5["Importance"],
        theta=top5["Feature"],
        fill='toself',
        name='Feature Impact',
        line=dict(color="#00ff88")
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                color="white"
            )
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title="Feature Impact Radar View"
    )

    st.plotly_chart(fig, width="stretch")


    st.markdown("""
    <div class="footer">

    ©Explainable AI Diabetes Prediction System

    Developed by Krishna Rajoo

    </div>
    """, unsafe_allow_html=True)

if page=="📊 Patient Analysis":

    st.markdown("""
    <div class="glass">
        <h1 style="text-align:center;color:#00ff88;">
            📊Diabetes Patient Analysis Dashboard
        </h1>
        <p style="text-align:center;font-size:18px;">
        The Power BI dashboard performs exploratory analysis
        of diabetes patients based on age groups and health factors. 
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
    <h2 style="text-align:center;color:#00ff88;">Analysis of Diabetes Patient of Different Age-group</h2>
    </div>
    """, unsafe_allow_html=True)

    st.image(
        "Dashboard_Page_1.png",
        width="stretch"
    )

    st.image(
        "Dashboard_Page_2.png",
        width="stretch"
    )
    st.markdown("""
    <div class="footer">

    ©Explainable AI Diabetes Prediction System

    Developed by Krishna Rajoo

    </div>
    """, unsafe_allow_html=True)