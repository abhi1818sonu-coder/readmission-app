import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
)

DATA_PATH = "traindata_v3.xlsx"

CATEGORICAL_OPTIONS = {
    "encounter_type": ["inpatient", "emergency"],
    "discharge_disposition": ["home", "referred", "nursing_facility"],
    "creatinine_flag": ["abnormal", "not_measured", "normal"],
    "heart_rate_flag": ["abnormal", "not_measured", "normal"],
    "hemoglobin_flag": ["abnormal", "not_measured", "normal"],
    "spo2_flag": ["abnormal", "not_measured"],
    "systolic_bp_flag": ["abnormal", "not_measured"],
}

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main > div { padding-top: 1.5rem; }
    .risk-card {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .risk-high { background: rgba(255, 75, 75, 0.12); border-color: rgba(255,75,75,0.4); }
    .risk-low  { background: rgba(46, 204, 113, 0.12); border-color: rgba(46,204,113,0.4); }
    .risk-label { font-size: 0.85rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.04em; }
    .risk-value { font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }
    .model-name { font-size: 1rem; font-weight: 600; margin-bottom: 0.3rem; }
    .footer-note {
        font-size: 0.8rem;
        opacity: 0.65;
        border-top: 1px solid rgba(255,255,255,0.1);
        padding-top: 0.75rem;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Load + train (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_excel(DATA_PATH)
    df = df.drop(columns=["encounter_id", "patient_id"])

    y = df["readmission_within_30days"]
    X_raw = df.drop(columns=["readmission_within_30days"])

    cat_cols = X_raw.select_dtypes(include="object").columns.tolist()
    X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    train_columns = X.columns.tolist()

    loo = LeaveOneOut()

    log_reg = LogisticRegression(max_iter=1000)
    lr_loo_pred = cross_val_predict(log_reg, X, y, cv=loo)
    lr_loo_acc = accuracy_score(y, lr_loo_pred)
    log_reg.fit(X, y)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_loo_pred = cross_val_predict(rf, X, y, cv=loo)
    rf_loo_acc = accuracy_score(y, rf_loo_pred)
    rf.fit(X, y)

    return log_reg, rf, train_columns, cat_cols, lr_loo_acc, rf_loo_acc, len(df)


log_reg, rf, train_columns, cat_cols, lr_loo_acc, rf_loo_acc, n_records = load_and_train()


def gauge(prob, title):
    color = "#ff4b4b" if prob >= 0.5 else "#2ecc71"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "rgba(46,204,113,0.15)"},
                    {"range": [50, 100], "color": "rgba(255,75,75,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🏥 Patient Readmission Risk Predictor")
st.caption(
    f"Trained on {n_records} historical encounters. Enter details in the sidebar "
    "and click **Predict** to see the risk from two different models side by side."
)

with st.expander("ℹ️ About this tool & model performance", expanded=False):
    m1, m2 = st.columns(2)
    m1.metric("Logistic Regression — LOOCV accuracy", f"{lr_loo_acc:.0%}")
    m2.metric("Random Forest — LOOCV accuracy", f"{rf_loo_acc:.0%}")
    st.markdown(
        f"- Trained on **{n_records} records** — a small dataset, so treat "
        "predictions as illustrative rather than clinically reliable.\n"
        "- Accuracy is estimated using Leave-One-Out Cross-Validation (LOOCV), "
        "the standard approach when there isn't enough data for a normal train/test split.\n"
        "- Two models are shown for cross-verification: if they disagree, "
        "that's a signal the case is genuinely ambiguous."
    )

st.divider()

# ----------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Patient Details")

    st.subheader("Encounter")
    encounter_type = st.selectbox("Encounter type", CATEGORICAL_OPTIONS["encounter_type"])
    age = st.slider("Age", min_value=0, max_value=110, value=60)
    discharge_disposition = st.selectbox(
        "Discharge disposition", CATEGORICAL_OPTIONS["discharge_disposition"]
    )

    st.subheader("Clinical history")
    chronic_condition_count = st.slider("Chronic condition count", 0, 15, 2)
    diagnosis_count_pre_discharge = st.slider("Diagnosis count pre-discharge", 0, 15, 1)
    any_abnormality_in_observation = st.radio(
        "Any abnormality in observation?", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True
    )

    st.subheader("Lab / vitals flags")
    creatinine_flag = st.selectbox("Creatinine flag", CATEGORICAL_OPTIONS["creatinine_flag"])
    heart_rate_flag = st.selectbox("Heart rate flag", CATEGORICAL_OPTIONS["heart_rate_flag"])
    hemoglobin_flag = st.selectbox("Hemoglobin flag", CATEGORICAL_OPTIONS["hemoglobin_flag"])
    spo2_flag = st.selectbox("SpO2 flag", CATEGORICAL_OPTIONS["spo2_flag"])
    systolic_bp_flag = st.selectbox("Systolic BP flag", CATEGORICAL_OPTIONS["systolic_bp_flag"])

    st.markdown("")
    predict_clicked = st.button("🔍 Predict readmission risk", use_container_width=True, type="primary")

# ----------------------------------------------------------------------
# Prediction + results
# ----------------------------------------------------------------------
if predict_clicked:
    input_dict = {
        "encounter_type": encounter_type,
        "age": age,
        "chronic_condition_count": chronic_condition_count,
        "discharge_disposition": discharge_disposition,
        "diagnosis_count_pre_discharge": diagnosis_count_pre_discharge,
        "any_abnormality_in_observation": any_abnormality_in_observation,
        "creatinine_flag": creatinine_flag,
        "heart_rate_flag": heart_rate_flag,
        "hemoglobin_flag": hemoglobin_flag,
        "spo2_flag": spo2_flag,
        "systolic_bp_flag": systolic_bp_flag,
    }
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)

    lr_pred = log_reg.predict(input_encoded)[0]
    lr_prob = log_reg.predict_proba(input_encoded)[0][1]

    rf_pred = rf.predict(input_encoded)[0]
    rf_prob = rf.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction results")

    tab1, tab2 = st.tabs(["📊 Risk Overview", "📋 Input Summary"])

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(gauge(lr_prob, "Logistic Regression"), use_container_width=True)
            css_class = "risk-high" if lr_pred == 1 else "risk-low"
            st.markdown(
                f"""<div class="risk-card {css_class}">
                <div class="risk-label">Predicted outcome</div>
                <div class="risk-value">{"⚠️ Readmission likely" if lr_pred == 1 else "✅ Readmission unlikely"}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        with c2:
            st.plotly_chart(gauge(rf_prob, "Random Forest"), use_container_width=True)
            css_class = "risk-high" if rf_pred == 1 else "risk-low"
            st.markdown(
                f"""<div class="risk-card {css_class}">
                <div class="risk-label">Predicted outcome</div>
                <div class="risk-value">{"⚠️ Readmission likely" if rf_pred == 1 else "✅ Readmission unlikely"}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        if lr_pred != rf_pred:
            st.warning(
                "The two models disagree on this case — treat the prediction with "
                "extra caution and rely on clinical judgment."
            )

    with tab2:
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    st.markdown(
        '<div class="footer-note">This model was trained on a very small dataset '
        "(13 records). Outputs are for demonstration only and are not a substitute "
        "for clinical judgment.</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("👈 Fill in the patient details in the sidebar and click **Predict** to see results here.")
