import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Readmission Risk Predictor", layout="centered")

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

    return log_reg, rf, train_columns, cat_cols, lr_loo_acc, rf_loo_acc, X_raw


log_reg, rf, train_columns, cat_cols, lr_loo_acc, rf_loo_acc, X_raw = load_and_train()

st.title("Patient Readmission Risk Predictor")
st.caption(
    "Trained on a small sample (13 encounters). LOOCV accuracy is shown for "
    "reference, not as a guarantee of real-world performance."
)

col1, col2 = st.columns(2)
col1.metric("Logistic Regression (LOOCV acc.)", f"{lr_loo_acc:.2f}")
col2.metric("Random Forest (LOOCV acc.)", f"{rf_loo_acc:.2f}")

st.divider()
st.subheader("Enter patient details")

with st.form("prediction_form"):
    encounter_type = st.selectbox("Encounter type", CATEGORICAL_OPTIONS["encounter_type"])
    age = st.number_input("Age", min_value=0, max_value=120, value=60)
    chronic_condition_count = st.number_input(
        "Chronic condition count", min_value=0, max_value=20, value=2
    )
    discharge_disposition = st.selectbox(
        "Discharge disposition", CATEGORICAL_OPTIONS["discharge_disposition"]
    )
    diagnosis_count_pre_discharge = st.number_input(
        "Diagnosis count pre-discharge", min_value=0, max_value=20, value=1
    )
    any_abnormality_in_observation = st.selectbox(
        "Any abnormality in observation", [0, 1]
    )
    creatinine_flag = st.selectbox("Creatinine flag", CATEGORICAL_OPTIONS["creatinine_flag"])
    heart_rate_flag = st.selectbox("Heart rate flag", CATEGORICAL_OPTIONS["heart_rate_flag"])
    hemoglobin_flag = st.selectbox("Hemoglobin flag", CATEGORICAL_OPTIONS["hemoglobin_flag"])
    spo2_flag = st.selectbox("SpO2 flag", CATEGORICAL_OPTIONS["spo2_flag"])
    systolic_bp_flag = st.selectbox("Systolic BP flag", CATEGORICAL_OPTIONS["systolic_bp_flag"])

    submitted = st.form_submit_button("Predict readmission risk")

if submitted:
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

    # Align columns with training data (add any missing dummy columns as 0)
    input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)

    lr_pred = log_reg.predict(input_encoded)[0]
    lr_prob = log_reg.predict_proba(input_encoded)[0][1]

    rf_pred = rf.predict(input_encoded)[0]
    rf_prob = rf.predict_proba(input_encoded)[0][1]

    st.divider()
    st.subheader("Prediction results")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Logistic Regression**")
        st.write("Readmission predicted:", "Yes" if lr_pred == 1 else "No")
        st.write(f"Probability of readmission: {lr_prob:.2%}")
    with c2:
        st.markdown("**Random Forest**")
        st.write("Readmission predicted:", "Yes" if rf_pred == 1 else "No")
        st.write(f"Probability of readmission: {rf_prob:.2%}")

    st.info(
        "This model was trained on only 13 records. Treat these outputs as "
        "illustrative, not clinically reliable."
    )
