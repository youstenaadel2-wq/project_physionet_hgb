import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import math
from datetime import datetime


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sepsis Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# MODEL FILES
# =========================================================

MODEL_FILE = "hgb.pkl"
IMPUTER_FILE = "Imputer_hgb.pkl"
THRESHOLD_FILE = "hgb_threshold.pkl"
FEATURES_FILE = "feature_names_hgb.pkl"

IMAGE_FILE = "doctor_patient.png"


# =========================================================
# LOAD MODEL
# =========================================================

try:
    hgb_model = joblib.load("hgb.pkl")
    imputer_hgb = joblib.load("Imputer_hgb.pkl")
    threshold = joblib.load("hgb_threshold.pkl")
    feature_names_hgb = joblib.load("feature_names_hgb.pkl")
except Exception as e:
    st.error("Could not load the HistGradientBoosting model files.")
    st.error(f"Error: {e}")
    st.info(
        "Make sure hgb.pkl, Imputer_hgb.pkl, "
        "hgb_threshold.pkl and feature_names_hgb.pkl "
        "are in the same folder as app.py."
    )
    st.stop()


# =========================================================
# MODEL FEATURES USED BY THE UI
# =========================================================

# These are the 14 user-facing features requested for the assessment.
# The two missing-indicator features are calculated automatically and
# are NOT entered manually by the user.
UI_FEATURES = [
    "Age__years",
    "Heart_rate__in_beats_per_minute",
    "Respiration_rate__breaths_per_minute",
    "Systolic_BP__mm_Hg",
    "Mean_arterial_pressure__mm_Hg",
    "Diastolic_BP__mm_Hg",
    "Pulse_oximetry__percentage",
    "Temperature__deg_C",
    "Female__0__or_male__1",
    "Administrative_identifier_for_ICU_unit__MICU",
    "Serum_glucose__mg_dL",
    "Fraction_of_inspired_oxygen__percentage",
    "missingindicator_Administrative_identifier_for_ICU_unit",
    "missingindicator_Fraction_of_inspired_oxygen",
]


def find_feature(prefix, required=True):
    """Find an exact feature or a feature beginning with a known prefix."""
    if prefix in feature_names_hgb:
        return prefix
    matches = [
        f for f in feature_names_hgb
        if str(f).startswith(prefix)
    ]
    if matches:
        return matches[0]
    if required:
        st.error(
            f"Required model feature was not found: {prefix}"
        )
        st.write("Available model features:", feature_names_hgb)
        st.stop()
    return None


# Resolve the model's actual names once. This also handles long encoded
# ICU feature names such as Administrative_identifier_for_ICU_unit__MICU__...
AGE_FEATURE = find_feature("Age__years")
HR_FEATURE = find_feature("Heart_rate__in_beats_per_minute")
RESP_FEATURE = find_feature("Respiration_rate__breaths_per_minute")
SYS_BP_FEATURE = find_feature("Systolic_BP__mm_Hg")
MAP_FEATURE = find_feature("Mean_arterial_pressure__mm_Hg")
DIA_BP_FEATURE = find_feature("Diastolic_BP__mm_Hg")
SPO2_FEATURE = find_feature("Pulse_oximetry__percentage")
TEMP_FEATURE = find_feature("Temperature__deg_C")
SEX_FEATURE = find_feature("Female__0__or_male__1")
ICU_FEATURE = find_feature(
    "Administrative_identifier_for_ICU_unit__MICU",
    required=False
)
GLUCOSE_FEATURE = find_feature("Serum_glucose__mg_dL")
FIO2_FEATURE = find_feature("Fraction_of_inspired_oxygen__percentage")
ICU_MISSING_FEATURE = find_feature(
    "missingindicator_Administrative_identifier_for_ICU_unit",
    required=False
)
FIO2_MISSING_FEATURE = find_feature(
    "missingindicator_Fraction_of_inspired_oxygen",
    required=False
)

# All 14 feature concepts shown to the user. The last two are automatic.
ui_features = [
    AGE_FEATURE,
    HR_FEATURE,
    RESP_FEATURE,
    SYS_BP_FEATURE,
    MAP_FEATURE,
    DIA_BP_FEATURE,
    SPO2_FEATURE,
    TEMP_FEATURE,
    SEX_FEATURE,
    ICU_FEATURE,
    GLUCOSE_FEATURE,
    FIO2_FEATURE,
    ICU_MISSING_FEATURE,
    FIO2_MISSING_FEATURE,
]


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "page": "Home",
    "section": "Vital Signs",

    "prediction": None,
    "probability": None,

    "records": [],

    "age": 45,
    "sex": 1,
    "icu_unit": "MICU",
    "heart_rate": 80,
    "pulse_ox": 98.0,
    "temperature": 37.0,
    "systolic_bp": 120,
    "map_bp": 80,
    "diastolic_bp": 70,
    "resp_rate": 18,
    "fio2": 21.0,
    "serum_glucose": 100.0,
}


for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================================
# CSS
# =========================================================

st.html(
        """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 90% 5%,
            rgba(99,190,238,.12),
            transparent 25%
        ),
        radial-gradient(
            circle at 5% 80%,
            rgba(117,205,239,.10),
            transparent 25%
        ),
        #f7fbff;
    color: #173f63;
}

.block-container {
    max-width: 1550px;
    padding: 18px 22px 35px;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #d9eaf5;
    min-width: 245px;
    max-width: 245px;
}

section[data-testid="stSidebar"] > div {
    padding: 18px 14px;
}

/* Sidebar navigation: keep labels on one readable line */
section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    min-height: 52px !important;
    height: auto !important;
    padding: 10px 12px !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    text-align: left !important;
}

section[data-testid="stSidebar"] .stButton {
    margin: 7px 0 !important;
}

.nav-logo {
    width: 54px;
    height: 54px;
    margin: 0 auto 24px;
    border-radius: 17px;
    background: #eef8ff;
    border: 1px solid #d8ebf7;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 27px;
    color: #0879d1;
}

.nav-button {
    margin: 7px 0;
}

.nav-button button {
    height: 64px !important;
    padding: 5px !important;

    border-radius: 16px !important;
    border: 1px solid #c5e5f7 !important;

    background: #dff2ff !important;
    color: #07549a !important;

    font-size: 10px !important;
    font-weight: 800 !important;

    box-shadow: none !important;
}

.nav-button button:hover {
    background: #c7e9ff !important;
    color: #064b88 !important;
    border-color: #8fcdf2 !important;
}

.nav-active button {
    background: #bfe5fb !important;
    color: #064b88 !important;
    font-weight: 900 !important;
    border-color: #91cbed !important;
}

.nav-footer {
    margin-top: 35px;
    text-align: center;
    font-size: 9px;
    line-height: 1.55;
    color: #7b9eb8;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    background: #dff2ff !important;
    color: #07549a !important;

    border: 1px solid #b9def5 !important;
    border-radius: 14px !important;

    font-weight: 800 !important;

    box-shadow:
        0 4px 12px rgba(8,116,201,.08) !important;
}

.stButton > button:hover {
    background: #c7e9ff !important;
    color: #064b88 !important;
    border-color: #8fcdf2 !important;
}

.stButton > button:focus,
.stButton > button:active {
    background: #b9e2fa !important;
    color: #064b88 !important;
    border-color: #75bce8 !important;

    box-shadow:
        0 0 0 2px rgba(8,116,201,.12) !important;
}

.stButton > button p {
    color: #07549a !important;
    font-weight: 800 !important;
}


/* =========================================================
   HEADER
   ========================================================= */

.header {
    min-height: 78px;

    background: #ffffff;
    border: 1px solid #d9eaf5;
    border-radius: 22px;

    padding: 12px 23px;
    margin-bottom: 14px;

    display: flex;
    align-items: center;
    justify-content: space-between;

    box-shadow:
        0 10px 30px rgba(28,113,167,.07);

    position: relative;
    overflow: hidden;
}

.header-title {
    font-size: 27px;
    font-weight: 900;
    color: #073f7c;
}

.header-sub {
    font-size: 12px;
    color: #7094af;
    margin-top: 5px;
}

.header-right {
    display: flex;
    align-items: center;
    gap: 10px;

    color: #527f9f;
    font-size: 11px;
}

.avatar {
    width: 38px;
    height: 38px;

    border-radius: 50%;
    background: #eaf6ff;
    border: 2px solid white;

    box-shadow: 0 2px 10px #cfe5f4;

    display: flex;
    align-items: center;
    justify-content: center;

    color: #1677bd;
    font-weight: 900;
}


/* =========================================================
   CARDS
   ========================================================= */

.panel,
.home-card,
.analysis-card,
.about-box {
    background: #ffffff;
    border: 1px solid #d9eaf5;
    border-radius: 22px;

    box-shadow:
        0 12px 34px rgba(32,117,169,.07);
}

.home-card {
    padding: 30px;
}

.home-title {
    font-size: 32px;
    font-weight: 900;
    color: #073f7c;
}

.home-text {
    font-size: 13px;
    color: #6688a5;
    line-height: 1.7;
    max-width: 800px;
}

.home-feature {
    padding: 18px;

    border: 1px solid #d9eaf5;
    border-radius: 16px;

    background: #ffffff;

    min-height: 130px;

    box-shadow:
        0 7px 20px rgba(32,117,169,.05);
}

.home-feature-title {
    font-weight: 900;
    color: #07549a;
    font-size: 13px;
}

.home-feature-text {
    font-size: 10px;
    color: #7595ae;
    line-height: 1.5;
    margin-top: 6px;
}


/* =========================================================
   CLINICAL
   ========================================================= */

.clinical-top {
    display: flex;
    align-items: center;
    gap: 11px;

    padding: 16px 18px;
}

.clinical-icon {
    width: 43px;
    height: 43px;

    border-radius: 14px;
    background: #eaf6ff;

    display: flex;
    align-items: center;
    justify-content: center;

    color: #0879d1;
    font-size: 22px;
}

.clinical-title {
    font-size: 21px;
    font-weight: 900;
    color: #07549a;
}

.clinical-sub {
    font-size: 10px;
    color: #7696b0;
    margin-top: 2px;
}


/* =========================================================
   STEPS
   ========================================================= */

.steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 7px;

    margin-bottom: 11px;
}

.step-button button {
    height: 61px !important;
    padding: 8px !important;

    border: 1px solid #b9def5 !important;
    border-radius: 14px !important;

    background: #dff2ff !important;
    color: #07549a !important;

    box-shadow: none !important;

    font-size: 10px !important;
    font-weight: 800 !important;
}

.step-button button:hover {
    background: #c7e9ff !important;
    border-color: #8fcdf2 !important;
}

.step-active button {
    background: #bfe5fb !important;
    border-color: #91cbed !important;
    color: #064b88 !important;
}


/* =========================================================
   SECTIONS
   ========================================================= */

.section {
    border: 1px solid #d9eaf5;
    border-radius: 17px;

    padding: 14px;
    margin-bottom: 9px;

    background: #ffffff;
}

.section.pink {
    background: #fffafd;
    border-color: #f2dce7;
}

.section.teal {
    background: #fbfffe;
    border-color: #d8eee9;
}

.section.purple {
    background: #fdfcff;
    border-color: #e5def8;
}

.section-head {
    display: flex;
    align-items: center;
    gap: 9px;

    margin-bottom: 12px;
}

.section-icon {
    width: 36px;
    height: 36px;

    border-radius: 12px;
    background: #edf8ff;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 18px;
    color: #0879d1;
}

.pink .section-icon {
    background: #fff0f6;
    color: #e94884;
}

.teal .section-icon {
    background: #eafaf7;
    color: #0da58f;
}

.purple .section-icon {
    background: #f1edff;
    color: #7054df;
}

.section-title-small {
    font-size: 14px;
    font-weight: 900;
    color: #07549a;
}

.pink .section-title-small {
    color: #d84079;
}

.teal .section-title-small {
    color: #078e7e;
}

.purple .section-title-small {
    color: #6449ca;
}

.section-desc {
    font-size: 9px;
    color: #7898b1;
    margin-top: 1px;
}


/* =========================================================
   FEATURE CARDS
   ========================================================= */

.feature-card {
    background: #ffffff;

    border: 1px solid #e2eef6;
    border-radius: 13px;

    padding: 11px;
    margin-bottom: 10px;

    box-shadow:
        0 4px 12px rgba(35,116,166,.035);
}

.feature-name {
    font-size: 10px;
    font-weight: 850;
    color: #185d91;
}

.feature-unit {
    font-size: 8px;
    color: #7898b1;
    margin-bottom: 5px;
}


/* =========================================================
   INPUTS
   ========================================================= */

div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    font-size: 14px !important;
    line-height: 1.4 !important;
    font-weight: 800 !important;
    color: #185d91 !important;
    margin-bottom: 4px !important;
}

div[data-testid="stNumberInput"] input {
    border-radius: 9px !important;
}


/* =========================================================
   ANALYSIS
   ========================================================= */

.analysis-card {
    padding: 25px;
    margin-bottom: 12px;
}

.analysis-title {
    font-size: 20px;
    font-weight: 900;
    color: #07549a;
}


/* =========================================================
   CIRCULAR RESULT
   ========================================================= */

.gauge {
    width: 210px;
    height: 210px;

    margin: 20px auto;

    border-radius: 50%;

    display: flex;
    align-items: center;
    justify-content: center;
}

.gauge-inner {
    width: 158px;
    height: 158px;

    border-radius: 50%;
    background: #ffffff;

    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;

    box-shadow:
        0 5px 20px rgba(20,110,170,.08);
}

.gauge-number {
    font-size: 31px;
    font-weight: 950;
    color: #084d8f;
}

.gauge-label {
    font-size: 10px;
    color: #7697b1;
    margin-top: 5px;
}


/* =========================================================
   RISK
   ========================================================= */

.risk {
    padding: 12px;

    border-radius: 14px;

    display: flex;
    gap: 10px;
}

.risk-high {
    background: #fff5f5;
    border: 1px solid #ffd5d5;
}

.risk-low {
    background: #f1fbf3;
    border: 1px solid #ccebd1;
}

.risk-badge {
    width: 31px;
    height: 31px;

    min-width: 31px;

    border-radius: 50%;

    color: #ffffff;

    display: flex;
    align-items: center;
    justify-content: center;

    font-weight: 900;
}

.badge-high {
    background: #ef4444;
}

.badge-low {
    background: #22a447;
}

.risk-title {
    font-size: 13px;
    font-weight: 900;
}

.risk-title-high {
    color: #e43d3d;
}

.risk-title-low {
    color: #218c3a;
}

.risk-text {
    font-size: 10px;
    line-height: 1.5;
    color: #7b7f86;
    margin-top: 3px;
}


/* =========================================================
   DISCLAIMER
   ========================================================= */

.disclaimer {
    margin-top: 10px;

    padding: 10px 11px;

    border-radius: 12px;

    background: #edf7ff;
    border: 1px solid #dceefa;

    color: #4c789b;

    font-size: 9.5px;
    line-height: 1.45;
}


/* =========================================================
   CARE MESSAGE
   ========================================================= */

.care-message {
    margin-top: 15px;

    padding: 17px;

    border-radius: 16px;

    background: #f8fcff;
    border: 1px solid #dcecf8;

    text-align: center;
}

.care-message h3 {
    color: #07549a;
    margin-bottom: 6px;
}

.care-message p {
    color: #6e8da5;
    font-size: 11px;
    line-height: 1.6;
}


/* =========================================================
   HISTORY
   ========================================================= */

.records-row {
    background: #ffffff;

    border: 1px solid #dcecf8;
    border-radius: 14px;

    padding: 12px;
    margin-bottom: 8px;

    display: flex;
    justify-content: space-between;
    align-items: center;
}

.records-date {
    font-size: 10px;
    color: #7898b1;
}

.records-value {
    font-size: 16px;
    font-weight: 900;
    color: #0879d1;
}


/* =========================================================
   ABOUT
   ========================================================= */

.about-box {
    padding: 25px;

    line-height: 1.7;

    color: #6688a5;
    font-size: 12px;
}


/* =========================================================
   ANALYZE BUTTON
   ========================================================= */

.analyze-button button {
    height: 48px !important;

    border-radius: 13px !important;

    background:
        linear-gradient(
            90deg,
            #0874c9,
            #1599e9
        ) !important;

    color: #ffffff !important;

    border: 0 !important;

    font-weight: 900 !important;
    font-size: 13px !important;

    box-shadow:
        0 8px 20px rgba(8,116,201,.18) !important;
}

.analyze-button button p {
    color: #ffffff !important;
}


/* =========================================================
   HIDE STREAMLIT DEFAULT
   ========================================================= */

#MainMenu,
footer {
    visibility: hidden;
}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media(max-width: 1100px) {
    .steps {
        grid-template-columns: 1fr 1fr;
    }
}

@media(max-width: 850px) {
    .header-right {
        display: none;
    }
}

</style>
"""
    )


# =========================================================
# NAVIGATION FUNCTIONS
# =========================================================

def go_page(page):
    st.session_state.page = page

    if page == "Clinical Data":
        st.session_state.section = "Patient Data"

    st.rerun()


def go_section(section):
    st.session_state.page = "Clinical Data"
    st.session_state.section = section

    st.rerun()


# =========================================================
# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.html(
    """
    <div class="nav-logo">🩺</div>
    """
)

nav_items = [
    ("⌂", "Dashboard", "Home"),
    ("♧", "Patient Assessment", "Clinical Data"),
    ("▥", "Risk Assessment", "Analysis"),
    ("◷", "Prediction Records", "Records"),
    ("ⓘ", "Model Information", "About")
]

for icon, display_name, page_name in nav_items:
    label = f"✓  {display_name}" if st.session_state.page == page_name else f"{icon}  {display_name}"
    if st.sidebar.button(
        label,
        key=f"nav_{page_name}",
        use_container_width=True
    ):
        go_page(page_name)

st.sidebar.html(
    """
    <div class="nav-footer">
        Better Data<br>
        Smarter Predictions<br>
        Healthier Tomorrows
    </div>
    """
)


# HEADER
# =========================================================

st.html(
    """
    <div class="header">

        <div>
            <div class="header-title">
                Sepsis Prediction
            </div>

            <div class="header-sub">
                AI-powered clinical risk assessment
            </div>
        </div>

        <div class="header-right">
            <span>AI</span>
            <span>|</span>

            <div class="avatar">
                ML
            </div>

            <b>
                Clinical AI
            </b>
        </div>

    </div>
    """
)


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "Home":

    st.html(
        """
        <div class="home-card">

            <div class="home-title">
                Welcome to Sepsis Prediction
            </div>

            <div class="home-text">
                An educational machine learning dashboard
                that uses selected clinical and laboratory
                measurements to estimate the probability
                of sepsis based on a trained
                HistGradientBoosting model.
            </div>

            <br>

            <div style="
                color:#0879d1;
                font-size:12px;
                font-weight:850;
            ">
                AI Clinical Risk Assessment
            </div>

        </div>

        <br>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.html(
            """
            <div class="home-feature">

                <div style="font-size:25px;">
                    🩺
                </div>

                <div class="home-feature-title">
                    Patient Assessment
                </div>

                <div class="home-feature-text">
                    Enter the clinical and laboratory
                    measurements required by the model.
                </div>

            </div>
            """
    )

    with col2:
        st.html(
        """
            <div class="home-feature">

                <div style="font-size:25px;">
                    🤖
                </div>

                <div class="home-feature-title">
                    Risk Prediction
                </div>

                <div class="home-feature-text">
                    HistGradientBoosting analyzes the
                    submitted clinical measurements.
                </div>

            </div>
            """
    )

    with col3:
        st.html(
        """
            <div class="home-feature">

                <div style="font-size:25px;">
                    📊
                </div>

                <div class="home-feature-title">
                    Clinical Risk Analysis
                </div>

                <div class="home-feature-text">
                    Review the estimated probability
                    and the model prediction.
                </div>

            </div>
            """
    )

    st.write("")

    if st.button(
        "Start Patient Assessment →",
        use_container_width=True
    ):
        go_page("Clinical Data")


# =========================================================
# CLINICAL DATA
# =========================================================

elif st.session_state.page == "Clinical Data":

    st.html(
        """
        <div class="panel">

            <div class="clinical-top">

                <div class="clinical-icon">
                    ♧
                </div>

                <div>

                    <div class="clinical-title">
                        Clinical Data
                    </div>

                    <div class="clinical-sub">
                        Enter all required clinical and
                        laboratory measurements before analysis.
                    </div>

                </div>

            </div>

        </div>
        """
    )

    st.write("")

    # -----------------------------------------------------
    # -----------------------------------------------------
    # SIMPLE STEP NAVIGATION
    # -----------------------------------------------------
    section_items = [
        ("1", "Vital Signs"),
        ("2", "Laboratory Results"),
        ("3", "Review & Predict")
    ]

    cols = st.columns(3)

    for i, (number, name) in enumerate(section_items):
        with cols[i]:
            is_active = st.session_state.section == name
            label = f"✓  {name}" if is_active else f"{number}  {name}"
            if st.button(
                label,
                key=f"section_{name}",
                use_container_width=True
            ):
                go_section(name)


    # =====================================================
    # VITAL SIGNS + PATIENT DETAILS
    # =====================================================

    if st.session_state.section == "Vital Signs":

        st.html(
        """
            <div class="section pink">
                <div class="section-head">
                    <div class="section-icon">♡</div>
                    <div>
                        <div class="section-title-small">
                            Patient & Vital Measurements
                        </div>
                        <div class="section-desc">
                            Enter each value using the label and unit shown below.
                            These are the selected model features used for the assessment.
                        </div>
                    </div>
                </div>
            </div>
        """
        )

        col1, col2 = st.columns(2)

        with col1:
            st.session_state.age = st.slider(
                "Age (years)", 0, 120,
                int(st.session_state.age), 1,
                help="Patient age in years."
            )

            sex_label = st.selectbox(
                "Sex",
                ["Female", "Male"],
                index=1 if st.session_state.sex == 1 else 0,
                help="Encoded for the model as Female = 0, Male = 1."
            )
            st.session_state.sex = 1 if sex_label == "Male" else 0

            # Detect available ICU one-hot categories from the trained model.
            icu_feature_map = {}
            for f in feature_names_hgb:
                f = str(f)
                prefix = "Administrative_identifier_for_ICU_unit__"
                if f.startswith(prefix):
                    suffix = f[len(prefix):].replace("__", " ").strip(" _")
                    display_name = suffix.split(" ")[0] if suffix else "MICU"
                    icu_feature_map[display_name] = f

            if not icu_feature_map:
                icu_feature_map = {"MICU": ICU_FEATURE}

            icu_options = list(icu_feature_map.keys())
            if st.session_state.icu_unit not in icu_options:
                st.session_state.icu_unit = icu_options[0]

            st.session_state.icu_unit = st.selectbox(
                "ICU Unit",
                icu_options,
                index=icu_options.index(st.session_state.icu_unit),
                help="Administrative ICU unit. Choose the unit recorded for the patient."
            )
            st.session_state.icu_feature_name = icu_feature_map[st.session_state.icu_unit]

            st.session_state.heart_rate = st.slider(
                "Heart Rate (beats/min)", 20, 250,
                int(st.session_state.heart_rate), 1,
                help="Heart rate in beats per minute."
            )

            st.session_state.resp_rate = st.slider(
                "Respiration Rate (breaths/min)", 5, 100,
                int(st.session_state.resp_rate), 1,
                help="Number of breaths per minute."
            )

        with col2:
            st.session_state.systolic_bp = st.slider(
                "Systolic Blood Pressure (mm Hg)", 50, 250,
                int(st.session_state.systolic_bp), 1,
                help="Systolic blood pressure."
            )

            st.session_state.map_bp = st.slider(
                "Mean Arterial Pressure (mm Hg)", 30, 200,
                int(st.session_state.map_bp), 1,
                help="Mean arterial pressure (MAP)."
            )

            st.session_state.diastolic_bp = st.slider(
                "Diastolic Blood Pressure (mm Hg)", 20, 150,
                int(st.session_state.diastolic_bp), 1,
                help="Diastolic blood pressure."
            )

            st.session_state.pulse_ox = st.slider(
                "Pulse Oximetry (SpO₂ %)", 50.0, 100.0,
                float(st.session_state.pulse_ox), 1.0,
                help="Peripheral oxygen saturation percentage."
            )

            st.session_state.temperature = st.slider(
                "Temperature (°C)", 30.0, 45.0,
                float(st.session_state.temperature), 0.1,
                help="Body temperature in degrees Celsius."
            )

            st.session_state.fio2 = st.slider(
                "FiO₂ — Fraction of Inspired Oxygen (%)", 21.0, 100.0,
                float(st.session_state.fio2), 1.0,
                help="Percentage of oxygen in the gas being inhaled. 21% is room air."
            )


    # =====================================================
    # LABORATORY
    # =====================================================

    elif st.session_state.section == "Laboratory Results":

        st.html(
        """
            <div class="section teal">
                <div class="section-head">
                    <div class="section-icon">⚗</div>
                    <div>
                        <div class="section-title-small">
                            Laboratory Measurements
                        </div>
                        <div class="section-desc">
                            Enter the serum glucose value used by the selected model features.
                        </div>
                    </div>
                </div>
            </div>
        """
        )

        st.session_state.serum_glucose = st.slider(
            "Serum Glucose (mg/dL)", 20.0, 600.0,
            float(st.session_state.serum_glucose), 1.0,
            help="Blood glucose concentration in mg/dL."
        )

        st.info(
            "The ICU-unit and FiO₂ missing indicators are calculated automatically; "
            "you do not need to enter 0/1 values manually."
        )


    # =====================================================
    # REVIEW & ANALYZE
    # =====================================================

    elif st.session_state.section == "Review & Analyze":

        st.html(
        """
            <div class="section purple">
                <div class="section-head">
                    <div class="section-icon">✓</div>
                    <div>
                        <div class="section-title-small">
                            Review Patient Data
                        </div>
                        <div class="section-desc">
                            Review the 14 model feature concepts before generating the prediction.
                        </div>
                    </div>
                </div>
            </div>
        """
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Age", f"{st.session_state.age} years")
            st.metric("Heart Rate", f"{st.session_state.heart_rate} beats/min")
            st.metric("Respiration Rate", f"{st.session_state.resp_rate} breaths/min")
            st.metric("Systolic BP", f"{st.session_state.systolic_bp} mm Hg")
            st.metric("Mean Arterial Pressure", f"{st.session_state.map_bp} mm Hg")

        with col2:
            st.metric("Diastolic BP", f"{st.session_state.diastolic_bp} mm Hg")
            st.metric("Pulse Oximetry", f"{st.session_state.pulse_ox:.0f}%")
            st.metric("Temperature", f"{st.session_state.temperature:.1f} °C")
            st.metric("Sex", "Male" if st.session_state.sex == 1 else "Female")
            st.metric("ICU Unit", st.session_state.icu_unit)

        with col3:
            st.metric("Serum Glucose", f"{st.session_state.serum_glucose:.0f} mg/dL")
            st.metric("FiO₂", f"{st.session_state.fio2:.0f}%")
            st.metric("ICU Missing Indicator", "0 (recorded)")
            st.metric("FiO₂ Missing Indicator", "0 (recorded)")

        st.write("")

        st.html(
            f"""
            <div class="disclaimer">
                <b>Model:</b> HistGradientBoosting<br>
                <b>Model features:</b> {len(feature_names_hgb)}<br>
                <b>Selected feature concepts:</b> 14<br>
                <b>Decision threshold:</b> {float(threshold):.4f}
            </div>
            """
        )


    # =====================================================
    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    st.write("")

    analyze = st.button(
        "🔍  Generate Risk Assessment",
        use_container_width=True
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    if analyze:

        try:

            # ---------------------------------------------
            # Create all model features
            # ---------------------------------------------

            values_dict = {
                feature: np.nan
                for feature in feature_names_hgb
            }


            # ---------------------------------------------
            # Fill the 14 selected feature concepts
            # ---------------------------------------------

            values_dict[AGE_FEATURE] = st.session_state.age
            values_dict[HR_FEATURE] = st.session_state.heart_rate
            values_dict[RESP_FEATURE] = st.session_state.resp_rate
            values_dict[SYS_BP_FEATURE] = st.session_state.systolic_bp
            values_dict[MAP_FEATURE] = st.session_state.map_bp
            values_dict[DIA_BP_FEATURE] = st.session_state.diastolic_bp
            values_dict[SPO2_FEATURE] = st.session_state.pulse_ox
            values_dict[TEMP_FEATURE] = st.session_state.temperature
            values_dict[SEX_FEATURE] = st.session_state.sex
            values_dict[GLUCOSE_FEATURE] = st.session_state.serum_glucose
            values_dict[FIO2_FEATURE] = st.session_state.fio2

            # ICU is one-hot encoded in the trained feature set.
            # Set all ICU category columns to 0, then activate the selected one.
            for feature in feature_names_hgb:
                if str(feature).startswith("Administrative_identifier_for_ICU_unit__"):
                    values_dict[feature] = 0

            selected_icu_feature = st.session_state.get("icu_feature_name", ICU_FEATURE)
            if selected_icu_feature is not None:
                values_dict[selected_icu_feature] = 1

            # Missing indicators are derived automatically.
            if ICU_MISSING_FEATURE is not None:
                values_dict[ICU_MISSING_FEATURE] = 0
            if FIO2_MISSING_FEATURE is not None:
                values_dict[FIO2_MISSING_FEATURE] = 0

            # ---------------------------------------------
            # Keep exact training feature order
            # ---------------------------------------------

            input_df = pd.DataFrame(
                [values_dict],
                columns=feature_names_hgb
            )


            # ---------------------------------------------
            # Apply trained imputer
            # ---------------------------------------------

            input_imputed = imputer_hgb.transform(input_df)


            # ---------------------------------------------
            # Get probability
            # ---------------------------------------------

            probability = float(
                hgb_model.predict_proba(input_imputed)[0][1]
            )


            # ---------------------------------------------
            # Apply trained threshold
            # ---------------------------------------------

            prediction = int(
                probability >= float(threshold)
            )


            # ---------------------------------------------
            # Save prediction
            # ---------------------------------------------

            st.session_state.probability = probability
            st.session_state.prediction = prediction


            # ---------------------------------------------
            # Save history
            # ---------------------------------------------

            st.session_state.records.append(
                {
                    "time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "probability": probability * 100,
                    "prediction": prediction
                }
            )


            # ---------------------------------------------
            # Go to Analysis
            # ---------------------------------------------

            st.session_state.page = "Analysis"

            st.rerun()


        except Exception as e:

            st.error(
                "Something went wrong while generating "
                "the prediction."
            )

            st.exception(e)


# =========================================================
# ANALYSIS
# =========================================================

elif st.session_state.page == "Analysis":

    prediction = st.session_state.prediction
    probability = st.session_state.probability


    if prediction is None or probability is None:

        st.html(
        """
            <div class="analysis-card">

                <div class="analysis-title">
                    Prediction Analysis
                </div>

                <p style="
                    color:#7595ae;
                    font-size:12px;
                ">
                    No prediction has been generated yet.
                    Enter the clinical data first and then
                    analyze it.
                </p>

            </div>
            """
    )

        if st.button(
            "Return to Patient Assessment →",
            use_container_width=True
        ):
            go_page("Clinical Data")


    else:

        probability_percent = probability * 100

        # Keep the circle between 0 and 100
        safe_probability = max(
            0.0,
            min(100.0, probability_percent)
        )

        degree = safe_probability * 3.6


        # -------------------------------------------------
        # RESULT TITLE
        # -------------------------------------------------

        if prediction == 1:

            risk_title = "Possible Sepsis Risk"
            badge_class = "badge-high"
            risk_class = "risk-high"
            title_class = "risk-title-high"
            badge_text = "!"

        else:

            risk_title = "Lower Predicted Risk"
            badge_class = "badge-low"
            risk_class = "risk-low"
            title_class = "risk-title-low"
            badge_text = "✓"


        # -------------------------------------------------
        # TWO COLUMNS
        # -------------------------------------------------

        left_col, right_col = st.columns(
            [1, 1.2],
            gap="large"
        )


        # -------------------------------------------------
        # IMAGE
        # -------------------------------------------------

        with left_col:

            if os.path.exists(IMAGE_FILE):

                st.image(
                    IMAGE_FILE,
                    use_container_width=True
                )

            else:

                st.html(
        """
                    <div class="analysis-card"
                         style="
                         min-height:360px;
                         display:flex;
                         align-items:center;
                         justify-content:center;
                         ">

                        <div style="
                            text-align:center;
                            color:#7898b1;
                            font-size:13px;
                        ">

                            🩺

                            <br><br>

                            Doctor & Patient

                            <br><br>

                            <span style="font-size:10px;">
                                Add doctor_patient.png
                                to display the image.
                            </span>

                        </div>

                    </div>
                    """
    )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        with right_col:

            st.html(
                f"""
                <div class="analysis-card">

                    <div class="analysis-title">
                        Prediction Result
                    </div>

                    <div style="
                        color:#7898b1;
                        font-size:10px;
                    ">
                        HistGradientBoosting model prediction
                    </div>

                    <div class="gauge"
                         style="
                         background:
                         conic-gradient(
                            #0879d1 0deg,
                            #0879d1 {degree}deg,
                            #d9edf9 {degree}deg,
                            #d9edf9 360deg
                         );
                         ">

                        <div class="gauge-inner">

                            <div class="gauge-number">
                                {probability_percent:.2f}%
                            </div>

                            <div class="gauge-label">
                                Model Probability
                            </div>

                        </div>

                    </div>


                    <div class="risk {risk_class}">

                        <div class="risk-badge {badge_class}">
                            {badge_text}
                        </div>

                        <div>

                            <div class="risk-title {title_class}">
                                {risk_title}
                            </div>

                            <div class="risk-text">
                                The model produced a
                                {probability_percent:.2f}%
                                estimated probability based
                                on the submitted clinical
                                features.
                            </div>

                        </div>

                    </div>


                    <div class="disclaimer">
                        ⓘ This is an educational ML prediction
                        and not a medical diagnosis.
                    </div>

                </div>
                """
            )


            st.progress(
                max(
                    0.0,
                    min(1.0, float(probability))
                )
            )


            st.html(
        """
                <div class="care-message">

                    <h3>
                        Stay Calm, Take Care ♡
                    </h3>

                    <p>
                        This prediction is only an estimate
                        from a machine learning model.
                        It is not a medical diagnosis.
                        If you have health concerns,
                        please speak with a healthcare
                        professional.
                    </p>

                </div>
                """
    )


# =========================================================
# HISTORY
# =========================================================

elif st.session_state.page == "Records":

    st.html(
        """
        <div class="analysis-card">

            <div class="analysis-title">
                Prediction Records
            </div>

            <div style="
                color:#7898b1;
                font-size:10px;
            ">
                Risk assessments generated during this session.
            </div>

        </div>
        """
    )


    if not st.session_state.records:

        st.info(
            "No risk assessments have been generated yet."
        )

    else:

        for item in reversed(
            st.session_state.records
        ):

            if item["prediction"] == 1:
                result_text = "Possible Sepsis Risk"
            else:
                result_text = "Lower Predicted Risk"


            st.html(
                f"""
                <div class="records-row">

                    <div>

                        <div style="
                            font-size:12px;
                            font-weight:850;
                            color:#07549a;
                        ">
                            {result_text}
                        </div>

                        <div class="records-date">
                            {item["time"]}
                        </div>

                    </div>

                    <div class="records-value">
                        {item["probability"]:.2f}%
                    </div>

                </div>
                """
            )


# =========================================================
# ABOUT
# =========================================================

elif st.session_state.page == "About":

    st.html(
        f"""
        <div class="about-box">

            <div style="
                font-size:24px;
                font-weight:900;
                color:#07549a;
            ">
                About the Project
            </div>

            <br>

            This project is an educational machine learning
            application for sepsis risk prediction.

            <br><br>

            <b style="color:#0879d1">
                Model:
            </b>
            HistGradientBoosting Classifier

            <br>

            <b style="color:#0879d1">
                Model Features:
            </b>
            48 features

            <br>

            <b style="color:#0879d1">
                User Input Features:
            </b>
            14 selected model feature concepts

            <br>

            <b style="color:#0879d1">
                Preprocessing:
            </b>
            Trained imputation process

            <br>

            <b style="color:#0879d1">
                Prediction:
            </b>
            Model probability compared with the trained threshold

            <br><br>

            The application is designed for learning and
            demonstration purposes. The prediction is not a
            medical diagnosis and should not replace professional
            medical evaluation.

        </div>
        """
    )