"""
Diabetes Prediction System
---------------------------
A Streamlit application that predicts diabetes risk from patient health
metrics using a pre-trained machine learning model.

Author: Faiza Fatima
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =========================================================
# CONSTANTS
# =========================================================

MODEL_PATH = Path("diabetes_model.pkl")
FEATURES_PATH = Path("diabetes_features.pkl")

FIELD_CONFIG = {
    "Pregnancies": dict(label="Pregnancies", min_value=0, max_value=20, value=2, step=1),
    "Glucose": dict(label="Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0),
    "BloodPressure": dict(label="Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0),
    "SkinThickness": dict(label="Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=25.0, step=1.0),
    "Insulin": dict(label="Insulin (mu U/mL)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0),
    "BMI": dict(label="BMI", min_value=0.0, max_value=70.0, value=30.5, step=0.1),
    "DiabetesPedigreeFunction": dict(label="Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.45, step=0.01),
    "Age": dict(label="Age (years)", min_value=1, max_value=120, value=35, step=1),
}

DIET_TIPS = [
    "Include plenty of vegetables and high-fiber foods.",
    "Choose whole grains and balanced meals.",
    "Include healthy protein sources such as pulses, beans, eggs or fish.",
    "Limit sugary drinks, sweets and highly refined carbohydrates.",
    "Pay attention to portion sizes.",
    "Stay physically active according to professional advice.",
    "Maintain healthy sleep and hydration habits.",
    "Follow your healthcare professional's recommendations for monitoring and treatment.",
]

HEALTHY_HABITS = [
    ("Eat Healthy", "Choose balanced meals with vegetables, fiber and nutritious foods."),
    ("Stay Active", "Regular physical activity can support overall health and wellbeing."),
    ("Sleep Well", "Maintain a consistent sleep routine and give your body enough rest."),
    ("Stay Positive", "Take small steps every day towards healthier and happier habits."),
]

# =========================================================
# THEME DEFINITIONS
# =========================================================

THEMES = {
    "light": {
        "bg-gradient": "linear-gradient(135deg, #f7f8ff 0%, #eef6ff 50%, #f6ffff 100%)",
        "header-gradient": "linear-gradient(135deg, #5125d8, #743bea, #9b55ff)",
        "header-text": "#ffffff",
        "card-bg": "#ffffff",
        "card-border": "#eeeeff",
        "text-primary": "#29294d",
        "text-secondary": "#707089",
        "input-bg": "#ffffff",
        "button-gradient": "linear-gradient(90deg, #6332df, #8c4df5)",
        "button-shadow": "rgba(99, 50, 223, 0.25)",
        "button-shadow-hover": "rgba(99, 50, 223, 0.35)",
        "result-positive-bg": "linear-gradient(135deg, #fff0f3, #ffe1e7)",
        "result-positive-border": "#ffb5c1",
        "result-positive-title": "#d62839",
        "result-negative-bg": "linear-gradient(135deg, #edfff4, #dcf9e8)",
        "result-negative-border": "#a4e6bc",
        "result-negative-title": "#16803c",
        "advice-border": "#7b3ff2",
        "advice-text": "#55556b",
        "disclaimer-bg": "#fff8e7",
        "disclaimer-border": "#ffd98b",
        "disclaimer-text": "#6b5318",
        "footer-text": "#77778c",
        "footer-name": "#6332df",
    },
    "dark": {
        "bg-gradient": "linear-gradient(135deg, #131320 0%, #14141f 50%, #101018 100%)",
        "header-gradient": "linear-gradient(135deg, #3c1c96, #55299c, #6d3fc4)",
        "header-text": "#f4f2ff",
        "card-bg": "#1b1b29",
        "card-border": "#2c2c40",
        "text-primary": "#eceaf7",
        "text-secondary": "#a3a1b8",
        "input-bg": "#232335",
        "button-gradient": "linear-gradient(90deg, #4d28a8, #7040c8)",
        "button-shadow": "rgba(112, 64, 200, 0.35)",
        "button-shadow-hover": "rgba(112, 64, 200, 0.5)",
        "result-positive-bg": "linear-gradient(135deg, #34101a, #290c14)",
        "result-positive-border": "#7a2f3f",
        "result-positive-title": "#ff6b81",
        "result-negative-bg": "linear-gradient(135deg, #0f2a1c, #0b2015)",
        "result-negative-border": "#2f6b47",
        "result-negative-title": "#4fd97f",
        "advice-border": "#9b6bff",
        "advice-text": "#c1bfd4",
        "disclaimer-bg": "#2a2210",
        "disclaimer-border": "#5c4a1a",
        "disclaimer-text": "#e3c878",
        "footer-text": "#8a88a0",
        "footer-name": "#a887ff",
    },
}

BASE_CSS = """
.stApp {
    background: var(--bg-gradient);
}
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.main-header {
    background: var(--header-gradient);
    padding: 35px 45px;
    border-radius: 25px;
    margin-bottom: 30px;
    color: var(--header-text);
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.20);
}
.main-header h1 { font-size: 42px; font-weight: 800; margin: 0; }
.main-header p { font-size: 18px; margin-top: 10px; }
.card {
    background: var(--card-bg);
    padding: 28px;
    border-radius: 22px;
    border: 1px solid var(--card-border);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);
    margin-bottom: 20px;
}
.section-title { font-size: 25px; font-weight: 750; color: var(--text-primary); }
.section-text { color: var(--text-secondary); font-size: 15px; margin-top: 5px; margin-bottom: 20px; }
label { font-weight: 600 !important; color: var(--text-primary) !important; }
[data-testid="stNumberInput"] input {
    background-color: var(--input-bg) !important;
    color: var(--text-primary) !important;
}
div.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 15px;
    border: none;
    background: var(--button-gradient);
    color: #ffffff;
    font-size: 19px;
    font-weight: 700;
    box-shadow: 0 8px 20px var(--button-shadow);
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px var(--button-shadow-hover);
}
.result-positive {
    background: var(--result-positive-bg);
    border: 2px solid var(--result-positive-border);
    padding: 35px;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 20px;
}
.result-negative {
    background: var(--result-negative-bg);
    border: 2px solid var(--result-negative-border);
    padding: 35px;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 20px;
}
.result-title { font-size: 31px; font-weight: 800; }
.result-probability { font-size: 19px; color: var(--text-secondary); margin-top: 10px; }
.advice {
    background: var(--card-bg);
    padding: 25px;
    border-radius: 18px;
    border-left: 6px solid var(--advice-border);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
    margin-bottom: 18px;
}
.advice h3 { color: var(--text-primary); }
.advice p { color: var(--advice-text); line-height: 1.7; }
.advice li { color: var(--advice-text); line-height: 1.8; margin-bottom: 5px; }
.tip {
    background: var(--card-bg);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    min-height: 110px;
    border: 1px solid var(--card-border);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
}
.tip-title { font-size: 18px; font-weight: 700; color: var(--text-primary); }
.tip-text { color: var(--text-secondary); font-size: 14px; line-height: 1.5; margin-top: 5px; }
.disclaimer {
    background: var(--disclaimer-bg);
    border: 1px solid var(--disclaimer-border);
    padding: 17px;
    border-radius: 15px;
    color: var(--disclaimer-text);
    font-size: 14px;
    line-height: 1.6;
    margin-top: 20px;
}
.footer { text-align: center; padding: 40px 20px 15px 20px; color: var(--footer-text); font-size: 14px; }
.footer-name { font-size: 17px; font-weight: 700; color: var(--footer-name); margin-top: 10px; }
.footer-line {
    width: 80px;
    height: 3px;
    background: var(--button-gradient);
    border-radius: 10px;
    margin: 15px auto;
}
"""


def build_theme_css(theme_name: str) -> str:
    """Build the full <style> block for the given theme."""
    theme = THEMES[theme_name]
    var_lines = "\n".join(f"    --{key}: {value};" for key, value in theme.items())
    root_block = f":root {{\n{var_lines}\n}}"
    return f"<style>\n{root_block}\n{BASE_CSS}\n</style>"


# =========================================================
# DATA / MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model_assets():
    """Load the trained model and expected feature order.

    Returns:
        tuple: (model, features) or (None, None) if loading fails.
    """
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, features
    except Exception:
        return None, None


# =========================================================
# UI SECTIONS
# =========================================================

def render_theme_toggle() -> None:
    """Render the light/dark mode toggle and store the choice in session state."""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    _, toggle_col = st.columns([6, 1])
    with toggle_col:
        st.toggle("Dark Mode", key="dark_mode")


def render_header() -> None:
    st.html("""
    <div class="main-header">
        <h1>Diabetes Prediction System</h1>
        <p>Check your diabetes risk using Machine Learning and take a step
        towards a healthier lifestyle.</p>
    </div>
    """)


def render_patient_form() -> dict:
    """Render the patient input form and return the collected values."""
    st.html("""
    <div class="card">
        <div class="section-title">Patient Information</div>
        <div class="section-text">
            Enter the patient's health information below to generate a
            prediction.
        </div>
    </div>
    """)

    left_keys = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness"]
    right_keys = ["Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

    values = {}
    col1, col2 = st.columns(2)

    with col1:
        for key in left_keys:
            values[key] = st.number_input(**FIELD_CONFIG[key])

    with col2:
        for key in right_keys:
            values[key] = st.number_input(**FIELD_CONFIG[key])

    return values


def render_disclaimer() -> None:
    st.html("""
    <div class="disclaimer">
        <b>Important:</b> This application is developed for educational
        and demonstration purposes only. The prediction should not be
        considered a medical diagnosis. Please consult a qualified
        healthcare professional for medical advice.
    </div>
    """)


def run_prediction(model, features: list, patient_values: dict):
    """Run the model on the patient's inputs.

    Returns:
        tuple: (prediction: int, probability: float)
    """
    input_data = pd.DataFrame([patient_values])[features]
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    return prediction, probability


def render_positive_result(probability: float) -> None:
    st.html(f"""
    <div class="result-positive">
        <div class="result-title" style="color: var(--result-positive-title);">
            Diabetes Risk Detected
        </div>
        <div class="result-probability">
            Estimated Probability: <b>{probability:.2%}</b>
        </div>
    </div>
    """)

    st.html("""
    <div class="advice">
        <h3>Please Take Care of Your Health</h3>
        <p>The model indicates a higher likelihood of diabetes based on the
        information provided.</p>
        <p>This prediction does not confirm a medical diagnosis. Please
        consult a qualified healthcare professional for proper testing and
        medical guidance.</p>
        <p>Taking care of nutrition, physical activity, sleep and regular
        health monitoring can support better long-term health.</p>
    </div>
    """)

    tips_html = "".join(f"<li>{tip}</li>" for tip in DIET_TIPS)
    st.html(f"""
    <div class="advice">
        <h3>Healthy Diet &amp; Lifestyle Suggestions</h3>
        <ul>{tips_html}</ul>
    </div>
    """)


def render_negative_result(probability: float) -> None:
    st.html(f"""
    <div class="result-negative">
        <div class="result-title" style="color: var(--result-negative-title);">
            No Diabetes Detected
        </div>
        <div class="result-probability">
            Estimated Probability: <b>{probability:.2%}</b>
        </div>
    </div>
    """)

    st.html("""
    <div class="advice">
        <h3>Great! Keep Taking Care of Yourself!</h3>
        <p>Based on the information provided, the model did not detect a
        diabetes outcome.</p>
        <p>Continue maintaining healthy habits, eating balanced meals,
        staying active and getting regular health check-ups.</p>
        <p>Small healthy choices today can contribute to better health
        tomorrow.</p>
        <p>Keep going — your health is worth taking care of every day!</p>
    </div>
    """)


def render_result_panel(model, features: list, patient_values: dict, predict_clicked: bool) -> None:
    st.html("""
    <div class="card">
        <div class="section-title">Prediction Result</div>
        <div class="section-text">
            Your Machine Learning prediction will appear here.
        </div>
    </div>
    """)

    if not predict_clicked:
        return

    if model is None or features is None:
        st.error(
            "Model files could not be loaded. Please make sure "
            "'diabetes_model.pkl' and 'diabetes_features.pkl' are present "
            "in the app directory."
        )
        return

    try:
        prediction, probability = run_prediction(model, features, patient_values)
    except Exception as exc:
        st.error(f"Something went wrong while generating the prediction: {exc}")
        return

    if prediction == 1:
        render_positive_result(probability)
    else:
        render_negative_result(probability)


def render_healthy_habits() -> None:
    st.write("")
    st.write("")
    st.html("""
    <div style="text-align:center; margin-bottom:25px;">
        <div class="section-title">Simple Healthy Habits</div>
        <div class="section-text">
            Small healthy choices can make a positive difference.
        </div>
    </div>
    """)

    columns = st.columns(4)
    for col, (title, text) in zip(columns, HEALTHY_HABITS):
        with col:
            st.html(f"""
            <div class="tip">
                <div class="tip-title">{title}</div>
                <div class="tip-text">{text}</div>
            </div>
            """)


def render_footer() -> None:
    st.html("""
    <div class="footer">
        <div><b>Diabetes Prediction System</b></div>
        <div class="footer-line"></div>
        <div>Predict &bull; Prevent &bull; Live Better</div>
        <div class="footer-name">Developed by Faiza Fatima</div>
        <br>
        <i>Machine Learning project developed for educational purposes.</i>
    </div>
    """)


# =========================================================
# APP ENTRY POINT
# =========================================================

def main() -> None:
    st.set_page_config(
        page_title="Diabetes Prediction System",
        layout="wide",
    )

    render_theme_toggle()
    theme_name = "dark" if st.session_state.dark_mode else "light"
    st.markdown(build_theme_css(theme_name), unsafe_allow_html=True)

    model, features = load_model_assets()

    render_header()

    left, right = st.columns([1, 1], gap="large")

    with left:
        patient_values = render_patient_form()
        st.write("")
        predict_clicked = st.button("Predict Diabetes", use_container_width=True)
        render_disclaimer()

    with right:
        render_result_panel(model, features, patient_values, predict_clicked)

    render_healthy_habits()
    render_footer()


if __name__ == "__main__":
    main()
