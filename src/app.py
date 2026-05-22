import os
import re
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")


# -----------------------------
# 2. Define model path
# -----------------------------
MODEL_PATH = "models/best_model.pkl"


# -----------------------------
# 3. Required model features
# -----------------------------
REQUIRED_FEATURES = [
    "cycle",
    "chI",
    "chV",
    "chT",
    "disI",
    "disV",
    "disT",
    "BCt",
    "SOH"
]


# -----------------------------
# 4. Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    """Load the saved battery RUL prediction model."""
    model = joblib.load(MODEL_PATH)
    return model


# -----------------------------
# 5. Parse user input
# -----------------------------
def parse_user_input(user_query):
    """
    Extract battery feature values from natural language text.

    Example:
    "cycle 120, charge current 1.5, charge voltage 4.1"
    """

    text = user_query.lower()

    patterns = {
        "cycle": r"cycle\s*[:=]?\s*(\d+\.?\d*)",
        "chI": r"(charge current|chi)\s*[:=]?\s*(\d+\.?\d*)",
        "chV": r"(charge voltage|chv)\s*[:=]?\s*(\d+\.?\d*)",
        "chT": r"(charge temperature|cht)\s*[:=]?\s*(\d+\.?\d*)",
        "disI": r"(discharge current|disi)\s*[:=]?\s*(\d+\.?\d*)",
        "disV": r"(discharge voltage|disv)\s*[:=]?\s*(\d+\.?\d*)",
        "disT": r"(discharge temperature|dist)\s*[:=]?\s*(\d+\.?\d*)",
        "BCt": r"(capacity time|battery capacity time|bct)\s*[:=]?\s*(\d+\.?\d*)",
        "SOH": r"(soh|state of health)\s*[:=]?\s*(\d+\.?\d*)"
    }

    features = {}

    for feature, pattern in patterns.items():
        match = re.search(pattern, text)

        if match:
            value = float(match.groups()[-1])
            features[feature] = value
        else:
            features[feature] = None

    return features


# -----------------------------
# 6. Check missing values
# -----------------------------
def check_missing_features(features):
    """Return a list of missing input features."""
    missing = []

    for key, value in features.items():
        if value is None:
            missing.append(key)

    return missing


# -----------------------------
# 7. Generate prediction
# -----------------------------
def predict_rul(model, features):
    """Convert parsed features into DataFrame and predict RUL."""

    input_df = pd.DataFrame([features])
    input_df = input_df[REQUIRED_FEATURES]

    prediction = model.predict(input_df)[0]

    return prediction


# -----------------------------
# 8. Generate natural response
# -----------------------------
def generate_response(predicted_rul):
    """Explain the RUL prediction in simple engineering language."""

    if predicted_rul > 200:
        status = "Healthy battery"
        explanation = (
            "The battery still has a high remaining useful life. "
            "This means the degradation level is low and the battery can likely continue operating normally."
        )

    elif predicted_rul > 100:
        status = "Moderate degradation"
        explanation = (
            "The battery is showing signs of aging. "
            "It can still operate, but performance may gradually decrease as cycles increase."
        )

    else:
        status = "High degradation"
        explanation = (
            "The battery has a low remaining useful life. "
            "This suggests significant degradation, reduced capacity, and possible need for replacement or maintenance soon."
        )

    response = f"""
### Prediction Result

**Predicted Remaining Useful Life:** {predicted_rul:.2f} cycles

**Battery Status:** {status}

**Explanation:**  
{explanation}
"""

    return response


# -----------------------------
# 9. Streamlit App
# -----------------------------
st.title("Battery RUL Prediction Assistant")

st.write(
    "Enter a battery condition in natural language. "
    "The app will extract feature values and predict Remaining Useful Life."
)

user_query = st.text_area(
    "Enter your battery information:",
    placeholder=(
        "Example: Predict RUL for cycle 120, charge current 1.5, "
        "charge voltage 4.1, charge temperature 30, discharge current 1.2, "
        "discharge voltage 3.7, discharge temperature 28, "
        "battery capacity time 250, and SOH 85."
    )
)

if st.button("Predict Battery RUL"):

    if not user_query.strip():
        st.warning("Please enter battery information before prediction.")

    else:
        features = parse_user_input(user_query)

        st.subheader("Parsed Model Inputs")
        st.json(features)

        missing_features = check_missing_features(features)

        if missing_features:
            st.error(
                "Some required battery information is missing. "
                f"Please provide: {', '.join(missing_features)}"
            )

        else:
            model = load_model()
            predicted_rul = predict_rul(model, features)
            response = generate_response(predicted_rul)

            st.markdown(response)