import os
import joblib
import numpy as np
import pandas as pd


def test_model_file_exists():
    """Test that the trained model file exists."""

    model_path = "models/best_model.pkl"

    assert os.path.exists(model_path)


def test_model_prediction_shape_and_type():
    """Test that model prediction is numeric and has correct shape."""

    model_path = "models/best_model.pkl"
    model = joblib.load(model_path)

    sample_input = pd.DataFrame([{
        "cycle": 120,
        "chI": 1.5,
        "chV": 4.1,
        "chT": 30,
        "disI": 1.2,
        "disV": 3.7,
        "disT": 28,
        "BCt": 250,
        "SOH": 85
    }])

    prediction = model.predict(sample_input)

    assert prediction.shape == (1,)
    assert isinstance(prediction[0], (int, float, np.integer, np.floating))