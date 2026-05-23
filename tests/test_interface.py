from src.app import parse_user_input, check_missing_features


def test_parse_user_input_extracts_features():
    """Test that natural language input is parsed into model features."""

    user_query = """
    Predict RUL for cycle 120, charge current 1.5,
    charge voltage 4.1, charge temperature 30,
    discharge current 1.2, discharge voltage 3.7,
    discharge temperature 28, battery capacity time 250,
    and SOH 85.
    """

    features = parse_user_input(user_query)

    assert features["cycle"] == 120
    assert features["chI"] == 1.5
    assert features["chV"] == 4.1
    assert features["chT"] == 30
    assert features["disI"] == 1.2
    assert features["disV"] == 3.7
    assert features["disT"] == 28
    assert features["BCt"] == 250
    assert features["SOH"] == 85


def test_missing_features_detected():
    """Test that missing battery features are detected."""

    user_query = "Predict battery RUL for cycle 120 and charge current 1.5"

    features = parse_user_input(user_query)
    missing = check_missing_features(features)

    assert "chV" in missing
    assert "SOH" in missing
    assert len(missing) > 0