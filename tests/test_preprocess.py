import pandas as pd


def test_missing_values_filled():
    """Test that missing values can be filled without leaving NaNs."""

    df = pd.DataFrame({
        "cycle": [1, 2, None],
        "chI": [1.2, None, 1.5],
        "RUL": [100, 90, 80]
    })

    cleaned_df = df.fillna(df.mean(numeric_only=True))

    assert cleaned_df.isnull().sum().sum() == 0


def test_original_dataframe_not_modified():
    """Test that preprocessing does not modify the original dataframe."""

    df = pd.DataFrame({
        "cycle": [1, 2, None],
        "chI": [1.2, None, 1.5],
        "RUL": [100, 90, 80]
    })

    original_df = df.copy(deep=True)

    cleaned_df = df.copy()
    cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))

    pd.testing.assert_frame_equal(df, original_df)


def test_numeric_scaling_shape():
    """Test that numeric scaling keeps the same dataframe shape."""

    from sklearn.preprocessing import StandardScaler

    df = pd.DataFrame({
        "cycle": [1, 2, 3],
        "chI": [1.2, 1.3, 1.5],
        "chV": [4.0, 4.1, 4.2]
    })

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    assert scaled_data.shape == df.shape