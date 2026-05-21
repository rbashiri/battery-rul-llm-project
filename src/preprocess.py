import pandas as pd


def load_data(input_path):
    """Load raw battery dataset."""
    return pd.read_csv(input_path)


def inspect_data(df):
    """Display dataset information."""
    
    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns)

    print("\nData Types:")
    print(df.dtypes)

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nStatistical Summary:")
    print(df.describe())


def clean_data(df):
    """Clean dataset."""
    
    df = df.copy()

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, output_path):
    """Save cleaned dataset."""
    df.to_csv(output_path, index=False)


if __name__ == "__main__":

    input_path = "data/raw/Battery_dataset.csv"
    output_path = "data/processed/battery_cleaned.csv"

    # Load data
    df = load_data(input_path)

    # Inspect raw data
    inspect_data(df)

    # Clean data
    df_cleaned = clean_data(df)

    # Save processed data
    save_data(df_cleaned, output_path)

    print("\nPreprocessing completed successfully.")
