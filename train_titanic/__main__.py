import pandas as pd

from .logging import LOGGER


def download_data() -> pd.DataFrame:
    """Download the data from web for training"""
    LOGGER.info("Starting dataset download.")

    df = pd.read_csv(
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    )

    LOGGER.info("Dataset download finished.")

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the training data by selecting columns and normalizing values
    Parameters:
        df (pd.DataFrame): Raw dataset
    Returns:
        (pd.DataFrame): Preprocessed dataset
    """

    LOGGER.info("Starting dataset preprocessing.")

    selected_columns = ["Pclass", "Sex", "Age", "Fare", "Survived"]

    preprocessed_df = df.loc[:, selected_columns].assign(
        Sex=lambda df_: df_["Sex"].map({"male": 0, "female": 1})
    )

    LOGGER.info("Dataset preprocessing finished.")

    return preprocessed_df
