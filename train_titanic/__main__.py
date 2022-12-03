import pickle
from typing import List, Union

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .logging import LOGGER


def download_data() -> pd.DataFrame:
    """Download the data from web for training"""
    LOGGER.info("Starting dataset download.")

    df = pd.read_csv(
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    )

    LOGGER.info("Dataset download finished.")

    return df


def preprocess_data(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    """
    Preprocess the training data by selecting columns and normalizing values
    Parameters:
        df (pd.DataFrame): Raw dataset
        features (List[str]): Columns used as features
        target (str): Target column
    Returns:
        (pd.DataFrame): Preprocessed dataset
    """

    LOGGER.info("Starting dataset preprocessing.")

    scaler = StandardScaler()

    preprocessed_df = df.loc[:, features + [target]].assign(
        Sex=lambda df_: df_["Sex"].map({"male": 0, "female": 1}),
        Age=lambda df_: scaler.fit_transform(df_[["Age"]]),
        Fare=lambda df_: scaler.fit_transform(df_[["Fare"]]),
    )

    LOGGER.info("Dataset preprocessing finished.")

    return preprocessed_df


def train_model(
    df: pd.DataFrame, features: List[str], target: str
) -> GradientBoostingClassifier:
    """
    Train a gradient boosting classifier
    Parameters:
        df (pd.Dataframe): Training dataframe
        features (List[str]): Columns used as features
        target (str): Target column
    Returns:
        (GradientBoostingClassifier): Trained classifier model
    """
    LOGGER.info("Training classifier model.")

    model = GradientBoostingClassifier()
    model.fit(df[features], df[target])

    LOGGER.info("Finished model training.")

    return model


def save_artifact(
    artifact: Union[pd.DataFrame, GradientBoostingClassifier], filename: str
):
    """
    Saves the artifact to local file system
    Parameters:
        artifact: The artifact to be saved
        filename: The name of the file
    """
    try:
        LOGGER.info(f"Saving artifact {filename}.")
        if isinstance(artifact, pd.DataFrame):
            artifact.to_parquet(f"./artifacts/{filename}.parquet")
        elif isinstance(artifact, GradientBoostingClassifier):
            with open(f"./artifacts/{filename}.pkl", "wb") as file:
                pickle.dump(artifact, file)
        else:
            raise TypeError(f"The type {type(artifact)} is not supported.")

        LOGGER.info(f"Artifact {filename} successfully saved.")
    except Exception as err:
        LOGGER.exception(err)


def main():
    """Main function to train titanic model"""
    features = ["Pclass", "Sex", "Age", "Fare"]
    target = "Survived"

    df = download_data()
    preprocessed_df = preprocess_data(df=df, features=features, target=target)
    train_df, test_df = train_test_split(preprocessed_df, test_size=10)
    model = train_model(df=train_df, features=features, target=target)

    save_artifact(train_df, "train_df")
    save_artifact(test_df, "test_df")
    save_artifact(model, "titanic_classifier")


if __name__ == "__main__":
    main()
