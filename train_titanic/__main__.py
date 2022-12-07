import pickle
from typing import List, Tuple, Union

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from titanic.logging import LOGGER


def download_data() -> pd.DataFrame:
    """Download the data from web for training"""
    LOGGER.info("Starting dataset download.")

    df = pd.read_csv(
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    )

    LOGGER.info("Dataset download finished.")

    return df


def preprocess_data(
    df: pd.DataFrame, features: List[str], target: str
) -> Tuple[StandardScaler, pd.DataFrame]:
    """
    Preprocess the training data by selecting columns and scaling values
    Parameters:
        df (pd.DataFrame): Raw dataset
        features (List[str]): Columns used as features
        target (str): Target column
    Returns:
        (pd.DataFrame): Preprocessed dataset
    """

    LOGGER.info("Starting dataset preprocessing.")

    scaler = StandardScaler()
    scaler.fit(df[["Age", "Fare"]])

    transformed_features = scaler.transform(df[["Age", "Fare"]])

    preprocessed_df = df.loc[:, features + [target]].assign(
        Sex=lambda df_: df_["Sex"].map({"male": 0, "female": 1})
    )

    preprocessed_df[["Age", "Fare"]] = transformed_features

    LOGGER.info("Dataset preprocessing finished.")

    return scaler, preprocessed_df


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
    artifact: Union[pd.DataFrame, GradientBoostingClassifier, StandardScaler],
    filename: str,
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
        elif isinstance(artifact, GradientBoostingClassifier) or isinstance(
            artifact, StandardScaler
        ):
            with open(f"./artifacts/{filename}.pkl", "wb") as file:
                pickle.dump(artifact, file)
        else:
            raise TypeError(f"The type {type(artifact)} is not supported.")

        LOGGER.info(f"Artifact {filename} successfully saved.")
    except Exception as err:
        LOGGER.exception(err)


def evaluate_model(
    model: GradientBoostingClassifier,
    test_df: pd.DataFrame,
    features: List[str],
    target: str,
):
    """
    Evaluates the model against a test dataset and prints the metrics
    Parameters:
        model (GradientBoostingClassifier): Trained classifier model
        test_df (pd.DataFrame): Test dataset
        features (List[str]): Columns used as features
        target (str): Target column
    """
    LOGGER.info("Starting model evaluation.")

    predictions = model.predict(test_df[features])

    f1 = f1_score(test_df[target], predictions)

    LOGGER.info(f"The model f1 score is: {f1}.")


def main():
    """Main function to train titanic model"""
    features = ["Pclass", "Sex", "Age", "Fare"]
    target = "Survived"

    df = download_data()
    scaler, preprocessed_df = preprocess_data(df=df, features=features, target=target)
    train_df, test_df = train_test_split(
        preprocessed_df, test_size=10, random_state=123
    )
    model = train_model(df=train_df, features=features, target=target)
    evaluate_model(model=model, test_df=test_df, features=features, target=target)

    save_artifact(train_df, "train_df")
    save_artifact(test_df, "test_df")
    save_artifact(model, "titanic_classifier")
    save_artifact(scaler, "titanic_features_scaler")


if __name__ == "__main__":
    main()
