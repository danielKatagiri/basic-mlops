import pickle
from typing import Tuple, Union

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .logging import LOGGER
from .settings import ARTIFACTS_HOME


class TitanicModel:
    def __init__(self):
        self.model = self._load_artifact(filename="titanic_classifier")
        self.scaler = self._load_artifact(filename="titanic_features_scaler")

    @staticmethod
    def _load_artifact(
        filename: str,
    ) -> Union[pd.DataFrame, GradientBoostingClassifier, StandardScaler]:
        """
        Loads model from the 'ARTIFACTS_HOME' folder.
        In a production environment, this might not be the best approach. Your company should have a model registry that
        you can use to save the models you have trained.

        Returns:
            (Union[pd.DataFrame, GradientBoostingClassifier, StandardScaler]): A trained model to inference if a
            passenger survived
        """
        LOGGER.info(f"Retrieving artifact: {filename}.")

        try:
            with open(f"{ARTIFACTS_HOME}/{filename}.pkl", "rb") as file:
                artifact = pickle.load(file)

            LOGGER.info(f"Artifact {filename} successfully loaded.")

            return artifact
        except Exception as err:
            raise Exception(
                f"The model could not be loaded from the `ARTIFACTS_HOME` folder. Make sure the artifact "
                f"{filename} is available."
            ) from err

    @staticmethod
    def encode_sex(sex: str) -> int:
        """
        Encodes the sex ('male', 'female') according to the preprocessing routine.
        Parameters:
            sex (str): The sex of the passenger. Accepted values as 'male' and 'female'.
        Returns:
            (int): Encoded sex value
        """
        LOGGER.info("Starting sex encoding.")

        if sex not in ["male", "female"]:
            raise ValueError(
                "Value for 'sex' not accepted. Supported values are 'male' and 'female'."
            )

        encoded_sex = 0 if sex == "male" else 0

        LOGGER.info("Finished sex encoding.")

        return encoded_sex

    def scale_features(
        self, age: Union[float, int], fare: Union[float, int]
    ) -> Tuple[float, float]:
        """
        Scales the numeric features according to the preprocessing scaling. It is important to scale the input features
        according to the process used in the training phase, so that our model works as expected.
        Parameters:
            age (float): The age of the person
            fare (float): The fare paid by the person
        Returns:
             (Tuple[float, float]): The scaled age and fare
        """
        LOGGER.info("Starting feature scaling for age and fare.")

        scaled_features = self.scaler.transform([[age, fare]]).reshape(-1).tolist()
        scaled_age = scaled_features[0]
        scaled_fare = scaled_features[1]

        LOGGER.info("Feature scaling for age and fare finished.")

        return scaled_age, scaled_fare

    def predict(
        self, pclass: int, sex: str, age: Union[float, int], fare: Union[float, int]
    ) -> bool:
        """
        Run the inference given the features
        Parameters:
            pclass: The pclass of the passenger
            sex: The sex of the passenger ['male', 'female']
            age: The age of the passenger
            fare: The fare paid by the passenger
        Returns:
            (bool): Returns a boolean indicating if the passenger survived or not
        """
        LOGGER.info("Starting inference pipeline.")

        encoded_sex = self.encode_sex(sex=sex)
        scaled_age, scaled_fare = self.scale_features(age=age, fare=fare)
        predictions = self.model.predict(
            [[pclass, encoded_sex, scaled_age, scaled_fare]]
        ).tolist()
        final_prediction = bool(predictions[0])

        LOGGER.info("Inference pipeline finished. Returning result.")

        return final_prediction
