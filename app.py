from typing import Union

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from titanic.titanic_model import TitanicModel


class Passenger(BaseModel):
    """Data model"""

    pclass: int
    sex: str
    age: Union[float, int]
    fare: Union[float, int]


def create_app():
    """Creates a FastAPI object with the prediction route"""
    application = FastAPI()
    model = TitanicModel()

    @application.get("/")
    async def root():
        return {
            "message": "This is the Titanic API. Access /docs to check the swagger."
        }

    @application.get("/predict")
    async def predict(
        pclass: int, sex: str, age: Union[float, int], fare: Union[float, int]
    ):
        return model.predict(pclass=pclass, sex=sex, age=age, fare=fare)

    return application


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)
