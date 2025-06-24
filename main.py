import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


encoder = load_model("model/encoder.pkl")


model = load_model("model/model.pkl")


app = FastAPI()

#input schema using Pydantic
class CensusInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

Data = CensusInput #reusability
categorical_features = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country"
]  
label = "salary"

@app.post("/model")
def predict(data: CensusInput):
    #convert input into dataframe
    input_df = pd.DataFrame([data.dict()])

    #preprocessing using same encoder/label binarizer
    X, _, _, _ = process_data(
        input_df,
        categorical_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=None
    )
    
    #Run inference
    prediction = inference(model, X)
    result = '>50k' if prediction[0] == 1 else '<50k'

    return {'prediction': result}


@app.get("/")
async def get_root():
    """ Say hello!"""
    return {"message": "Welcome to the Census Income Inference API"}


@app.post("/data/")
async def post_inference(data: Data):
  
    data_dict = data.dict()
    
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    categorical_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    data_processed, _, _, _ = process_data(
        data,
        categorical_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=None
    )
    _inference = inference(model, data_processed)
    return {"result": apply_label(_inference)}
