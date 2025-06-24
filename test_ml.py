import pytest
import numpy as np
import pandas as pd
from ml.model import train_model, inference
from ml.model import process_data


#sample data
@pytest.fixture
def census_sample():
    df = pd.read_csv("data/census.csv")
    return df.sample(100, random_state=42)

def test_one(census_sample):
    """
    test that process_data correctly returns feature and label arrays of the expected shape.
    """
    categorical_features = [
        "workclass", "education", "marital-status", "occupation", "relationship",
        "race", "sex", "native-country"
    ]
    label = "salary"
    X, y, encoder, lb = process_data(
        census_sample, categorical_features, label, training=True
    )

    assert X.shape[0] == census_sample.shape[0]
    assert y.shape[0] == census_sample.shape[0]
    assert len(X.shape) == 2
    assert set(np.unique(y)).issubset({0,1 })


def test_two(census_sample):
    """
    Test that train_model returns a fitted model and that it can make predictions.
    """
    categorical_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
   ]
    label = "salary"
    X, y, encoder, lb = process_data(census_sample, categorical_features, label, training=True)
    model = train_model(X, y)

    preds = inference(model, X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0 ,1})



def test_three(census_sample):
    """
   Test that inference returns a Numpy array of expected shape and type.
    """
    categorical_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
    ]
    label = "salary"
    X, y, encoder, lb = process_data(census_sample, categorical_features, label, training=True)
    model = train_model(X, y)
    preds = inference(model ,X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (X.shape[0],)
