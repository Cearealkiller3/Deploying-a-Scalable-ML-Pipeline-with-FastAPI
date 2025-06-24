import json

import requests


r = requests.get('http://127.0.0.1:8000')


print(f"Status code: {r.status_code}")

print(f"Result: {r.json()['message']}")



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education_num": 10,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}

r = requests.post('http://127.0.0.1:8000/model', json=data)


print(f'Status code: {r.status_code}')
print(f'Result: {r.json()['prediction']}')
