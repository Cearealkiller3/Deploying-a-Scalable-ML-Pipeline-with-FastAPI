import os

import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)
print(f"data shape: {data.shape}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

    print(f"fold {fold+1}")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print("=" * 40)




categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

label = "salary"


X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features,
    label,
    True
    )

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=categorical_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)


model = train_model(X_train, y_train)

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# load the model
model = load_model(
    model_path
) 


preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# iterate through the categorical features
for col in categorical_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=categorical_features,
            label=label,
            encoder=encoder,
            lb=lb,
            model=model
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
