
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from huggingface_hub import upload_file
import os

HF_TOKEN = os.getenv("HF_TOKEN")

train_df = pd.read_csv("tourism_project/data/train.csv")
test_df = pd.read_csv("tourism_project/data/test.csv")

X_train = train_df.drop("ProdTaken", axis=1)
y_train = train_df["ProdTaken"]

X_test = test_df.drop("ProdTaken", axis=1)
y_test = test_df["ProdTaken"]

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Accuracy:", acc)

joblib.dump(model, "best_model.pkl")

upload_file(
    "best_model.pkl",
    "best_model.pkl",
    "gayathri1909/tourism-model",
    repo_type="model",
    token=HF_TOKEN
)

print("Model training complete")
