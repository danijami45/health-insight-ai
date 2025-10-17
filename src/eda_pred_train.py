#Importin important libraries
import pandas as pd
import numpy as np
import joblib
import os
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

#Data loading
#Fetching dataset, id=45 is for heart desease
dataset = fetch_ucirepo(id=45)

#get the full orignal dataframe
df = dataset.data.original
print(f"Data Shape: {df.shape}")

#Feature Engineering and Data Preparation
#Simplify target and convert num > 0 = 1 (desease present)
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.drop('num', axis=1, inplace=True)

# Handle missing values
df = df.fillna(df.median())

# Split features/labels, Separate index (X) and output (y)
X = df.drop('target', axis=1)
y = df['target']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 42, stratify =y )
print("Train Shape: ", X_train.shape, "\nTest Shape: ", X_test.shape)

#Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nData scalled successfully.")

# Save scaler
joblib.dump(scaler, "../models/scaler.pkl")
print("Data Prepared & Scaler Saved")

#Model Training and Tracking#
#############################

mlflow.set_experiment("HealthInsightAI")

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc_score = accuracy_score(y_test, y_pred)

    print(f"Model trained successfully with Accuracy: {acc_score:.4f}")

    # Log metrics & params
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc_score)

    # Save model locally and to MLflow
    joblib.dump(model, "../models/heart_model.pkl")
    mlflow.sklearn.log_model(model, "model")

#Model Evaluation and Logging#
##############################

report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

pd.DataFrame({"accuracy": [acc_score]}).to_csv("../logs/metrics.csv", index=False)
with open("../logs/training_log.txt", "a") as f:
    f.write(f"Model trained successfully with Accuracy: {acc_score:.4f}\n")

print("Evaluation Complete!")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Artifacts saved in /models and /logs folders.")
