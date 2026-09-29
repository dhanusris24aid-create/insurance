import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = pd.read_csv(
    r"C:\Users\Dhanuselvakumar\Downloads\vehicle_dataset\extracted\fraud_oracle.csv"
)

print("Dataset loaded successfully!")
print("Shape:", data.shape)


# Features used by our website
features = [
    "Age",
    "Sex",
    "VehicleCategory",
    "VehiclePrice",
    "PastNumberOfClaims",
    "AgeOfVehicle",
    "DriverRating",
    "PoliceReportFiled",
    "WitnessPresent",
    "Fault",
    "PolicyType"
]


# Input and target
X = data[features]
y = data["FraudFound_P"]


# Categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Train
print("Training Random Forest model...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# Test
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model
joblib.dump(pipeline, "vehicle_fraud_model.pkl")

print("\nModel saved successfully!")
print("File: vehicle_fraud_model.pkl")