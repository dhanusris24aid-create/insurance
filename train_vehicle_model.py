import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load vehicle dataset
data_path = r"C:\Users\Dhanuselvakumar\Downloads\vehicle_dataset\extracted\fraud_oracle.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFraud distribution:")
print(df["FraudFound_P"].value_counts())


# Features used by the ML model
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

X = df[features]
y = df["FraudFound_P"]


# Categorical features
categorical_columns = [
    "Sex",
    "VehicleCategory",
    "VehiclePrice",
    "PastNumberOfClaims",
    "AgeOfVehicle",
    "PoliceReportFiled",
    "WitnessPresent",
    "Fault",
    "PolicyType"
]


# Numerical features
numerical_columns = [
    "Age",
    "DriverRating"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "num",
            "passthrough",
            numerical_columns
        )
    ]
)


# Machine Learning Model
# Higher weight is given to fraudulent claims
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight={0: 1, 1: 5},
    min_samples_leaf=2
)


# Complete ML pipeline
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
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train the model
print("\nTraining model...")

pipeline.fit(X_train, y_train)


# Test the model
predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========================================")
print("Vehicle Insurance Fraud Model Completed")
print("========================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# Save trained model
model_path = "vehicle_fraud_model.pkl"

joblib.dump(pipeline, model_path)

print("\nModel saved successfully as:")
print(model_path)

print("\nTraining completed successfully!")