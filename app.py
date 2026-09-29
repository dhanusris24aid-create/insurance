import os
from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained Random Forest model
model = joblib.load("vehicle_fraud_model.pkl")


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # --------------------------------
        # GET USER INPUT
        # --------------------------------

        age = int(data["age"])
        gender = data["gender"]
        vehicle_type = data["vehicleType"]
        vehicle_age = int(data["vehicleAge"])
        experience = int(data["experience"])
        previous_claims = data["previousClaims"]
        income = float(data["income"])
        premium = float(data["premium"])
        police_report = data["policeReport"]

        # --------------------------------
        # CONVERT VEHICLE TYPE
        # --------------------------------

        if vehicle_type == "Car":
            vehicle_category = "Sedan"

        elif vehicle_type == "SUV":
            vehicle_category = "Utility"

        elif vehicle_type == "Two Wheeler":
            vehicle_category = "Utility"

        elif vehicle_type == "Truck":
            vehicle_category = "Utility"

        else:
            vehicle_category = "Sedan"

        # --------------------------------
        # CONVERT VEHICLE AGE
        # --------------------------------

        if vehicle_age == 0:
            age_of_vehicle = "new"

        elif vehicle_age <= 2:
            age_of_vehicle = "2 years"

        elif vehicle_age == 3:
            age_of_vehicle = "3 years"

        elif vehicle_age == 4:
            age_of_vehicle = "4 years"

        elif vehicle_age == 5:
            age_of_vehicle = "5 years"

        elif vehicle_age == 6:
            age_of_vehicle = "6 years"

        elif vehicle_age == 7:
            age_of_vehicle = "7 years"

        else:
            age_of_vehicle = "more than 7"

        # --------------------------------
        # CONVERT PREVIOUS CLAIMS
        # --------------------------------

        if previous_claims == "None":
            past_claims = "none"

        elif previous_claims == "1":
            past_claims = "1"

        elif previous_claims == "2 to 4":
            past_claims = "2 to 4"

        else:
            past_claims = "more than 4"

        # --------------------------------
        # DRIVER RATING
        # --------------------------------

        if experience <= 2:
            driver_rating = 1

        elif experience <= 5:
            driver_rating = 2

        elif experience <= 10:
            driver_rating = 3

        else:
            driver_rating = 4

        # --------------------------------
        # VEHICLE PRICE CATEGORY
        # --------------------------------

        if income < 300000:
            vehicle_price = "less than 20000"

        elif income < 500000:
            vehicle_price = "20000 to 29000"

        elif income < 800000:
            vehicle_price = "30000 to 39000"

        elif income < 1200000:
            vehicle_price = "40000 to 59000"

        elif income < 2000000:
            vehicle_price = "60000 to 69000"

        else:
            vehicle_price = "more than 69000"

        # --------------------------------
        # CREATE MODEL INPUT
        # --------------------------------

        input_data = pd.DataFrame([{

            "Age": age,
            "Sex": gender,
            "VehicleCategory": vehicle_category,
            "VehiclePrice": vehicle_price,
            "PastNumberOfClaims": past_claims,
            "AgeOfVehicle": age_of_vehicle,
            "DriverRating": driver_rating,
            "PoliceReportFiled": police_report,
            "WitnessPresent": "No",
            "Fault": "Policy Holder",
            "PolicyType": "Sedan - Liability"

        }])

        # --------------------------------
        # REAL RANDOM FOREST PREDICTION
        # --------------------------------

        prediction = model.predict(input_data)[0]

        ml_probability = model.predict_proba(
            input_data
        )[0][1] * 100

        ml_probability = round(float(ml_probability), 2)

        # --------------------------------
        # FINAL RESULT
        # --------------------------------

        if prediction == 1:

            risk_level = "HIGH"

            explanation = (
                "The machine learning model indicates a higher "
                "possibility of fraudulent activity. Additional "
                "claim verification and review may be required."
            )

        else:

            risk_level = "LOW"

            explanation = (
                "The machine learning model indicates a lower "
                "possibility of fraudulent activity. The claim "
                "can continue through the normal verification process."
            )

        # --------------------------------
        # RISK FACTORS
        # --------------------------------

        risk_factors = []

        if age < 25:
            risk_factors.append("Young driver")

        if vehicle_age >= 7:
            risk_factors.append("Older vehicle")

        if previous_claims == "1":
            risk_factors.append("Previous insurance claim")

        elif previous_claims == "2 to 4":
            risk_factors.append("Multiple previous claims")

        elif previous_claims == "more than 4":
            risk_factors.append("Many previous claims")

        if police_report == "Yes":
            risk_factors.append("Police report filed")

        if experience <= 2:
            risk_factors.append("Low driving experience")

        if premium >= 30000:
            risk_factors.append("Higher annual premium")

        if income < 400000:
            risk_factors.append("Lower annual income")

        # --------------------------------
        # SEND RESULT TO WEBSITE
        # --------------------------------

        return jsonify({

            "prediction": risk_level,

            "probability": ml_probability,

            "result": risk_level + " Risk",

            "risk_factors": risk_factors,

            "explanation": explanation,

            "ml_probability": ml_probability

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)