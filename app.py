import os
from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("vehicle_fraud_model.pkl")


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        age = int(data.get("age", 0))
        gender = data.get("gender", "Male")
        vehicle_type = data.get("vehicleType", "Car")
        vehicle_age = int(data.get("vehicleAge", 0))
        experience = int(data.get("experience", 0))
        previous_claims = data.get("previousClaims", "None")
        income = float(data.get("income", 0))
        premium = float(data.get("premium", 0))
        police_report = data.get("policeReport", "Yes")

        policy_active = data.get("policyActive", "Yes")
        emergency = data.get("emergency", "Yes")

        # -----------------------------
        # VEHICLE CATEGORY
        # -----------------------------
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

        # -----------------------------
        # VEHICLE AGE
        # -----------------------------
        if vehicle_age == 0:
            age_of_vehicle = "new"
        elif vehicle_age <= 2:
            age_of_vehicle = "2 years"
        elif vehicle_age <= 4:
            age_of_vehicle = "4 years"
        elif vehicle_age <= 6:
            age_of_vehicle = "6 years"
        elif vehicle_age <= 7:
            age_of_vehicle = "7 years"
        else:
            age_of_vehicle = "more than 7"

        # -----------------------------
        # PREVIOUS CLAIMS
        # -----------------------------
        if previous_claims == "None":
            past_claims = "none"
        elif previous_claims == "1":
            past_claims = "1"
        elif previous_claims == "2 to 4":
            past_claims = "2 to 4"
        else:
            past_claims = "more than 4"

        # -----------------------------
        # DRIVER RATING
        # -----------------------------
        if experience <= 2:
            driver_rating = 1
        elif experience <= 5:
            driver_rating = 2
        elif experience <= 10:
            driver_rating = 3
        else:
            driver_rating = 4

        # -----------------------------
        # VEHICLE PRICE
        # -----------------------------
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

        # -----------------------------
        # ML MODEL INPUT
        # -----------------------------
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

        # ML probability
        ml_probability = float(
            model.predict_proba(input_data)[0][1] * 100
        )

        # -----------------------------
        # RISK FACTORS
        # -----------------------------
        risk_factors = []
        risk_points = 0

        if age < 25:
            risk_points += 15
            risk_factors.append("Young driver")

        if vehicle_age >= 7:
            risk_points += 15
            risk_factors.append("Older vehicle")

        if experience <= 2:
            risk_points += 15
            risk_factors.append("Low driving experience")

        if previous_claims == "1":
            risk_points += 10
            risk_factors.append("Previous insurance claim")

        elif previous_claims == "2 to 4":
            risk_points += 20
            risk_factors.append("Multiple previous claims")

        elif previous_claims == "more than 4":
            risk_points += 35
            risk_factors.append("Many previous claims")

        if premium >= 30000:
            risk_points += 10
            risk_factors.append("Higher annual premium")

        if income < 400000:
            risk_points += 10
            risk_factors.append("Lower annual income")

        if emergency == "Yes" and police_report == "No":
            risk_points += 15
            risk_factors.append(
                "Emergency reported without police report"
            )

        # -----------------------------
        # FINAL RISK SCORE
        # -----------------------------
        final_probability = min(risk_points, 100)

        # Let the ML score add risk without lowering the rule-based score.
        ml_adjusted_score = (
            (final_probability * 0.85) +
            (ml_probability * 0.15)
        )
        final_probability = round(
            max(final_probability, ml_adjusted_score),
            2
        )

        final_probability = min(
            max(final_probability, 0),
            100
        )

        # -----------------------------
        # RISK LEVEL
        # -----------------------------
        if final_probability < 40:

            risk_level = "LOW RISK"

            explanation = (
                "The entered information shows a lower "
                "possibility of fraudulent activity. "
                "Normal claim verification may continue."
            )

        elif final_probability < 70:

            risk_level = "MEDIUM RISK"

            explanation = (
                "The entered information shows a moderate "
                "possibility of fraudulent activity. "
                "Additional verification may be required."
            )

        else:

            risk_level = "HIGH RISK"

            explanation = (
                "The entered information shows a higher "
                "possibility of fraudulent activity. "
                "Additional verification and review "
                "may be required."
            )

        # -----------------------------
        # CLAIM ELIGIBILITY
        # -----------------------------
        if policy_active == "No":

            claim_status = "NOT ELIGIBLE"

            claim_reason = (
                "The insurance policy is inactive. "
                "Please verify the policy before submitting "
                "the claim."
            )

            next_steps = [
                "Verify your insurance policy status.",
                "Contact the insurance provider.",
                "Submit the claim when the policy is active."
            ]

        elif final_probability >= 70:

            claim_status = "NOT ELIGIBLE"

            claim_reason = (
                "The claim has a high fraud-risk score and "
                "does not meet the eligibility criteria."
            )

            next_steps = [
                "Review the submitted claim details.",
                "Contact the insurance provider for guidance.",
                "Provide supporting documents if requested."
            ]

        elif final_probability >= 40:

            claim_status = "NEEDS VERIFICATION"

            claim_reason = (
                "The claim has a medium risk score. Additional "
                "verification is required before processing."
            )

            next_steps = [
                "Keep all supporting documents ready.",
                "Verify the accident and claim details.",
                "Submit the claim for insurance review."
            ]

        else:

            claim_status = "LIKELY ELIGIBLE"

            claim_reason = (
                "Based on the information entered, the claim "
                "can proceed to the normal insurance verification "
                "process. Final approval depends on policy terms "
                "and insurer verification."
            )

            next_steps = [
                "Keep your insurance policy details ready.",
                "Keep accident-related documents ready.",
                "Submit the claim for insurance verification."
            ]

        # -----------------------------
        # SEND RESULT
        # -----------------------------
        return jsonify({
            "prediction": risk_level,
            "risk_level": risk_level,
            "probability": round(final_probability, 2),
            "ml_probability": round(ml_probability, 2),
            "risk_factors": risk_factors,
            "claim_status": claim_status,
            "claim_reason": claim_reason,
            "next_steps": next_steps,
            "explanation": explanation
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )