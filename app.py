
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "final_random_forest_model.pkl"
)

model = joblib.load(MODEL_PATH)

REQUIRED_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SuperKart Sales Prediction API is running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body is empty"
            }), 400

        missing_features = [
            feature for feature in REQUIRED_FEATURES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        input_data = pd.DataFrame(
            [[data[feature] for feature in REQUIRED_FEATURES]],
            columns=REQUIRED_FEATURES
        )

        prediction = model.predict(input_data)[0]

        return jsonify({
            "predicted_sales": round(float(prediction), 2)
        })

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        data = request.get_json()

        if not isinstance(data, list) or len(data) == 0:
            return jsonify({
                "error": "Request body must be a non-empty list"
            }), 400

        input_data = pd.DataFrame(data)

        missing_features = [
            feature for feature in REQUIRED_FEATURES
            if feature not in input_data.columns
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        input_data = input_data[REQUIRED_FEATURES]

        predictions = model.predict(input_data)

        results = [
            round(float(prediction), 2)
            for prediction in predictions
        ]

        return jsonify({
            "predictions": results
        })

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )
