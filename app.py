import json
import os
from pathlib import Path

import joblib
import mysql.connector
import pandas as pd
from flask import Flask, current_app, jsonify, request, render_template

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__, static_folder=str(BASE_DIR / "templates" / "static"))
DATABASE_NAME = "factorysenseiot"


class HistoryDatabaseConfigurationError(Exception):
    pass


# Load trained models
rul_model = joblib.load(MODEL_DIR / "rul_regressor.joblib")
rul_scaler = joblib.load(MODEL_DIR / "rul_scaler.joblib")
failure_model = joblib.load(MODEL_DIR / "failure_classifier.joblib")
rul_metadata = json.loads(
    (MODEL_DIR / "rul_metadata.json").read_text(encoding="utf-8")
)

# Features expected by the NASA RUL model
RUL_FEATURES = [
    "op_setting_1", "op_setting_2",
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_5", "sensor_6", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11",
    "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_16", "sensor_17",
    "sensor_20", "sensor_21"
]

# Features expected by the AI4I failure model
FAILURE_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


def get_database_connection():
    password = os.getenv("FACTORYSENSE_DB_PASSWORD")
    if password is None:
        raise HistoryDatabaseConfigurationError(
            "Set FACTORYSENSE_DB_PASSWORD before using prediction history."
        )

    try:
        port = int(os.getenv("FACTORYSENSE_DB_PORT", "3306"))
    except ValueError as error:
        raise HistoryDatabaseConfigurationError(
            "FACTORYSENSE_DB_PORT must be an integer."
        ) from error

    return mysql.connector.connect(
        host=os.getenv("FACTORYSENSE_DB_HOST", "127.0.0.1"),
        port=port,
        user=os.getenv("FACTORYSENSE_DB_USER", "root"),
        password=password,
        database=DATABASE_NAME,
        connection_timeout=3
    )


def save_prediction_history(
    machine_id,
    prediction_type,
    input_data,
    predicted_result,
    model_name,
    failure_probability=None,
    predicted_rul=None
):
    connection = get_database_connection()
    cursor = None

    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_history (
                machine_id,
                prediction_type,
                input_data,
                predicted_result,
                failure_probability,
                predicted_rul,
                model_name
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                machine_id,
                prediction_type,
                json.dumps(input_data),
                json.dumps(predicted_result),
                failure_probability,
                predicted_rul,
                model_name
            )
        )
        connection.commit()
    except mysql.connector.Error:
        connection.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def check_history_database():
    connection = None
    cursor = None

    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1 FROM prediction_history LIMIT 0")
        return True
    except (mysql.connector.Error, HistoryDatabaseConfigurationError) as error:
        current_app.logger.warning("Prediction history database unavailable: %s", error)
        return False
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


def decode_json_field(value):
    return json.loads(value) if isinstance(value, str) else value


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    history_database_connected = check_history_database()
    return jsonify({
        "status": "success",
        "application": "FactorySenseIoT",
        "rul_model_loaded": True,
        "failure_model_loaded": True,
        "history_database_connected": history_database_connected
    })


@app.route("/api/predictions/history", methods=["GET"])
def prediction_history():
    prediction_type = request.args.get("prediction_type")
    machine_id = request.args.get("machine_id")

    if prediction_type and prediction_type not in {"failure", "rul"}:
        return jsonify({
            "error": "prediction_type must be 'failure' or 'rul'."
        }), 400

    if machine_id is not None and len(machine_id) > 100:
        return jsonify({"error": "machine_id must be 100 characters or fewer."}), 400

    try:
        limit = int(request.args.get("limit", "50"))
    except ValueError:
        return jsonify({"error": "limit must be an integer."}), 400

    if not 1 <= limit <= 200:
        return jsonify({"error": "limit must be between 1 and 200."}), 400

    filters = []
    parameters = []
    if prediction_type:
        filters.append("prediction_type = %s")
        parameters.append(prediction_type)
    if machine_id:
        filters.append("machine_id = %s")
        parameters.append(machine_id)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = f"""
        SELECT
            prediction_id,
            machine_id,
            prediction_type,
            input_data,
            predicted_result,
            failure_probability,
            predicted_rul,
            risk_level,
            model_name,
            created_at
        FROM prediction_history
        {where_clause}
        ORDER BY created_at DESC, prediction_id DESC
        LIMIT %s
    """
    parameters.append(limit)

    connection = None
    cursor = None
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query, parameters)
        rows = cursor.fetchall()

        for row in rows:
            row["input_data"] = decode_json_field(row["input_data"])
            row["predicted_result"] = decode_json_field(row["predicted_result"])
            row["failure_probability"] = (
                float(row["failure_probability"])
                if row["failure_probability"] is not None
                else None
            )
            row["predicted_rul"] = (
                float(row["predicted_rul"])
                if row["predicted_rul"] is not None
                else None
            )
            row["created_at"] = row["created_at"].isoformat(
                sep=" ",
                timespec="seconds"
            )

        return jsonify({"predictions": rows, "count": len(rows)})
    except (mysql.connector.Error, HistoryDatabaseConfigurationError) as error:
        current_app.logger.exception("Could not read prediction history")
        return jsonify({
            "error": f"Could not read prediction history: {error}"
        }), 503
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


@app.route("/api/predict/rul", methods=["POST"])
def predict_rul():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object."}), 400

    missing = [feature for feature in RUL_FEATURES if feature not in data]
    if missing:
        return jsonify({
            "error": "Missing required NASA features.",
            "missing_features": missing
        }), 400

    try:
        values = pd.DataFrame(
            [[float(data[feature]) for feature in RUL_FEATURES]],
            columns=RUL_FEATURES
        )

        if rul_metadata["model_type"] == "RandomForestRegressor":
            prediction = float(rul_model.predict(values)[0])
        elif rul_metadata["model_type"] == "LinearRegression":
            scaled_values = rul_scaler.transform(values)
            prediction = float(rul_model.predict(scaled_values)[0])
        else:
            raise RuntimeError(
                f"Unsupported RUL model type: {rul_metadata['model_type']}"
            )

        result = {
            "predicted_rul_cycles": round(max(0.0, prediction), 2),
            "unit": "operating cycles",
            "model": "NASA C-MAPSS FD001"
        }
        machine_id = data.get("machine_id")
        if machine_id is not None and (
            not isinstance(machine_id, str) or len(machine_id) > 100
        ):
            return jsonify({
                "error": "machine_id must be a string of 100 characters or fewer."
            }), 400

        try:
            save_prediction_history(
                machine_id=machine_id or None,
                prediction_type="rul",
                input_data=values.iloc[0].to_dict(),
                predicted_result=result,
                model_name=result["model"],
                predicted_rul=result["predicted_rul_cycles"]
            )
            result["history_saved"] = True
        except (mysql.connector.Error, HistoryDatabaseConfigurationError) as error:
            current_app.logger.exception("Could not save RUL prediction history")
            result["history_saved"] = False
            result["history_warning"] = (
                "Prediction calculated, but not saved to history: "
                f"{error}"
            )

        return jsonify(result)

    except (TypeError, ValueError) as error:
        return jsonify({"error": f"Invalid sensor values: {error}"}), 400


@app.route("/api/predict/failure", methods=["POST"])
def predict_failure():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object."}), 400

    missing = [feature for feature in FAILURE_FEATURES if feature not in data]
    if missing:
        return jsonify({
            "error": "Missing required AI4I features.",
            "missing_features": missing
        }), 400

    try:
        values = pd.DataFrame(
            [{
                "Type": str(data["Type"]),
                "Air temperature [K]": float(data["Air temperature [K]"]),
                "Process temperature [K]": float(data["Process temperature [K]"]),
                "Rotational speed [rpm]": float(data["Rotational speed [rpm]"]),
                "Torque [Nm]": float(data["Torque [Nm]"]),
                "Tool wear [min]": float(data["Tool wear [min]"])
            }],
            columns=FAILURE_FEATURES
        )

        prediction = int(failure_model.predict(values)[0])
        probability = failure_model.predict_proba(values)[0][1]

        result = {
            "predicted_failure": prediction,
            "status": "Failure risk detected" if prediction == 1 else "No failure predicted",
            "failure_probability_percent": round(float(probability) * 100, 2),
            "model": "AI4I 2020"
        }
        machine_id = data.get("machine_id")
        if machine_id is not None and (
            not isinstance(machine_id, str) or len(machine_id) > 100
        ):
            return jsonify({
                "error": "machine_id must be a string of 100 characters or fewer."
            }), 400

        try:
            save_prediction_history(
                machine_id=machine_id or None,
                prediction_type="failure",
                input_data=values.iloc[0].to_dict(),
                predicted_result=result,
                model_name=result["model"],
                failure_probability=result["failure_probability_percent"]
            )
            result["history_saved"] = True
        except (mysql.connector.Error, HistoryDatabaseConfigurationError) as error:
            current_app.logger.exception("Could not save failure prediction history")
            result["history_saved"] = False
            result["history_warning"] = (
                "Prediction calculated, but not saved to history: "
                f"{error}"
            )

        return jsonify(result)

    except (TypeError, ValueError) as error:
        return jsonify({"error": f"Invalid sensor values: {error}"}), 400


if __name__ == "__main__":
    app.run(debug=True)