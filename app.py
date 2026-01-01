from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# ==========================
# Load Trained Models & Scaler
# ==========================
rf = joblib.load("models/rf.pkl") #rf.pkl → trained Random Forest model
xgb = joblib.load("models/xgb.pkl") #xgb.pkl → trained XGBoost model
scaler = joblib.load("models/scaler.pkl") #scaler.pkl → used to normalize & reverse-normalize data

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        lat = float(request.form["latitude"])
        lon = float(request.form["longitude"])
        depth = float(request.form["depth"])

        # Create dummy sequence (same as training)
        seq_len = 5
        sequence = np.array([[lat, lon, depth, 0]] * seq_len)

        # Scale input
        sequence_scaled = scaler.transform(sequence)

        # Flatten (5 x 3 = 15 features)
        X_flat = sequence_scaled[:, :-1].reshape(1, -1)

        # Predict (scaled)
        rf_scaled = rf.predict(X_flat)[0]
        xgb_scaled = xgb.predict(X_flat)[0]

        # Inverse scaling
        dummy_rf = np.zeros((1, 4))
        dummy_rf[0, -1] = rf_scaled
        rf_real = scaler.inverse_transform(dummy_rf)[0, -1]

        dummy_xgb = np.zeros((1, 4))
        dummy_xgb[0, -1] = xgb_scaled
        xgb_real = scaler.inverse_transform(dummy_xgb)[0, -1]

        prediction = {
            "Random Forest": round(rf_real, 2),
            "XGBoost": round(xgb_real, 2)
        }

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__": # starts flask server
    app.run(debug=True)
