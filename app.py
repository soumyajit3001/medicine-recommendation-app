from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# ================= LOAD MODEL =================
svc = pickle.load(open("svc.pkl", "rb"))

# ================= LOAD DATA =================
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
medications = pd.read_csv("medications.csv")
diets = pd.read_csv("diets.csv")
workout = pd.read_csv("workout_df.csv")

# ================= SYMPTOM LIST =================
symptom_list = sym_des.columns.tolist()

# ================= HELPER FUNCTION =================
def helper(dis):
    desc = sym_des[sym_des['Disease'] == dis]['Description'].values
    pre = precautions[precautions['Disease'] == dis].iloc[:, 1:].values.flatten().tolist()
    med = medications[medications['Disease'] == dis]['Medication'].values
    die = diets[diets['Disease'] == dis]['Diet'].values
    wrkout = workout[workout['Disease'] == dis]['Workout'].values

    return {
        "description": desc[0] if len(desc) > 0 else "No description available",
        "precautions": pre,
        "medications": med.tolist(),
        "diet": die.tolist(),
        "workout": wrkout.tolist()
    }

# ================= PREDICTION FUNCTION =================
def get_predicted_value(symptoms):
    input_vector = np.zeros(len(symptom_list))
    for symptom in symptoms:
        if symptom in symptom_list:
            input_vector[symptom_list.index(symptom)] = 1

    prediction = svc.predict([input_vector])[0]
    return prediction

# ================= API ROUTE =================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "symptoms" not in data:
        return jsonify({"error": "Symptoms not provided"}), 400

    user_symptoms = data["symptoms"]

    predicted_disease = get_predicted_value(user_symptoms)
    details = helper(predicted_disease)

    return jsonify({
        "predicted_disease": predicted_disease,
        "description": details["description"],
        "precautions": details["precautions"],
        "medications": details["medications"],
        "diet": details["diet"],
        "workout": details["workout"],
        "disclaimer": "This app is for educational purposes only. Please consult a doctor."
    })

# ================= RUN SERVER =================
@app.route("/")
def home():
    return "Backend is running successfully!"

@app.route("/test")
def test_api():
    symptoms = ["itching", "skin_rash", "fatigue"]

    predicted_disease = get_predicted_value(symptoms)
    details = helper(predicted_disease)

    return jsonify({
        "predicted_disease": predicted_disease,
        "medications": details["medications"],
        "precautions": details["precautions"]
    })


if __name__ == "__main__":
    app.run(debug=True)
