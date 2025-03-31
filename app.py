import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained model
with open("model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

app = Flask(__name__)

def preprocess_input_data(input_data):
    required_columns = ["fear_attention", "anxious_speaking", "avoid_strangers",
                        "excessive_worry", "uncomfortable_around_people",
                        "under_confidence", "physical_symptoms",
                        "sleep_disturbances", "avoid_gatherings",
                        "avoid_eye_contact"]

    input_df = pd.DataFrame([input_data])
    
    for col in required_columns:
        if col not in input_df.columns:
            return f"Missing required feature: {col}", False
    
    return input_df, True

@app.route('/predict', methods=['POST'])
def predict_anxiety_level():
    data = request.get_json()

    # Preprocess input
    input_df, valid = preprocess_input_data(data)
    if not valid:
        return jsonify({"error": input_df}), 400

    # Predict
    prediction = rf_model.predict(input_df)
    
    # Calculate total score
    total_score = input_df.sum(axis=1).values[0]

    # Determine anxiety level
    if total_score < 18:
        anxiety_level = 'Mild'
    elif 18 <= total_score <= 24:
        anxiety_level = 'Moderate'
    else:
        anxiety_level = 'Severe'

    return jsonify({"prediction": anxiety_level})

if __name__ == "__main__":
    app.run(debug=True)
