import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS

model = load("decision_tree_model.joblib")

feature_names = [
    "Education Level", "Institution Type", "Gender", "Age", "Device",
    "IT Student", "Location", "Financial Condition", "Internet Type", "Network Type"
]

api = Flask(__name__)
CORS(api)

@api.route('/api/students_adaptability_prediction', methods=['POST'])
def predict_adaptability():
    try:
        data_list = request.json.get('inputs', [])
        if not data_list or not isinstance(data_list, list):
            return jsonify({"error": "No valid input data received"}), 400

        prepared_inputs = []
        for data in data_list:
            full_data = {}
            for feature in feature_names:
                try:
                    full_data[feature] = float(data.get(feature, 0))
                except (ValueError, TypeError):
                    full_data[feature] = 0
            prepared_inputs.append(full_data)

        input_df = pd.DataFrame(prepared_inputs)
        prediction_probs = model.predict_proba(input_df)
        class_labels = model.classes_

        response = []
        for prob in prediction_probs:
            prob_dict = {}
            for k, v in zip(class_labels, prob):
                prob_dict[str(k)] = round(float(v) * 100, 2)
            response.append(prob_dict)

        return jsonify({'prediction': response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred during the prediction process."}), 500

if __name__ == '__main__':
    api.run(port=8000, debug=True)
