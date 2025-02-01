from flask import Flask, request, jsonify
import numpy as np
from smoking_history_prediction.models.predict import load_model, predict

app = Flask(__name__)

input_dim = 23
model_path = "smoking_nn.pth"
model = load_model(input_dim, model_path)

@app.route("/")
def home():
    return jsonify({"message": "Smoking Prediction API is running."})

@app.route("/predict", methods=["POST"])
def predict_smoking():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Invalid input format. Provide 'features' list."}), 400

        input_data = np.array(data["features"]).reshape(1, -1) 
        result = predict(model, input_data)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4000)
