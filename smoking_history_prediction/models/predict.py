import torch
from smoking_history_prediction.models.model import SmokingPredictionNN

def load_model(input_dim, MODEL_PATH):
    model = SmokingPredictionNN(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def predict(model, input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return {"smoking_status": "Yes" if prediction > 0.5 else "No", "probability": round(prediction, 2)}