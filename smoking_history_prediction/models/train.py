import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from smoking_history_prediction.models.model import SmokingPredictionNN

def load_and_prepare_data(DATA_PATH, test_size):
    df = pd.read_csv(DATA_PATH)
    X = df.drop("SMK_stat_type_cd",axis=1).values
    y = df["SMK_stat_type_cd"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def train_and_test(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size, learning_rate, epochs):

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

    input_dim = X_train_tensor.shape[1]
    model = SmokingPredictionNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("Smoking_Prediction_Experiment")

    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    mlflow.log_metric("final_loss", loss.item())
    mlflow.pytorch.log_model(model, "model")
    
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).round()
        accuracy = accuracy_score(y_test_tensor, y_pred_test)
        print(f"Model Accuracy: {accuracy:.2f}")
    
    torch.save(model.state_dict(), "smoking_nn.pth")