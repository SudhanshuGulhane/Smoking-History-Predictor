# 🚀 Smoking History Prediction API (MLOps)

This project is an **end-to-end MLOps pipeline** for predicting a person's smoking history using medical diagnostic data. The system is built with **PyTorch for deep learning**, **Flask for API deployment**, and **Docker & Azure for cloud deployment**.

---

## 🌟 **Features**
✅ **Neural Network-based Smoking Prediction**  
✅ **Automated Data Preprocessing Pipeline**  
✅ **Flask API for Model Deployment**  
✅ **Containerization with Docker**  
✅ **CI/CD Pipeline using GitHub Actions**  
✅ **Cloud Deployment on Azure Container Instances (ACI)**  

---

## 📁 **Project Structure**
```
├── config/                  # model configurations
├── data/                    # Raw & Processed Data
├── models/                  # store trained pytorch model
├── notebooks/               # Jupyter Notebooks for EDA & Training
├── smoking_history_prediction/
│   ├── models/              # Neural Network Model & Training
│   ├── data/                # Data Processing Pipeline
├── tests/                   # Unit Tests
├── .github/workflows/       # GitHub Actions CI/CD Pipeline
├── Dockerfile               # Docker Container Setup
├── requirements.txt         # Dependencies
├── app.py                   # Flask API
├── README.md                # Documentation
└── .gitignore               # Ignore Unnecessary Files
```

---

## 📊 **1. Data Pipeline**
- Loads raw medical data (`CSV`).
- Cleans missing values & standardizes features.
- Saves processed data for training.

---

## 🤖 **2. Model Development**
- **Neural Network** built using **PyTorch**.
- Tracks experiments using **MLflow**.
- Saves trained model as `smoking_nn.pth`.

```bash
python run_scripts.py
```

---

## 🔥 **3. API Deployment (Flask)**
- Flask-based REST API to serve predictions.
- Endpoint: `/predict` (POST request).
```bash
python app.py
```

Example Request:

```json
{
  "features": [0,40,160,50,72.0,1.0,0.5,1.0,1.0,103.0,70.0,85.0,233.0,96.0,117.0,95.0,13.5,1.0,0.8,20.0,14.0,14.0,1.0]
}
```

```
features:[sex,age,height,weight,waistline,sight_left,sight_right,hear_left,hear_right,SBP,DBP,BLDS,tot_chole,HDL_chole,LDL_chole,triglyceride,hemoglobin,urine_protein,serum_creatinine,SGOT_AST,SGOT_ALT,gamma_GTP,DRK_YN]
```
---

## 🐳 **4. Containerization with Docker**
Build and run the Docker container:
```bash
docker build -t smoking-prediction-api .
docker run -p 4000:4000 smoking-prediction-api
```
---

## ☁️ **5. Cloud Deployment (Azure)**
- The Docker container is deployed on **Azure Container Instances (ACI)**.
- Steps:
  1. Push the Docker image to **Azure Container Registry (ACR)**.
  2. Deploy the image to ACI.

```bash
az container create \
    --resource-group myResourceGroup \
    --name smoking-api-container \
    --image myacrregistry.azurecr.io/smoking-prediction-api:v1 \
    --dns-name-label smoking-prediction-app-demo \
    --ports 4000
```

Test API on Azure:
```bash
curl -X POST "http://smoking-prediction-app-demo.eastus.azurecontainer.io:4000/predict" -H "Content-Type: application/json" \
-d '{"features": [0,40,160,50,72.0,1.0,0.5,1.0,1.0,103.0,70.0,85.0,233.0,96.0,117.0,95.0,13.5,1.0,0.8,20.0,14.0,14.0,1.0]}'
```

---

## ⚡ **6. CI/CD Pipeline (GitHub Actions)**
- Every code push triggers:
  1. Automated Tests (`pytest`)
  2. Docker Image Build & Push
  3. Azure Deployment  

File: `.github/workflows/deploy.yml`

---

## 🛠 **Setup & Installation**
1️⃣ **Clone this repo**  
```bash
git clone https://github.com/SudhanshuGulhane/Smoking-History-Predictor.git
```
2️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```
3️⃣ **Run API Locally**  
```bash
python app.py
```

---

## 🎯 **Future Enhancements**
- [ ] Implement model monitoring with **Prometheus & Grafana**.
- [ ] Deploy on **Azure Kubernetes Service (AKS)** for auto-scaling.
- [ ] Improve model explainability with **SHAP**.

---

## 👨‍💻 **Contributors**
- **[Sudhanshu Gulhane]** - _MLOps Engineer & Software Developer_

---

## 📜 **License**
This project is licensed under the **MIT License**.

---

### 🌟 _If you like this project, don't forget to ⭐ it!_

### 📷 **Snapshot of the UI**

![image](https://github.com/user-attachments/assets/7f2cf00c-58d6-40f9-ac7f-6e508f3eb0de)
