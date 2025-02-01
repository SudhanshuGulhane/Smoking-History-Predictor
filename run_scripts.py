import os
import logging
import yaml
from smoking_history_prediction.data.preprocess import load_data, clean_and_process_data, save_data
from smoking_history_prediction.models.train import load_and_prepare_data, train_and_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DATA_PATH = os.path.join("data", "raw", "smoking_driking_dataset.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "smoking_driking_dataset_cleaned.csv")
CONFIG_PATH = os.path.join("config", "model_config.yaml")

def load_config(config_path):
    """Load model hyperparameters from YAML config file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info("Config file loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

def preprocess_data():
    """Load, clean, and save processed data."""
    try:
        logging.info("Loading raw data...")
        data = load_data(RAW_DATA_PATH)
        
        logging.info("Cleaning and processing data...")
        data = clean_and_process_data(data)
        
        logging.info(f"Saving processed data to {PROCESSED_DATA_PATH}...")
        save_data(data, PROCESSED_DATA_PATH)
        
        logging.info("Data preprocessing completed successfully.")
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise

def train_and_test_model():
    """Prepare data and train/test the model."""
    try:
        logging.info("Loading configuration...")
        config = load_config(CONFIG_PATH)

        logging.info("Preparing training and testing datasets...")
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_prepare_data(
            PROCESSED_DATA_PATH, config["test_size"]
        )

        logging.info("Starting model training and testing...")
        train_and_test(
            X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
            config["batch_size"], config["learning_rate"], config["epochs"]
        )

        logging.info("Model evaluation completed successfully.")
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise

if __name__ == "__main__":
    logging.info("Starting the pipeline...")

    # # Preprocess Data
    # preprocess_data()

    # logging.info("Data pre-processing pipeline execution completed.")

    # # Train and Test Model
    # train_and_test_model()

    # logging.info("Training and testing pipeline execution completed.")
