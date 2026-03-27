import torch

# Paths
MODEL_CKPT = "distilbert-base-uncased"
OUTPUT_DIR = "./results"
SAVED_MODEL_DIR = "./spam_model_finetuned"
LOG_DIR = "./logs"

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128  # Max token length for DistilBERT

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"