# Spam Message Detection Model

This repository contains a machine learning pipeline for detecting spam SMS messages using a fine-tuned DistilBERT model. 

## Overview
The model uses `distilbert-base-uncased` from the Hugging Face Transformers library and is trained on the `sms_spam` dataset. It handles class imbalance by utilizing weighted loss and features a complete end-to-end pipeline from data preparation to inference. 

For optimized inference on Apple Silicon (ARM64) and CPUs, the model utilizes dynamic PyTorch quantization.

## Project Structure
- `config.py`: Contains training hyperparameters, file paths, and device configurations.
- `data_loader.py`: Handles fetching the `sms_spam` dataset, tokenization, creating train/val/test splits, and computing class weights for data imbalance.
- `train.py`: Initializes the model and dataset, sets up the training loop using a custom trainer (supporting weighted loss), and saves the final model to `./spam_model_finetuned`.
- `custom_trainer.py`: Custom Hugging Face Trainer implementation supporting weighted loss (used during training).
- `inference.py`: Loads the fine-tuned model and applies dynamic quantization for faster inference. Provides a simple `predict(text)` function to classify messages as HAM or SPAM.
- `requirements.txt`: Required Python packages.

## Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
To train the model from scratch and save the checkpoints:
```bash
python train.py
```
This will download the dataset and the base model, train the sequence classifier, and save the best model locally.

### Inference
To make predictions using the trained model, run:
```bash
python inference.py
```
The script includes a few test messages to verify the inference pipeline is working correctly. It outputs the predicted label and confidence score.

## Technical Details
- **Architecture**: DistilBERT (`distilbert-base-uncased`)
- **Optimization**: Dynamic Quantization (INT8 via `qnnpack`) used during inference for enhanced performance.
- **Evaluation Metrics**: Accuracy, F1-score, Precision, and Recall.
- **Frameworks**: PyTorch, Hugging Face Transformers, & Datasets.
