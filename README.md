# Spam Message Detection Model

This repository contains a machine learning pipeline for detecting spam SMS messages using a fine-tuned DistilBERT model. 

## Overview
The model uses `distilbert-base-uncased` from the Hugging Face Transformers library and is trained on the `sms_spam` dataset. It handles class imbalance by utilizing weighted loss and features a complete end-to-end pipeline from data preparation to inference. 

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
