import config
from data_loader import SpamDataset
from custom_trainer import WeightedLossTrainer
from transformers import DistilBertForSequenceClassification, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Prepare Data
print("Loading and processing data...")
spam_data = SpamDataset()
datasets, weights_tensor = spam_data.prepare_data()

# 2. Metrics Function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# 3. Model Initialization
print("Initializing Model...")
model = DistilBertForSequenceClassification.from_pretrained(
    config.MODEL_CKPT, 
    num_labels=2
).to(config.DEVICE)

# 4. Training Args (UPDATED)
training_args = TrainingArguments(
    output_dir=config.OUTPUT_DIR,
    num_train_epochs=config.EPOCHS,
    learning_rate=config.LEARNING_RATE,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",           # <--- CHANGED THIS LINE
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# 5. Initialize Custom Trainer
trainer = WeightedLossTrainer(
    class_weights=weights_tensor,
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    compute_metrics=compute_metrics,
)

# 6. Train & Save
print("Starting Training...")
trainer.train()

print(f"Saving model to {config.SAVED_MODEL_DIR}...")
model.save_pretrained(config.SAVED_MODEL_DIR)
spam_data.tokenizer.save_pretrained(config.SAVED_MODEL_DIR)
print("Training Complete!")