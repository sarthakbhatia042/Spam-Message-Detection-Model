import torch
import numpy as np
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from sklearn.utils.class_weight import compute_class_weight
import config

class SpamDataset:
    def __init__(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_CKPT)
        self.dataset = load_dataset("sms_spam")
        
    def tokenize_function(self, batch):
        return self.tokenizer(
            batch['sms'], 
            padding="max_length", 
            truncation=True, 
            max_length=config.MAX_LENGTH
        )

    def prepare_data(self):
        # 1. Tokenize
        encoded_data = self.dataset.map(self.tokenize_function, batched=True)
        encoded_data = encoded_data.rename_column("label", "labels")
        
        # 2. Split Data (Train: 80%, Val: 10%, Test: 10%)
        # Note: We split BEFORE setting the format to ensure it sticks
        train_test = encoded_data["train"].train_test_split(test_size=0.2, seed=42)
        test_val = train_test["test"].train_test_split(test_size=0.5, seed=42)

        datasets = {
            "train": train_test["train"],
            "validation": test_val["train"],
            "test": test_val["test"]
        }

        # 3. Set Format for PyTorch (Explicitly on the splits)
        for split in datasets:
            datasets[split].set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # 4. Calculate Class Weights for Imbalance
        # FIX: Use np.array() to wrap the data, which handles both Tensors and Lists safely
        print("Calculating class weights...")
        train_labels = np.array(datasets["train"]["labels"])
        
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels
        )
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(config.DEVICE)
        
        print(f"Class Weights: {weights_tensor}")

        return datasets, weights_tensor