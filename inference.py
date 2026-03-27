import torch
import config
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import warnings

# Suppress specific qnnpack warning
warnings.filterwarnings("ignore", message=".*qnnpack incorrectly ignores reduce_range.*")

print("Loading saved model...")
# Load the saved model (CPU is fine for inference)
loaded_model = DistilBertForSequenceClassification.from_pretrained(config.SAVED_MODEL_DIR).cpu()
tokenizer = DistilBertTokenizerFast.from_pretrained(config.SAVED_MODEL_DIR)

# --- OPTIMIZATION STEP: DYNAMIC QUANTIZATION ---
print("Applying Dynamic Quantization...")
try:
    # Set the quantization engine for ARM64 (Apple Silicon)
    torch.backends.quantized.engine = 'qnnpack'
    
    quantized_model = torch.quantization.quantize_dynamic(
        loaded_model, 
        {torch.nn.Linear},  # Quantize the linear layers
        dtype=torch.qint8
    )
    print("Model Quantized!")
except Exception as e:
    print(f"Quantization failed: {e}")
    print("Falling back to unquantized model for inference.")
    quantized_model = loaded_model

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=config.MAX_LENGTH)
    
    with torch.no_grad():
        logits = quantized_model(**inputs).logits
        
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    
    label_map = {0: "HAM (Safe)", 1: "SPAM (Danger)"}
    return label_map[predicted_class.item()], confidence.item()

if __name__ == "__main__":
    # Test cases
    test_texts = [
        "Hey man, are we still on for gym tomorrow?",
        "URGENT! You have won a $500 Amazon voucher. Call now to claim!"
    ]
    
    for text in test_texts:
        label, conf = predict(text)
        print(f"\nMessage: {text}")
        print(f"Prediction: {label} | Confidence: {conf:.2%}")