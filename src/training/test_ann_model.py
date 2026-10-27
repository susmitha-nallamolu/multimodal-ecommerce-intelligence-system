
import torch
import joblib
from pathlib import Path

from src.models.ann_model import ANNProductClassifier

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Saved files
model_path = PROJECT_ROOT / "models" / "ann_product_classifier.pth"
scaler_path = PROJECT_ROOT / "models" / "ann_feature_scaler.pkl"

# Load scaler
scaler = joblib.load(scaler_path)

# Create model architecture
model = ANNProductClassifier(input_dim=10, num_classes=21)

# Load saved weights

checkpoint = torch.load(
    model_path,
    map_location="cpu",
    weights_only=True
)

# The checkpoint contains metadata and the actual model weights.
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

# Test with one sample containing 10 numerical features
sample = torch.randn(1, 10)

with torch.no_grad():
    output = model(sample)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()

print("ANN model loaded successfully!")
print("Scaler loaded:", scaler is not None)
print("Output shape:", output.shape)
print("Predicted class ID:", predicted_class)
print(f"Test confidence: {confidence:.2%}")

assert output.shape == (1, 21)
print("ANN verification passed!")
