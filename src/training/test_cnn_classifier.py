
import torch

from src.models.cnn_classifier import CNNImageClassifier

model = CNNImageClassifier()
model.eval()

dummy_images = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    predictions = model(dummy_images)

print("Input image shape:", dummy_images.shape)
print("Output shape:", predictions.shape)
print("Predicted class IDs:", predictions.argmax(dim=1))
