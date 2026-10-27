
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification

project_root = Path(
    r"C:\Users\susmi\multimodal_ecommerce_intelligence"
)

checkpoint_path = (
    project_root
    / "models"
    / "transformer_distilbert_3epoch"
    / "best_model.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

state_dict = checkpoint.get("model_state_dict", checkpoint)

print("Checkpoint exists:", checkpoint_path.exists())
print("Number of saved tensors:", len(state_dict))
print(
    "Classification head saved:",
    "classifier.weight" in state_dict
    and "classifier.bias" in state_dict
)

print("Loading model architecture...")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=21
)

model.load_state_dict(state_dict, strict=True)

print("SUCCESS: All checkpoint weights loaded correctly.")
