
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "transformer_distilbert"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "processed" / "val_processed.csv"

DEVICE = torch.device("cpu")
MAX_LENGTH = 128

print("=" * 60)
print("TESTING SAVED DISTILBERT PRODUCT CLASSIFIER")
print("=" * 60)

# 1. Check saved files
checkpoint_path = MODEL_DIR / "best_model.pth"

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

# 2. Load checkpoint
checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True,
)

num_classes = checkpoint["num_classes"]
model_name = checkpoint["model_name"]
max_length = checkpoint.get("max_length", MAX_LENGTH)

# 3. Load tokenizer and model architecture
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Checkpoint loaded successfully!")
print("Number of classes:", num_classes)

# 4. Get a real product description from validation data
df = pd.read_csv(DATA_PATH)
sample = df.iloc[0]

product_text = str(sample["text"])
actual_label = int(sample["label"])
actual_category = str(sample["categories"])

# 5. Tokenize product text
inputs = tokenizer(
    product_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=max_length,
)
inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

# 6. Predict
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)[0]

top_scores, top_indices = torch.topk(probabilities, k=3)

# 7. Display results
label_mapping = checkpoint.get("label_mapping", {})

print("\nProduct text:")
print(product_text[:500])

print("\nActual category:", actual_category)
print("Actual label ID:", actual_label)

print("\nTop 3 predictions:")

for rank, (score, class_id) in enumerate(
    zip(top_scores.tolist(), top_indices.tolist()),
    start=1,
):
    category = label_mapping.get(str(class_id), f"Class {class_id}")
    print(
        f"{rank}. {category} "
        f"(class {class_id}) - {score:.2%}"
    )

print("\nInference test completed successfully!")
