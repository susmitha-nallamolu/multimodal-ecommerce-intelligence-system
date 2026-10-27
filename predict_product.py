
from pathlib import Path
import json
import re

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.models.multimodal_model import MultimodalModel


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "multimodal_model_fasttrack.pth"

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed"
    / "vocabulary.json"
)

LABEL_MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed"
    / "label_mapping.csv"
)

IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 100
PAD_ID = 0
UNK_ID = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. LOAD VOCABULARY
# ============================================================

with open(VOCAB_PATH, "r", encoding="utf-8") as file:
    vocabulary = json.load(file)

print(f"Vocabulary loaded: {len(vocabulary)} tokens")


# ============================================================
# 3. LOAD LABEL MAPPING
# ============================================================

label_df = pd.read_csv(LABEL_MAPPING_PATH)

# Your CSV contains the columns: category and label.
label_to_category = {
    int(row["label"]): str(row["category"])
    for _, row in label_df.iterrows()
}

num_classes = len(label_to_category)

print(f"Categories loaded: {num_classes}")


# ============================================================
# 4. IMAGE TRANSFORMATION
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================
model = MultimodalModel(
    num_classes=num_classes
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

# Support either a raw state dictionary or a checkpoint dictionary.
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

print(f"Model loaded on: {DEVICE}")


# ============================================================
# 6. PREPROCESS TEXT
# ============================================================

def preprocess_text(title, description):
    """
    Combine the product title and description,
    convert words into vocabulary IDs, and pad/truncate
    the sequence to MAX_TEXT_LENGTH.
    """

    combined_text = f"{title} {description}".lower()

    tokens = re.findall(r"\b\w+\b", combined_text)

    token_ids = [
        vocabulary.get(token, UNK_ID)
        for token in tokens
    ]

    # Truncate if the sequence is too long.
    token_ids = token_ids[:MAX_TEXT_LENGTH]

    # Pad if the sequence is too short.
    if len(token_ids) < MAX_TEXT_LENGTH:
        token_ids.extend(
            [PAD_ID] * (MAX_TEXT_LENGTH - len(token_ids))
        )

    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


# ============================================================
# 7. LOAD AND PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):
    """
    Load an image, convert it to RGB, resize it,
    and convert it into a PyTorch tensor.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = image_transform(image)

    # Add the batch dimension.
    return image_tensor.unsqueeze(0)


# ============================================================
# 8. PREDICT PRODUCT CATEGORY
# ============================================================

def predict_product(image_path, title, description):
    """
    Predict the product category using both image and text.
    Returns the predicted category and the top 3 predictions.
    """

    image_tensor = preprocess_image(image_path).to(DEVICE)

    text_tensor = preprocess_text(
        title,
        description
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor, text_tensor)

        # Convert model outputs into normalized scores.
        probabilities = torch.softmax(outputs, dim=1)[0]

        # Get the three highest-scoring classes.
        top_count = min(3, probabilities.numel())

        top_scores, top_indices = torch.topk(
            probabilities,
            k=top_count
        )

    predicted_label = int(top_indices[0].item())
    confidence = float(top_scores[0].item()) * 100

    top_predictions = []

    for score, index in zip(top_scores, top_indices):
        label_id = int(index.item())

        top_predictions.append({
            "label_id": label_id,
            "category": label_to_category.get(
                label_id,
                "Unknown category"
            ),
            "confidence": float(score.item()) * 100
        })

    return {
        "label_id": predicted_label,
        "category": label_to_category.get(
            predicted_label,
            "Unknown category"
        ),
        "confidence": confidence,
        "top_predictions": top_predictions
    }


# ============================================================
# 9. RUN THE PREDICTION PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print(" MULTIMODAL E-COMMERCE INTELLIGENCE SYSTEM ")
    print("=" * 55)

    image_path = input(
        "\nEnter the full path to a product image: "
    ).strip().strip('"')

    title = input(
        "Enter product title: "
    ).strip()

    description = input(
        "Enter product description: "
    ).strip()

    try:
        result = predict_product(
            image_path=image_path,
            title=title,
            description=description
        )

        print("\n" + "=" * 55)
        print(" PRODUCT PREDICTION ")
        print("=" * 55)

        print(f"Predicted category: {result['category']}")
        print(f"Predicted label ID: {result['label_id']}")
        print(
            f"Model confidence score: "
            f"{result['confidence']:.2f}%"
        )

        print("\nTOP 3 PREDICTED CATEGORIES")
        print("-" * 55)

        for rank, prediction in enumerate(
            result["top_predictions"],
            start=1
        ):
            print(
                f"{rank}. {prediction['category']} "
                f"(Label {prediction['label_id']}) "
                f"- {prediction['confidence']:.2f}%"
            )

        print("\nPrediction completed.")

    except FileNotFoundError as error:
        print(f"\nFile error: {error}")

    except Exception as error:
        print(f"\nPrediction failed: {error}")
        raise
