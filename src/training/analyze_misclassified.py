
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data_preprocessing.product_dataset import ProductDataset
from src.models.multimodal_model import MultimodalModel


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\susmi\multimodal_ecommerce_intelligence"
)

VAL_CSV = PROJECT_ROOT / "data" / "raw" / "processed" / "val_processed.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "multimodal_model.pth"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "misclassified_products.csv"

NUM_CLASSES = 21
SAMPLES_PER_CLASS = 10
BATCH_SIZE = 8
DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "All Beauty",
    "All Electronics",
    "Appliances",
    "Arts Crafts & Sewing",
    "Automotive",
    "Baby",
    "Baby Products",
    "Beauty",
    "Cell Phones & Accessories",
    "Clothing Shoes & Jewelry",
    "Electronics",
    "Grocery & Gourmet Food",
    "Health & Personal Care",
    "Industrial & Scientific",
    "Musical Instruments",
    "Office Products",
    "Patio Lawn & Garden",
    "Pet Supplies",
    "Sports & Outdoors",
    "Tools & Home Improvement",
    "Toys & Games",
]


# --------------------------------------------------
# 2. SELECT THE SAME 210 VALIDATION PRODUCTS
# --------------------------------------------------

def select_validation_indices(dataset):
    labels = dataset.data["label"].to_numpy()
    selected_indices = []
    rng = __import__("numpy").random.default_rng(43)

    for label in sorted(set(labels)):
        class_indices = __import__("numpy").where(labels == label)[0]

        chosen = rng.choice(
            class_indices,
            size=min(SAMPLES_PER_CLASS, len(class_indices)),
            replace=False,
        )

        selected_indices.extend(chosen.tolist())

    return sorted(selected_indices)


# --------------------------------------------------
# 3. LOAD THE TRAINED MODEL
# --------------------------------------------------

def load_model():
    model = MultimodalModel(num_classes=NUM_CLASSES)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# 4. PREDICT PRODUCTS
# --------------------------------------------------

def main():
    print("Loading validation dataset...")

    dataset = ProductDataset(str(VAL_CSV))
    indices = select_validation_indices(dataset)

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = load_model()

    misclassified = []
    correct_count = 0
    total_count = 0

    print("Analyzing predictions...")

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(DEVICE)
            texts = batch["text"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(images, texts)
            predictions = outputs.argmax(dim=1)

            for i in range(len(labels)):
                actual = int(labels[i].item())
                predicted = int(predictions[i].item())

                total_count += 1

                if actual == predicted:
                    correct_count += 1
                    continue

                # Recover the corresponding original CSV row.
                subset_position = batch_index * BATCH_SIZE + i
                original_position = indices[subset_position]
                product = dataset.data.iloc[original_position]

                misclassified.append({
                    "ImgId": product.get("ImgId", ""),
                    "title": product.get("title", ""),
                    "description": product.get("description", ""),
                    "actual_label": actual,
                    "actual_category": CLASS_NAMES[actual],
                    "predicted_label": predicted,
                    "predicted_category": CLASS_NAMES[predicted],
                })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "ImgId",
        "title",
        "description",
        "actual_label",
        "actual_category",
        "predicted_label",
        "predicted_category",
    ]

    results_df = pd.DataFrame(misclassified, columns=columns)
    results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 55)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 55)
    print(f"Total products evaluated: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions: {len(misclassified)}")
    print(f"Accuracy: {correct_count / total_count:.2%}")
    print(f"Report saved to: {OUTPUT_PATH}")

    print("\nFirst 10 misclassified products:")

    for item in misclassified[:10]:
        print("\nTitle:", item["title"])
        print("Actual:", item["actual_category"])
        print("Predicted:", item["predicted_category"])
        print("Description:", str(item["description"])[:200])


if __name__ == "__main__":
    main()
