
import os
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.data_preprocessing.product_dataset import ProductDataset
from src.models.ann_model import ANNProductClassifier
from src.models.cnn_classifier import CNNImageClassifier
from src.models.lstm_classifier import LSTMTextClassifier
from src.models.multimodal_model import MultimodalModel


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\susmi\multimodal_ecommerce_intelligence"
)

VAL_CSV = (
    PROJECT_ROOT / "data" / "raw" / "processed"
    / "val_processed.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

CNN_MODEL_PATH = MODEL_DIR / "cnn_image_only.pth"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_text_only.pth"
MULTIMODAL_MODEL_PATH = MODEL_DIR / "multimodal_model_fasttrack.pth"

ANN_MODEL_PATH = MODEL_DIR / "ann_product_classifier.pth"
ANN_SCALER_PATH = MODEL_DIR / "ann_feature_scaler.pkl"

TRANSFORMER_DIR = MODEL_DIR / "transformer_distilbert_3epoch"
TRANSFORMER_MODEL_PATH = TRANSFORMER_DIR / "best_model.pth"
BATCH_SIZE = 8
ANN_BATCH_SIZE = 128
NUM_CLASSES = 21
SAMPLES_PER_CLASS = 400

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

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
    "Toys & Games"
]

# These names and their order must match train_ann.py.
FEATURE_NAMES = [
    "title_character_count",
    "title_word_count",
    "description_character_count",
    "description_word_count",
    "combined_character_count",
    "combined_word_count",
    "unique_word_ratio",
    "average_word_length",
    "digit_count",
    "punctuation_count"
]


# ============================================================
# 2. RECREATE THE ANN FEATURES
# ============================================================

def extract_features(title, description):
    """Reproduce the numerical feature extraction used during ANN training."""

    title = str(title) if pd.notna(title) else ""
    description = str(description) if pd.notna(description) else ""

    combined_text = f"{title} {description}".strip()

    title_words = re.findall(r"\b\w+\b", title.lower())
    description_words = re.findall(r"\b\w+\b", description.lower())
    combined_words = re.findall(r"\b\w+\b", combined_text.lower())

    unique_ratio = (
        len(set(combined_words)) / len(combined_words)
        if combined_words else 0.0
    )

    average_word_length = (
        np.mean([len(word) for word in combined_words])
        if combined_words else 0.0
    )

    digit_count = sum(character.isdigit() for character in combined_text)

    punctuation_count = sum(
        not character.isalnum() and not character.isspace()
        for character in combined_text
    )

    return [
        len(title),
        len(title_words),
        len(description),
        len(description_words),
        len(combined_text),
        len(combined_words),
        unique_ratio,
        average_word_length,
        digit_count,
        punctuation_count
    ]


def build_feature_matrix(dataframe):
    features = [
        extract_features(row.title, row.description)
        for row in dataframe.itertuples(index=False)
    ]
    return np.asarray(features, dtype=np.float32)


# ============================================================
# 3. SELECT THE SAME VALIDATION PRODUCTS FOR EVERY MODEL
# ============================================================

def select_validation_indices(dataset):
    labels = dataset.data["label"].to_numpy()
    rng = np.random.default_rng(43)

    selected_indices = []

    for label in sorted(np.unique(labels)):
        class_indices = np.where(labels == label)[0]

        chosen = rng.choice(
            class_indices,
            size=min(SAMPLES_PER_CLASS, len(class_indices)),
            replace=False
        )

        selected_indices.extend(chosen.tolist())

    return sorted(selected_indices)


# ============================================================
# 4. LOAD CHECKPOINTS
# ============================================================

def read_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]

    return checkpoint


def load_checkpoint(model, path):
    checkpoint = read_checkpoint(path)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# 5. CALCULATE COMMON EVALUATION METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    accuracy = accuracy_score(actual, predicted)

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=list(range(NUM_CLASSES)),
            average="weighted",
            zero_division=0
        )
    )

    return {
        "actual": actual,
        "predicted": predicted,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }


# ============================================================
# 6. EVALUATE CNN, LSTM, AND MULTIMODAL MODELS
# ============================================================

def evaluate_image_text_model(model, data_loader, model_type):
    actual = []
    predicted = []

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            labels = batch["label"].to(DEVICE)

            if model_type == "cnn":
                outputs = model(batch["image"].to(DEVICE))

            elif model_type == "lstm":
                outputs = model(batch["text"].to(DEVICE))

            elif model_type == "multimodal":
                outputs = model(
                    batch["image"].to(DEVICE),
                    batch["text"].to(DEVICE)
                )

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            predictions = outputs.argmax(dim=1)

            actual.extend(labels.cpu().tolist())
            predicted.extend(predictions.cpu().tolist())

    return calculate_metrics(actual, predicted)


# ============================================================
# 7. EVALUATE ANN
# ============================================================

def evaluate_ann(val_df):
    print("\nLoading ANN model and scaler...")

    checkpoint = read_checkpoint(ANN_MODEL_PATH)

    if not isinstance(checkpoint, dict):
        raise ValueError("Unexpected ANN checkpoint format.")

    checkpoint_features = checkpoint.get("feature_names")

    if checkpoint_features != FEATURE_NAMES:
        raise ValueError(
            "ANN feature names/order do not match the saved checkpoint. "
            f"Checkpoint features: {checkpoint_features}"
        )

    with open(ANN_SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    model = ANNProductClassifier(
        input_dim=checkpoint.get("input_dim", len(FEATURE_NAMES)),
        num_classes=checkpoint.get("num_classes", NUM_CLASSES)
    )

    model.load_state_dict(extract_state_dict(checkpoint))
    model.to(DEVICE)
    model.eval()

    raw_features = build_feature_matrix(val_df)
    scaled_features = scaler.transform(raw_features).astype(np.float32)

    dataset = torch.from_numpy(scaled_features)
    loader = DataLoader(
        dataset,
        batch_size=ANN_BATCH_SIZE,
        shuffle=False
    )

    predicted = []

    with torch.no_grad():
        for features in loader:
            outputs = model(features.to(DEVICE))
            predicted.extend(outputs.argmax(dim=1).cpu().tolist())

    actual = val_df["label"].to_numpy(dtype=np.int64).tolist()

    return calculate_metrics(actual, predicted)


# ============================================================
# 8. EVALUATE DISTILBERT
# ============================================================

def evaluate_transformer(val_df):
    print("\nLoading fine-tuned DistilBERT and tokenizer...")

    if not TRANSFORMER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Transformer checkpoint not found: {TRANSFORMER_MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(TRANSFORMER_DIR),
        local_files_only=True
    )

    checkpoint = read_checkpoint(TRANSFORMER_MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=NUM_CLASSES
    )

    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    actual = []
    predicted = []

    texts = (
        val_df["title"].fillna("").astype(str)
        + " "
        + val_df["description"].fillna("").astype(str)
    ).tolist()

    labels = val_df["label"].to_numpy(dtype=np.int64)

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]
        batch_labels = labels[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = model(**encoded)
            batch_predictions = outputs.logits.argmax(dim=1)

        actual.extend(batch_labels.tolist())
        predicted.extend(batch_predictions.cpu().tolist())

        completed = min(start + BATCH_SIZE, len(texts))

        if completed % 400 == 0 or completed == len(texts):
            print(
                f"DistilBERT evaluation: {completed}/{len(texts)} products"
            )

    return calculate_metrics(actual, predicted)


# ============================================================
# 9. SAVE CLASSIFICATION REPORTS AND CONFUSION MATRICES
# ============================================================

def save_results(results, model_name):
    report = classification_report(
        results["actual"],
        results["predicted"],
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    report_path = REPORT_DIR / f"{model_name}_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    matrix = confusion_matrix(
        results["actual"],
        results["predicted"],
        labels=list(range(NUM_CLASSES))
    )

    matrix_path = REPORT_DIR / f"{model_name}_confusion_matrix.csv"

    pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    ).to_csv(matrix_path)

    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {matrix_path}")


# ============================================================
# 10. SAVE COMPARISON CHART
# ============================================================

def save_comparison_chart(comparison_df):
    chart_path = REPORT_DIR / "model_comparison.png"

    plot_df = comparison_df.set_index("Model")[
        ["Accuracy", "Macro F1", "Weighted F1"]
    ]

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 6),
        ylim=(0, 1)
    )

    ax.set_title("Five-Model Product Classification Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved comparison chart:", chart_path)


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main():
    print("=" * 70)
    print("FIVE-MODEL PRODUCT CLASSIFICATION COMPARISON")
    print("=" * 70)

    if not VAL_CSV.exists():
        raise FileNotFoundError(f"Validation CSV not found: {VAL_CSV}")

    print("\nLoading validation dataset...")
    dataset = ProductDataset(VAL_CSV)

    indices = select_validation_indices(dataset)
    validation_subset = Subset(dataset, indices)

    data_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # The raw CSV rows must correspond to the exact same selected indices.
    val_df = pd.read_csv(VAL_CSV).iloc[indices].reset_index(drop=True)

    print("Validation products:", len(validation_subset))
    print("Number of categories:", val_df["label"].nunique())
    print("Device:", DEVICE)

    results = {}

    print("\n[1/5] Evaluating ANN...")
    results["ANN"] = evaluate_ann(val_df)
    save_results(results["ANN"], "ann")

    print("\n[2/5] Evaluating CNN image-only model...")
    cnn_model = load_checkpoint(
        CNNImageClassifier(num_classes=NUM_CLASSES),
        CNN_MODEL_PATH
    )
    results["CNN Image Only"] = evaluate_image_text_model(
        cnn_model, data_loader, "cnn"
    )
    save_results(results["CNN Image Only"], "cnn")

    print("\n[3/5] Evaluating LSTM text-only model...")
    lstm_model = load_checkpoint(
        LSTMTextClassifier(num_classes=NUM_CLASSES),
        LSTM_MODEL_PATH
    )
    results["LSTM Text Only"] = evaluate_image_text_model(
        lstm_model, data_loader, "lstm"
    )
    save_results(results["LSTM Text Only"], "lstm_text_only")

    print("\n[4/5] Evaluating CNN + LSTM multimodal model...")
    multimodal_model = load_checkpoint(
        MultimodalModel(num_classes=NUM_CLASSES),
        MULTIMODAL_MODEL_PATH
    )
    results["CNN + LSTM Multimodal"] = evaluate_image_text_model(
        multimodal_model, data_loader, "multimodal"
    )
    save_results(results["CNN + LSTM Multimodal"], "multimodal")

    print("\n[5/5] Evaluating DistilBERT Transformer...")
    results["DistilBERT Transformer"] = evaluate_transformer(val_df)
    save_results(results["DistilBERT Transformer"], "distilbert")

    metric_names = [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
        ("weighted_precision", "Weighted Precision"),
        ("weighted_recall", "Weighted Recall"),
        ("weighted_f1", "Weighted F1")
    ]

    rows = []

    for model_name, model_results in results.items():
        row = {"Model": model_name}

        for key, label in metric_names:
            row[label] = round(model_results[key], 4)

        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    comparison_path = REPORT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 90)
    print("FINAL MODEL COMPARISON")
    print("=" * 90)
    print(comparison_df.to_string(index=False))

    print("\nSaved comparison table:", comparison_path)

    save_comparison_chart(comparison_df)

    print("\nALL FIVE MODELS EVALUATED SUCCESSFULLY.")
    print("Reports folder:", REPORT_DIR)


if __name__ == "__main__":
    main()
