
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_preprocessing.product_dataset import ProductDataset
from models.multimodal_model import MultimodalModel


# 1. Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed"
    / "train_processed.csv"
)


# 2. Load the training dataset
dataset = ProductDataset(TRAIN_CSV)

# Use a small batch of just 4 products
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

batch = next(iter(loader))

images = batch["image"]
texts = batch["text"]
labels = batch["label"]


# 3. Create the model and training tools
model = MultimodalModel(num_classes=21)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Target labels:", labels.tolist())
print("\nStarting repeated-batch learning test...")


# Track the best model parameters
best_loss = float("inf")
best_state = None



model.train()

for step in range(1, 101):

    # 1. Clear old gradients
    optimizer.zero_grad()

    # 2. Forward pass
    outputs = model(images, texts)

    # 3. Calculate training loss
    loss = criterion(outputs, labels)

    # 4. Calculate gradients
    loss.backward()

    # 5. Update model parameters
    optimizer.step()

    # 6. Evaluate AFTER the update
    with torch.no_grad():
        updated_outputs = model(images, texts)
        updated_loss = criterion(updated_outputs, labels)

        predictions = updated_outputs.argmax(dim=1)
        correct = (predictions == labels).sum().item()

        current_loss = updated_loss.item()

    # 7. Save a copy when loss improves
    if current_loss < best_loss:
        best_loss = current_loss

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    # 8. Print progress
    if step == 1 or step % 10 == 0:
        print(
            f"Step {step:03d} | "
            f"Loss: {current_loss:.4f} | "
            f"Correct: {correct}/4 | "
            f"Predictions: {predictions.tolist()}"
        )


# 9. Restore the best model from this test
if best_state is not None:
    model.load_state_dict(best_state)

    print(f"\nBest loss restored: {best_loss:.4f}")

    with torch.no_grad():
        final_outputs = model(images, texts)
        final_predictions = final_outputs.argmax(dim=1)

        correct = (
            final_predictions == labels
        ).sum().item()

    print("Correct predictions:", f"{correct}/4")
    print("Final predictions:", final_predictions.tolist())

print("\nRepeated-batch test completed.")
