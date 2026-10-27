
import pandas as pd

df = pd.read_csv("data/raw/processed/train_processed.csv")

mapping = (
    df[["categories", "label"]]
    .drop_duplicates()
    .sort_values("label")
)

print(mapping.to_string(index=False))
