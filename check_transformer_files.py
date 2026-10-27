from pathlib import Path

project_root = Path(r"C:\Users\susmi\multimodal_ecommerce_intelligence")

folders = [
    project_root / "models" / "transformer_distilbert",
    project_root / "models" / "transformer_distilbert_3epoch",
]

for folder in folders:
    print(f"\nFolder: {folder}")
    print("Exists:", folder.exists())

    if folder.exists():
        for file in folder.iterdir():
            print(" -", file.name)