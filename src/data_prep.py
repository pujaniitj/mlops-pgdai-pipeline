"""
Data preparation script for the Emotion dataset (dair-ai/emotion).

What this script does:
1. Downloads the dataset from Hugging Face (~6 MB, cached after first run)
2. Inspects sizes, classes, distribution, and sample text
3. Saves the label mapping to data/id2label.json
   (this is the ONLY data file we commit to git per the brief)

Run with:
    python src/data_prep.py
"""

from datasets import load_dataset
from collections import Counter
import json
import os


def main():
    print("=" * 70)
    print("MLOps Pipeline - Task 2: Data Preparation")
    print("=" * 70)

    # ---- Step 1: Load the dataset from Hugging Face ----
    print("\n[1/4] Loading dair-ai/emotion dataset from Hugging Face...")
    ds = load_dataset("dair-ai/emotion")
    print(f"   Loaded. Splits available: {list(ds.keys())}")

    # ---- Step 2: Inspect dataset structure and sizes ----
    print("\n[2/4] Dataset inspection:")
    print(f"   Train      : {len(ds['train']):>6} samples")
    print(f"   Validation : {len(ds['validation']):>6} samples")
    print(f"   Test       : {len(ds['test']):>6} samples")

    label_names = ds['train'].features['label'].names
    print(f"   Number of classes: {len(label_names)}")
    print(f"   Class names      : {label_names}")

    # ---- Step 3: Class distribution (for the report) ----
    print("\n[3/4] Class distribution (train split):")
    train_labels = ds['train']['label']
    counts = Counter(train_labels)
    total = len(train_labels)
    for label_id in sorted(counts.keys()):
        name = label_names[label_id]
        count = counts[label_id]
        percent = 100 * count / total
        bar = "#" * int(percent / 2)
        print(f"   {name:>10s} | {count:>5d} ({percent:5.2f}%) {bar}")

    # ---- Step 4: Build id2label mapping and save to JSON ----
    print("\n[4/4] Building and saving label mapping...")
    id2label = {i: name for i, name in enumerate(label_names)}

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "id2label.json")
    with open(output_path, "w") as f:
        json.dump(id2label, f, indent=2)
    print(f"   Saved id2label mapping to: {output_path}")
    print(f"   Mapping: {id2label}")

    # ---- Bonus: show a few sample texts for sanity check ----
    print("\n[BONUS] First 3 training samples:")
    for i in range(3):
        sample = ds['train'][i]
        print(f"   [{label_names[sample['label']]:>8s}] {sample['text']}")

    print("\n" + "=" * 70)
    print("Data preparation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()