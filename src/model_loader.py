"""
Model loader for the Emotion classification task.

Loads:
  - The DistilBERT tokenizer
  - The DistilBERT model configured for 6-class classification
    (matching our id2label mapping from data_prep.py)

This module is imported by the Kaggle training notebook (Task 4)
and by src/inference.py (Task 7).
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os


MODEL_NAME = "distilbert-base-uncased"


def load_id2label(mapping_path="data/id2label.json"):
    """Load the id->label mapping created by data_prep.py."""
    with open(mapping_path, "r") as f:
        raw = json.load(f)
    id2label = {int(k): v for k, v in raw.items()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def load_model_and_tokenizer(model_name=MODEL_NAME):
    """Load tokenizer and a sequence-classification model with the right head."""
    id2label, label2id = load_id2label()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    return tokenizer, model


def main():
    print("=" * 70)
    print("MLOps Pipeline - Task 3: Load Hugging Face Model")
    print("=" * 70)

    print(f"\nModel: {MODEL_NAME}")
    print("Loading tokenizer and model...")
    tokenizer, model = load_model_and_tokenizer()

    print("\nTokenizer loaded:")
    print(f"   Vocab size      : {tokenizer.vocab_size}")
    print(f"   Max length      : {tokenizer.model_max_length}")

    print("\nModel loaded:")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Num labels      : {model.config.num_labels}")
    print(f"   id2label        : {model.config.id2label}")

    # Quick smoke test - tokenize and forward-pass a sample
    print("\nSmoke test on sample text:")
    sample_text = "I am so happy today!"
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    print(f"   Input text      : {sample_text!r}")
    print(f"   Output shape    : {tuple(outputs.logits.shape)}")
    print(f"   (Predictions are random because the head is untrained "
          f"- training happens on Kaggle)")

    print("\n" + "=" * 70)
    print("Model loader works correctly. Ready for Kaggle training (Task 4).")
    print("=" * 70)


if __name__ == "__main__":
    main()