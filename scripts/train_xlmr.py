"""Fine-tune XLM-RoBERTa for suicide/non-suicide text classification.

Examples:
    python scripts/train_xlmr.py --epochs 3
    python scripts/train_xlmr.py --explain-text "your text here"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import ClassLabel, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "xlm-roberta-base"
LABELS = ["non-suicide", "suicide"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
DATA_FILES = (
    "nonsuicide_english.csv",
    "nonsuicide_urdu.csv",
    "nonsuicide_roman_urdu.csv",
    "suicide_english.csv",
    "suicide_urdu.csv",
    "suicide_roman_urdu.csv",
)


def load_dataset(data_dir: Path, rows_per_file: int = 20) -> Dataset:
    """Load the first rows_per_file rows from each language-specific export."""
    if rows_per_file < 1:
        raise ValueError("rows_per_file must be at least 1")

    examples: dict[tuple[str, str], dict[str, Any]] = {}
    for filename in DATA_FILES:
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file was not found: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or "text" not in reader.fieldnames:
                raise ValueError(f"{path} must contain a 'text' column")

            inferred_label = "suicide" if filename.startswith("suicide_") else "non-suicide"
            for row_number, row in enumerate(reader):
                if row_number >= rows_per_file:
                    break
                text = (row.get("text") or "").strip()
                label = (row.get("label") or inferred_label).strip().lower()
                if text and label in LABEL_TO_ID:
                    examples[(text, label)] = {
                        "text": text,
                        "label": LABEL_TO_ID[label],
                    }

    if not examples:
        raise ValueError(f"No usable labeled examples found in {data_dir}")
    dataset = Dataset.from_list(list(examples.values()))
    return dataset.cast_column("label", ClassLabel(names=LABELS))


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    labels = eval_prediction.label_ids
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision_suicide": precision_score(
            labels, predictions, pos_label=1, zero_division=0
        ),
        "recall_suicide": recall_score(labels, predictions, pos_label=1, zero_division=0),
        "f1_suicide": f1_score(labels, predictions, pos_label=1, zero_division=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data_set"))
    parser.add_argument("--output-dir", type=Path, default=Path("fyp-xlmr-model"))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=20,
        help="Read only the first N rows from each of the six CSV files.",
    )
    parser.add_argument("--epochs", type=float, default=3, choices=(3, 4, 5))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--explain-text",
        help="Skip training and explain one text with an existing saved model.",
    )
    parser.add_argument(
        "--explanation-output",
        type=Path,
        default=Path("shap-explanation.html"),
    )
    return parser.parse_args()


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    common = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "recall_suicide",
        "greater_is_better": True,
        "seed": args.seed,
        "report_to": "none",
    }
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_dataset(args.data_dir, args.rows_per_file)
    split = dataset.train_test_split(
        test_size=0.2,
        seed=args.seed,
        stratify_by_column="label",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = split.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    )
    trainer = Trainer(
        model=model,
        args=build_training_arguments(args),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps({"examples": len(dataset), **metrics}, indent=2, default=float))


def explain(args: argparse.Namespace) -> None:
    import shap

    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def predict(texts: list[str]) -> np.ndarray:
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            return torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()

    explainer = shap.Explainer(predict, shap.maskers.Text(tokenizer))
    values = explainer([args.explain_text])
    args.explanation_output.parent.mkdir(parents=True, exist_ok=True)
    plot = shap.plots.text(values[0], display=False)
    shap.save_html(str(args.explanation_output), plot)
    print(f"Saved SHAP explanation to {args.explanation_output}")


def main() -> None:
    args = parse_args()
    if not 3 <= args.epochs <= 5:
        raise ValueError("--epochs must be between 3 and 5")
    if args.explain_text:
        explain(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
