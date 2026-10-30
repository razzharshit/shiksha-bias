import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class StereotypeDataset(Dataset):
    """PyTorch Dataset for Stereotype Binary Classification."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    """Calculates evaluation metrics: Accuracy, Precision, Recall, Macro F1, and AUC."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    # Calculate softmax probabilities for AUC
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    pos_probs = probs[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    bin_precision, bin_recall, bin_f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    try:
        auc = roc_auc_score(labels, pos_probs)
    except Exception:
        auc = 0.5

    return {
        'accuracy': float(acc),
        'macro_f1': float(f1),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'stereotype_f1': float(bin_f1),
        'stereotype_precision': float(bin_precision),
        'stereotype_recall': float(bin_recall),
        'roc_auc': float(auc)
    }


def load_benchmark_data(benchmark_path: str, use_fold: int = -1):
    """
    Loads benchmark records from JSON.
    If use_fold >= 0, returns (train_data, val_data, test_data) based on 5-fold GroupKFold.
    Otherwise, returns the primary 70/15/15 group split.
    """
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = data['records']
    
    train_texts, train_labels = [], []
    val_texts, val_labels = [], []
    test_texts, test_labels = [], []

    label_map = {"anti_stereotype": 0, "stereotype": 1}

    for r in records:
        text = r['sentence'].strip()
        label = label_map[r['classification']]

        if use_fold >= 0:
            # GroupKFold strategy: fold K is val, next is test, rest is train
            if r['group_kfold'] == use_fold:
                val_texts.append(text)
                val_labels.append(label)
            elif r['group_kfold'] == (use_fold + 1) % 5:
                test_texts.append(text)
                test_labels.append(label)
            else:
                train_texts.append(text)
                train_labels.append(label)
        else:
            # 70/15/15 Group Split strategy
            if r['split_group'] == 'train':
                train_texts.append(text)
                train_labels.append(label)
            elif r['split_group'] == 'val':
                val_texts.append(text)
                val_labels.append(label)
            else:
                test_texts.append(text)
                test_labels.append(label)

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def train(
    benchmark_path: str,
    output_dir: str = "./models/deberta-v3-stereotype",
    model_name: str = "microsoft/deberta-v3-base",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    use_fold: int = -1
):
    print("=" * 70)
    print("DEBERTA-V3 GENDER STEREOTYPE FINE-TUNING PIPELINE")
    print("=" * 70)
    print(f"Base Model    : {model_name}")
    print(f"Max Seq Length: {max_length}")
    print(f"Batch Size    : {batch_size}")
    print(f"Learning Rate : {learning_rate}")
    print(f"Epochs        : {epochs}")
    print(f"Strategy      : {'70/15/15 Group Split' if use_fold < 0 else f'GroupKFold (Fold {use_fold})'}")
    print("=" * 70)

    # 1. Load Data
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_benchmark_data(benchmark_path, use_fold)
    print(f"[+] Loaded Train : {len(train_texts)} sentences (Stereotypes: {sum(train_labels)}, Anti: {len(train_labels) - sum(train_labels)})")
    print(f"[+] Loaded Val   : {len(val_texts)} sentences (Stereotypes: {sum(val_labels)}, Anti: {len(val_labels) - sum(val_labels)})")
    print(f"[+] Loaded Test  : {len(test_texts)} sentences (Stereotypes: {sum(test_labels)}, Anti: {len(test_labels) - sum(test_labels)})")

    # 2. Tokenize with official DeBERTa-v3 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    train_dataset = StereotypeDataset(train_encodings, train_labels)
    val_dataset = StereotypeDataset(val_encodings, val_labels)
    test_dataset = StereotypeDataset(test_encodings, test_labels)

    # 3. Model setup with 2-class head
    id2label = {0: "anti_stereotype", 1: "stereotype"}
    label2id = {"anti_stereotype": 0, "stereotype": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    # 4. Training Arguments - Filter dynamically against valid parameters in installed transformers
    total_training_steps = (len(train_texts) // batch_size + 1) * epochs
    warmup_steps = max(1, int(total_training_steps * 0.10))

    import inspect
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    
    raw_args = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.01,
        "learning_rate": learning_rate,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": 10,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": torch.cuda.is_available()
    }

    if "eval_strategy" in valid_params:
        raw_args["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in valid_params:
        raw_args["evaluation_strategy"] = "epoch"

    if "logging_dir" in valid_params:
        raw_args["logging_dir"] = f"{output_dir}/logs"

    # Strict filtering: keep only arguments accepted by the installed version
    safe_args = {k: v for k, v in raw_args.items() if k in valid_params}
    training_args = TrainingArguments(**safe_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 5. Train
    print("\n[*] Starting DeBERTa-v3 fine-tuning...")
    trainer.train()

    # 6. Final Evaluation on Hold-Out Test Set
    print("\n[*] Evaluating Best Checkpoint on Unseen Hold-out Test Set...")
    test_results = trainer.evaluate(test_dataset)
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION RESULTS:")
    print("=" * 50)
    for k, v in test_results.items():
        if isinstance(v, float):
            print(f"  • {k:<25}: {v:.4f}")

    # 7. Save Model & Tokenizer
    model_save_path = f"{output_dir}/best_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\n[+] Fine-tuned model saved to: {model_save_path}")

    # Save metrics JSON
    with open(f"{model_save_path}/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    return test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa-v3 for Gender Stereotype Detection")
    parser.add_argument("--benchmark", type=str, default="data/benchmarks/indicasa_split_benchmark.json")
    parser.add_argument("--output_dir", type=str, default="./models/deberta-v3-stereotype")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--use_fold", type=int, default=-1, help="-1 for 70/15/15 split, 0..4 for GroupKFold")

    args = parser.parse_args()
    train(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_fold=args.use_fold
    )

