import os
import sys
import json
import inspect
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


class DiagnosticTrainer(Trainer):
    """
    Subclass of Trainer that strictly validates finite loss at every training step.
    Fails immediately upon non-finite loss to prevent silent weight corruption.
    """
    def training_step(self, model, inputs, num_items_in_batch=None):
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"[CRITICAL ERROR] Non-finite loss ({loss.item()}) detected during training step! "
                "Halting immediately to prevent weight divergence."
            )
        return loss


def compute_metrics(eval_pred):
    """Calculates evaluation metrics: Accuracy, Precision, Recall, Macro F1, and AUC."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    # Explicit finite-value assertion
    if not np.isfinite(logits).all():
        raise FloatingPointError(
            "[CRITICAL ERROR] Non-finite values (NaN/Inf) encountered in model evaluation logits! "
            "Model weights have diverged."
        )

    preds = np.argmax(logits, axis=1)

    # Class prediction distribution diagnostics
    pred_0 = int(np.sum(preds == 0))
    pred_1 = int(np.sum(preds == 1))
    true_0 = int(np.sum(labels == 0))
    true_1 = int(np.sum(labels == 1))

    print(f"\n  [Diagnostics] Evaluation Predictions : Anti-Stereotype (0): {pred_0}, Stereotype (1): {pred_1}")
    print(f"  [Diagnostics] Ground Truth Reference: Anti-Stereotype (0): {true_0}, Stereotype (1): {true_1}")

    # Numerically stable softmax computation
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    pos_probs = probs[:, 1]  # Class 1 = "stereotype"

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    bin_precision, bin_recall, bin_f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)

    # Calculate ROC-AUC using positive-class probability
    try:
        if len(np.unique(labels)) > 1:
            auc = float(roc_auc_score(labels, pos_probs))
        else:
            auc = float('nan')
    except Exception as e:
        print(f"  [Diagnostics] Could not calculate ROC-AUC: {e}")
        auc = float('nan')

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
    batch_size: int = 8,
    grad_accum: int = 2,
    learning_rate: float = 5e-6,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    adam_epsilon: float = 1e-6,
    warmup_ratio: float = 0.1,
    max_length: int = 256,
    use_fold: int = 0,
    patience: int = 5,
    fp16: bool = False
):
    print("=" * 70)
    print("DEBERTA-V3 CONSERVATIVE FINE-TUNING PIPELINE (NUMERICALLY STABILIZED)")
    print("=" * 70)
    print(f"Base Model              : {model_name}")
    print(f"Max Sequence Length     : {max_length}")
    print(f"Per-Device Batch Size   : {batch_size}")
    print(f"Gradient Accumulation   : {grad_accum} (Effective Batch Size: {batch_size * grad_accum})")
    print(f"Learning Rate           : {learning_rate}")
    print(f"Weight Decay            : {weight_decay}")
    print(f"Max Gradient Norm       : {max_grad_norm}")
    print(f"Adam Epsilon            : {adam_epsilon}")
    print(f"Warmup Ratio            : {warmup_ratio}")
    print(f"Epochs                  : {epochs}")
    print(f"Early Stopping Patience : {patience if patience > 0 else 'Disabled'}")
    print(f"Precision Mode          : {'FP16' if fp16 else 'FP32 (Recommended for DeBERTa-v3)'}")
    print(f"Validation Strategy     : {'70/15/15 Group Split' if use_fold < 0 else f'GroupKFold (Fold {use_fold})'}")
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

    # 4. Training Arguments - Safe filtering for transformers version compatibility
    effective_batch_size = batch_size * grad_accum
    total_training_steps = (len(train_texts) // effective_batch_size + 1) * epochs
    warmup_steps = max(1, int(total_training_steps * warmup_ratio))

    import inspect
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    raw_args = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "adam_epsilon": adam_epsilon,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": 10,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": fp16
    }

    if "eval_strategy" in valid_params:
        raw_args["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in valid_params:
        raw_args["evaluation_strategy"] = "epoch"

    if "logging_dir" in valid_params:
        raw_args["logging_dir"] = f"{output_dir}/logs"

    safe_args = {k: v for k, v in raw_args.items() if k in valid_params}
    training_args = TrainingArguments(**safe_args)

    callbacks = []
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = DiagnosticTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks
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
    trainer.save_model(model_save_path)
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--use_fold", type=int, default=0, help="-1 for 70/15/15 split, 0..4 for GroupKFold")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (set >= epochs to allow all epochs)")
    parser.add_argument("--fp16", action="store_true", default=False, help="Enable FP16 (default False, FP32 recommended for DeBERTa-v3 stability)")

    args = parser.parse_args()
    train(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        adam_epsilon=args.adam_epsilon,
        warmup_ratio=args.warmup_ratio,
        use_fold=args.use_fold,
        patience=args.patience,
        fp16=args.fp16
    )
