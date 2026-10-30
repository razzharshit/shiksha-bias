import os
import sys
import json
import inspect
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


class StereotypeDataset(Dataset):
    """PyTorch Dataset for Stereotype Binary Classification."""
    def __init__(self, encodings, labels, indices=None):
        self.encodings = encodings
        self.labels = labels
        self.indices = indices if indices is not None else list(range(len(labels)))

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['record_idx'] = torch.tensor(self.indices[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class DiagnosticTrainer(Trainer):
    """
    Subclass of Trainer that strictly validates finite loss at every training step.
    Fails immediately upon non-finite loss to prevent silent weight corruption.
    """
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Pop record_idx if present before model forward
        inputs_clean = {k: v for k, v in inputs.items() if k != 'record_idx'}
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs_clean, num_items_in_batch=num_items_in_batch)
        else:
            loss = super().training_step(model, inputs_clean)

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
            "[CRITICAL ERROR] Non-finite values (NaN/Inf) encountered in model evaluation logits!"
        )

    preds = np.argmax(logits, axis=1)

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    pos_probs = probs[:, 1]  # Class 1 = "stereotype"

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    bin_precision, bin_recall, bin_f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)

    try:
        if len(np.unique(labels)) > 1:
            auc = float(roc_auc_score(labels, pos_probs))
        else:
            auc = float('nan')
    except Exception:
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


def load_fold_data(records: List[Dict[str, Any]], test_fold_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Partitions records into train, validation, and test splits with zero context leakage:
    - Test: group_kfold == test_fold_idx
    - Val: group_kfold == (test_fold_idx + 1) % 5
    - Train: group_kfold in remaining 3 groups
    """
    label_map = {"anti_stereotype": 0, "stereotype": 1}

    train_recs, val_recs, test_recs = [], [], []
    val_fold_idx = (test_fold_idx + 1) % 5

    for r in records:
        f = r['group_kfold']
        if f == test_fold_idx:
            test_recs.append(r)
        elif f == val_fold_idx:
            val_recs.append(r)
        else:
            train_recs.append(r)

    # Strict assertion of context isolation
    train_ctxs = set(r['context_id'] for r in train_recs)
    val_ctxs = set(r['context_id'] for r in val_recs)
    test_ctxs = set(r['context_id'] for r in test_recs)

    assert not (train_ctxs & val_ctxs), f"Context leakage detected between Train and Val in Fold {test_fold_idx}!"
    assert not (train_ctxs & test_ctxs), f"Context leakage detected between Train and Test in Fold {test_fold_idx}!"
    assert not (val_ctxs & test_ctxs), f"Context leakage detected between Val and Test in Fold {test_fold_idx}!"

    def package(recs):
        return {
            'texts': [r['sentence'].strip() for r in recs],
            'labels': [label_map[r['classification']] for r in recs],
            'records': recs
        }

    return package(train_recs), package(val_recs), package(test_recs)


def evaluate_test_split(
    model,
    tokenizer,
    test_package: Dict[str, Any],
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 256
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Runs model inference on test split and computes:
    - Confusion Matrix
    - Full classification metrics (Accuracy, Macro-F1, Per-class Precision/Recall/F1, ROC-AUC)
    - Detailed per-sample prediction list with probabilities
    - Extracted False-Positive (FP) and False-Negative (FN) examples
    """
    model.eval()
    texts = test_package['texts']
    labels = test_package['labels']
    raw_records = test_package['records']

    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits.detach().cpu().numpy()
            all_logits.append(logits)

    logits = np.concatenate(all_logits, axis=0)

    # Numerically stable softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    preds = np.argmax(logits, axis=1)

    y_true = np.array(labels)
    y_pred = preds
    pos_probs = probs[:, 1]

    # Metrics computation
    acc = float(accuracy_score(y_true, y_pred))
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    try:
        auc = float(roc_auc_score(y_true, pos_probs))
    except Exception:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp = cm[0]
    fn, tp = cm[1]

    metrics = {
        'total_samples': len(y_true),
        'accuracy': acc,
        'macro_f1': float(macro_f1),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'roc_auc': float(auc),
        'anti_stereotype': {
            'precision': float(precisions[0]),
            'recall': float(recalls[0]),
            'f1': float(f1s[0]),
            'support': int(supports[0])
        },
        'stereotype': {
            'precision': float(precisions[1]),
            'recall': float(recalls[1]),
            'f1': float(f1s[1]),
            'support': int(supports[1])
        },
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'matrix': cm
        }
    }

    # Extract sample-level predictions and error examples
    predictions = []
    false_positives = []
    false_negatives = []

    id2label = {0: "anti_stereotype", 1: "stereotype"}

    for idx, r in enumerate(raw_records):
        true_l = int(y_true[idx])
        pred_l = int(y_pred[idx])
        p0 = float(probs[idx, 0])
        p1 = float(probs[idx, 1])

        if true_l == 1 and pred_l == 1:
            outcome = "TP"
        elif true_l == 0 and pred_l == 0:
            outcome = "TN"
        elif true_l == 0 and pred_l == 1:
            outcome = "FP"
        else:
            outcome = "FN"

        item_result = {
            'benchmark_id': r.get('id', f"idx_{idx}"),
            'context_id': r.get('context_id'),
            'sentence': r.get('sentence'),
            'classification': r.get('classification'),
            'annotator_ratings': r.get('annotator_ratings', []),
            'valid_annotator_ratings': r.get('valid_annotator_ratings', True),
            'true_label': true_l,
            'true_label_name': id2label[true_l],
            'pred_label': pred_l,
            'pred_label_name': id2label[pred_l],
            'prob_anti_stereotype': p0,
            'prob_stereotype': p1,
            'outcome': outcome,
            'is_correct': bool(true_l == pred_l)
        }
        predictions.append(item_result)

        if outcome == "FP":
            false_positives.append(item_result)
        elif outcome == "FN":
            false_negatives.append(item_result)

    return metrics, predictions, false_positives, false_negatives


def run_single_fold(
    fold_idx: int,
    records: List[Dict[str, Any]],
    output_dir: str,
    base_model: str = "microsoft/deberta-v3-base",
    epochs: int = 5,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 5e-6,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    adam_epsilon: float = 1e-6,
    max_length: int = 256,
    device: torch.device = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Trains and evaluates a single fold of GroupKFold."""
    print("\n" + "=" * 70)
    print(f"RUNNING EXPERIMENT FOR FOLD {fold_idx} (Test Fold: {fold_idx}, Val Fold: {(fold_idx + 1) % 5})")
    print("=" * 70)

    fold_out_dir = Path(output_dir) / f"fold_{fold_idx}"
    fold_out_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, test_data = load_fold_data(records, fold_idx)
    print(f"[Fold {fold_idx}] Train: {len(train_data['texts'])} samples ({len(set(r['context_id'] for r in train_data['records']))} contexts)")
    print(f"[Fold {fold_idx}] Val  : {len(val_data['texts'])} samples ({len(set(r['context_id'] for r in val_data['records']))} contexts)")
    print(f"[Fold {fold_idx}] Test : {len(test_data['texts'])} samples ({len(set(r['context_id'] for r in test_data['records']))} contexts)")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    train_enc = tokenizer(train_data['texts'], truncation=True, padding=True, max_length=max_length)
    val_enc = tokenizer(val_data['texts'], truncation=True, padding=True, max_length=max_length)

    train_dataset = StereotypeDataset(train_enc, train_data['labels'])
    val_dataset = StereotypeDataset(val_enc, val_data['labels'])

    id2label = {0: "anti_stereotype", 1: "stereotype"}
    label2id = {"anti_stereotype": 0, "stereotype": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    effective_batch_size = batch_size * grad_accum
    total_training_steps = (len(train_data['texts']) // effective_batch_size + 1) * epochs
    warmup_steps = max(1, int(total_training_steps * 0.1))

    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    raw_args = {
        "output_dir": str(fold_out_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "learning_rate": lr,
        "max_grad_norm": max_grad_norm,
        "adam_epsilon": adam_epsilon,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": 10,
        "save_total_limit": 1,
        "report_to": "none",
        "fp16": False
    }

    if "eval_strategy" in valid_params:
        raw_args["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in valid_params:
        raw_args["evaluation_strategy"] = "epoch"

    safe_args = {k: v for k, v in raw_args.items() if k in valid_params}
    training_args = TrainingArguments(**safe_args)

    trainer = DiagnosticTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print(f"\n[*] Training Fold {fold_idx}...")
    trainer.train()

    best_ckpt_dir = fold_out_dir / "best_model"
    trainer.save_model(str(best_ckpt_dir))
    tokenizer.save_pretrained(str(best_ckpt_dir))
    print(f"[+] Saved best checkpoint for Fold {fold_idx} to: {best_ckpt_dir}")

    # Evaluate on held-out test split
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\n[*] Evaluating Fold {fold_idx} on unseen test fold {fold_idx} ({len(test_data['texts'])} items)...")
    metrics, predictions, fps, fns = evaluate_test_split(
        model=model,
        tokenizer=tokenizer,
        test_package=test_data,
        device=device,
        batch_size=batch_size,
        max_length=max_length
    )

    # Save fold outputs
    with open(fold_out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(fold_out_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    with open(fold_out_dir / "false_positives.json", "w", encoding="utf-8") as f:
        json.dump(fps, f, indent=2)

    with open(fold_out_dir / "false_negatives.json", "w", encoding="utf-8") as f:
        json.dump(fns, f, indent=2)

    print(f"[+] Fold {fold_idx} Complete: Accuracy = {metrics['accuracy']:.4f}, Macro-F1 = {metrics['macro_f1']:.4f}, ROC-AUC = {metrics['roc_auc']:.4f}")
    print(f"    Stereotype Precision = {metrics['stereotype']['precision']:.4f}, Recall = {metrics['stereotype']['recall']:.4f}")
    print(f"    False Positives = {len(fps)}, False Negatives = {len(fns)}")

    return metrics, predictions, fps, fns


def run_5fold_evaluation(
    benchmark_path: str,
    output_dir: str,
    folds_to_run: List[int],
    base_model: str = "microsoft/deberta-v3-base",
    epochs: int = 5,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 5e-6,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    adam_epsilon: float = 1e-6,
    max_length: int = 256
):
    """Executes 5-fold cross-validation and computes aggregate mean ± std."""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = data['records']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    all_fold_metrics = {}
    pooled_cm = np.zeros((2, 2), dtype=int)
    all_fps = []
    all_fns = []

    for fold_idx in folds_to_run:
        metrics, predictions, fps, fns = run_single_fold(
            fold_idx=fold_idx,
            records=records,
            output_dir=output_dir,
            base_model=base_model,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            adam_epsilon=adam_epsilon,
            max_length=max_length,
            device=device
        )
        all_fold_metrics[f"fold_{fold_idx}"] = metrics
        cm = np.array(metrics['confusion_matrix']['matrix'])
        pooled_cm += cm
        all_fps.extend(fps)
        all_fns.extend(fns)

    # Compute summary statistics across evaluated folds
    metric_keys = [
        'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'roc_auc'
    ]
    stereo_keys = ['precision', 'recall', 'f1']
    anti_keys = ['precision', 'recall', 'f1']

    aggregate_report = {
        'folds_evaluated': folds_to_run,
        'per_fold': all_fold_metrics,
        'pooled_confusion_matrix': {
            'tn': int(pooled_cm[0, 0]),
            'fp': int(pooled_cm[0, 1]),
            'fn': int(pooled_cm[1, 0]),
            'tp': int(pooled_cm[1, 1]),
            'matrix': pooled_cm.tolist()
        },
        'summary_statistics': {}
    }

    print("\n" + "=" * 70)
    print("5-FOLD GROUPED CROSS-VALIDATION SUMMARY REPORT")
    print("=" * 70)

    for k in metric_keys:
        vals = [all_fold_metrics[f"fold_{f}"][k] for f in folds_to_run if not np.isnan(all_fold_metrics[f"fold_{f}"][k])]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        aggregate_report['summary_statistics'][k] = {'mean': mean_val, 'std': std_val}
        print(f"  • {k:<25}: {mean_val:.4f} ± {std_val:.4f}")

    for k in stereo_keys:
        vals = [all_fold_metrics[f"fold_{f}"]['stereotype'][k] for f in folds_to_run]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        aggregate_report['summary_statistics'][f"stereotype_{k}"] = {'mean': mean_val, 'std': std_val}
        print(f"  • {'stereotype_' + k:<25}: {mean_val:.4f} ± {std_val:.4f}")

    for k in anti_keys:
        vals = [all_fold_metrics[f"fold_{f}"]['anti_stereotype'][k] for f in folds_to_run]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        aggregate_report['summary_statistics'][f"anti_stereotype_{k}"] = {'mean': mean_val, 'std': std_val}
        print(f"  • {'anti_stereotype_' + k:<25}: {mean_val:.4f} ± {std_val:.4f}")

    out_file = Path(output_dir) / "deberta_5fold_eval_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(aggregate_report, f, indent=2)

    # Also mirror into data/benchmarks/
    mirror_file = Path("data/benchmarks/deberta_5fold_eval_report.json")
    mirror_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mirror_file, "w", encoding="utf-8") as f:
        json.dump(aggregate_report, f, indent=2)

    print(f"\n[+] Full 5-Fold Evaluation Report saved to: {out_file}")
    print(f"[+] Mirrored report saved to: {mirror_file}")


def evaluate_existing_checkpoint(
    checkpoint_path: str,
    benchmark_path: str,
    test_fold_idx: int,
    output_dir: str,
    max_length: int = 256
):
    """Evaluates an already trained checkpoint on its designated test fold without retraining."""
    print("=" * 70)
    print(f"EVALUATING EXISTING CHECKPOINT: {checkpoint_path}")
    print(f"Designated Test Fold: {test_fold_idx}")
    print("=" * 70)

    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = data['records']

    _, _, test_data = load_fold_data(records, test_fold_idx)
    print(f"[+] Loaded Test Fold {test_fold_idx}: {len(test_data['texts'])} samples across {len(set(r['context_id'] for r in test_data['records']))} contexts.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running inference on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(device)

    metrics, predictions, fps, fns = evaluate_test_split(
        model=model,
        tokenizer=tokenizer,
        test_package=test_data,
        device=device,
        max_length=max_length
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    with open(out_dir / "false_positives.json", "w", encoding="utf-8") as f:
        json.dump(fps, f, indent=2)

    with open(out_dir / "false_negatives.json", "w", encoding="utf-8") as f:
        json.dump(fns, f, indent=2)

    print("\n" + "=" * 50)
    print("TEST EVALUATION RESULTS:")
    print("=" * 50)
    print(f"Total Samples           : {metrics['total_samples']}")
    print(f"Accuracy                : {metrics['accuracy']:.4f}")
    print(f"Macro-F1                : {metrics['macro_f1']:.4f}")
    print(f"Macro-Precision         : {metrics['macro_precision']:.4f}")
    print(f"Macro-Recall            : {metrics['macro_recall']:.4f}")
    print(f"ROC-AUC                 : {metrics['roc_auc']:.4f}")
    print("-" * 50)
    print("Stereotype (Class 1) Metrics:")
    print(f"  Precision             : {metrics['stereotype']['precision']:.4f}")
    print(f"  Recall                : {metrics['stereotype']['recall']:.4f}")
    print(f"  F1-Score              : {metrics['stereotype']['f1']:.4f}")
    print(f"  Support               : {metrics['stereotype']['support']}")
    print("Anti-Stereotype (Class 0) Metrics:")
    print(f"  Precision             : {metrics['anti_stereotype']['precision']:.4f}")
    print(f"  Recall                : {metrics['anti_stereotype']['recall']:.4f}")
    print(f"  F1-Score              : {metrics['anti_stereotype']['f1']:.4f}")
    print(f"  Support               : {metrics['anti_stereotype']['support']}")
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"  True Negatives (TN)   : {metrics['confusion_matrix']['tn']}")
    print(f"  False Positives (FP)  : {metrics['confusion_matrix']['fp']}")
    print(f"  False Negatives (FN)  : {metrics['confusion_matrix']['fn']}")
    print(f"  True Positives (TP)   : {metrics['confusion_matrix']['tp']}")
    print("=" * 50)
    print(f"[+] Exported {len(fps)} False Positives to: {out_dir / 'false_positives.json'}")
    print(f"[+] Exported {len(fns)} False Negatives to: {out_dir / 'false_negatives.json'}")
    print(f"[+] Exported Full Predictions to: {out_dir / 'test_predictions.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproducible 5-Fold Grouped Cross-Validation Evaluation for DeBERTa-v3")
    parser.add_argument("--benchmark", type=str, default="data/benchmarks/indicasa_split_benchmark.json")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--base_model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Folds to run (default: 0 1 2 3 4)")
    parser.add_argument("--eval_only_checkpoint", type=str, default=None, help="Path to pre-trained checkpoint to evaluate without training")
    parser.add_argument("--test_fold", type=int, default=1, help="Test fold index when evaluating a single checkpoint")

    args = parser.parse_args()

    if args.eval_only_checkpoint:
        evaluate_existing_checkpoint(
            checkpoint_path=args.eval_only_checkpoint,
            benchmark_path=args.benchmark,
            test_fold_idx=args.test_fold,
            output_dir=args.output_dir,
            max_length=args.max_length
        )
    else:
        run_5fold_evaluation(
            benchmark_path=args.benchmark,
            output_dir=args.output_dir,
            folds_to_run=args.folds,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            adam_epsilon=args.adam_epsilon,
            max_length=args.max_length
        )

