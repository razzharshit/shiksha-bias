import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def load_benchmark_records(benchmark_path: str) -> List[Dict[str, Any]]:
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['records']


def get_fold_partitions(records: List[Dict[str, Any]], test_fold: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Partitions benchmark records strictly by group_kfold:
    - Test: group_kfold == test_fold
    - Val: group_kfold == (test_fold + 1) % 5
    - Train: remaining 3 folds
    """
    label_map = {"anti_stereotype": 0, "stereotype": 1}
    val_fold = (test_fold + 1) % 5

    train_recs, val_recs, test_recs = [], [], []

    for r in records:
        f = r['group_kfold']
        if f == test_fold:
            test_recs.append(r)
        elif f == val_fold:
            val_recs.append(r)
        else:
            train_recs.append(r)

    # Strictly verify context isolation
    train_ctx = set(r['context_id'] for r in train_recs)
    val_ctx = set(r['context_id'] for r in val_recs)
    test_ctx = set(r['context_id'] for r in test_recs)

    assert not (train_ctx & val_ctx), f"Context overlap between Train and Val in Fold {test_fold}!"
    assert not (train_ctx & test_ctx), f"Context overlap between Train and Test in Fold {test_fold}!"
    assert not (val_ctx & test_ctx), f"Context overlap between Val and Test in Fold {test_fold}!"

    def pkg(recs):
        return {
            'texts': [r['sentence'].strip() for r in recs],
            'labels': [label_map[r['classification']] for r in recs],
            'records': recs
        }

    return pkg(train_recs), pkg(val_recs), pkg(test_recs)


def evaluate_split(clf, vectorizer, split_pkg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    texts = split_pkg['texts']
    labels = split_pkg['labels']
    raw_records = split_pkg['records']

    X = vectorizer.transform(texts)
    y_true = np.array(labels)
    y_pred = clf.predict(X)
    probs = clf.predict_proba(X)
    pos_probs = probs[:, 1]

    acc = float(accuracy_score(y_true, y_pred))
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    try:
        auc = float(roc_auc_score(y_true, pos_probs))
    except Exception:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
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

    id2label = {0: "anti_stereotype", 1: "stereotype"}
    predictions = []
    fps = []
    fns = []

    for idx, r in enumerate(raw_records):
        t_l = int(y_true[idx])
        p_l = int(y_pred[idx])
        p0 = float(probs[idx, 0])
        p1 = float(probs[idx, 1])

        if t_l == 1 and p_l == 1:
            outcome = "TP"
        elif t_l == 0 and p_l == 0:
            outcome = "TN"
        elif t_l == 0 and p_l == 1:
            outcome = "FP"
        else:
            outcome = "FN"

        item = {
            'benchmark_id': r.get('id', f"idx_{idx}"),
            'context_id': r.get('context_id'),
            'sentence': r.get('sentence'),
            'classification': r.get('classification'),
            'true_label': t_l,
            'true_label_name': id2label[t_l],
            'pred_label': p_l,
            'pred_label_name': id2label[p_l],
            'prob_anti_stereotype': p0,
            'prob_stereotype': p1,
            'outcome': outcome,
            'is_correct': bool(t_l == p_l)
        }
        predictions.append(item)

        if outcome == "FP":
            fps.append(item)
        elif outcome == "FN":
            fns.append(item)

    return metrics, predictions, fps, fns


def train_eval_fold(records: List[Dict[str, Any]], fold_idx: int, output_dir: Path) -> Dict[str, Any]:
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_pkg, val_pkg, test_pkg = get_fold_partitions(records, fold_idx)

    print(f"\n--- FOLD {fold_idx} ---")
    print(f"Train: {len(train_pkg['texts'])} samples | Val: {len(val_pkg['texts'])} samples | Test: {len(test_pkg['texts'])} samples")

    # Fit TF-IDF only on the training partition (unigrams + bigrams)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_pkg['texts'])
    y_train = np.array(train_pkg['labels'])

    X_val = vectorizer.transform(val_pkg['texts'])
    y_val = np.array(val_pkg['labels'])

    # Hyperparameter selection using validation Macro-F1 only
    best_c = 1.0
    best_val_f1 = -1.0
    candidate_c = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for c in candidate_c:
        clf_try = LogisticRegression(C=c, max_iter=1000, random_state=42)
        clf_try.fit(X_train, y_train)
        y_val_pred = clf_try.predict(X_val)
        _, _, v_f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='macro', zero_division=0)
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_c = c

    print(f"Selected hyperparameter C = {best_c} (Validation Macro-F1 = {best_val_f1:.4f})")

    # Retrain model on train with best C (or evaluate directly)
    best_clf = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
    best_clf.fit(X_train, y_train)

    # Evaluate on held-out test fold
    metrics, predictions, fps, fns = evaluate_split(best_clf, vectorizer, test_pkg)

    metrics['best_c'] = best_c
    metrics['val_macro_f1'] = float(best_val_f1)

    with open(fold_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(fold_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    with open(fold_dir / "false_positives.json", "w", encoding="utf-8") as f:
        json.dump(fps, f, indent=2)

    with open(fold_dir / "false_negatives.json", "w", encoding="utf-8") as f:
        json.dump(fns, f, indent=2)

    print(f"Test Accuracy: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Stereotype Precision: {metrics['stereotype']['precision']:.4f} | Recall: {metrics['stereotype']['recall']:.4f}")
    print(f"False Positives: {len(fps)} | False Negatives: {len(fns)}")

    return metrics


def run_baseline_5fold(benchmark_path: str, output_dir: str):
    records = load_benchmark_records(benchmark_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_fold_metrics = {}
    pooled_cm = np.zeros((2, 2), dtype=int)

    for f_idx in range(5):
        m = train_eval_fold(records, f_idx, out_path)
        all_fold_metrics[f"fold_{f_idx}"] = m
        pooled_cm += np.array(m['confusion_matrix']['matrix'])

    metric_keys = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'roc_auc']
    stereo_keys = ['precision', 'recall', 'f1']
    anti_keys = ['precision', 'recall', 'f1']

    summary_stats = {}
    print("\n" + "=" * 70)
    print("TF-IDF + LOGISTIC REGRESSION 5-FOLD BASELINE SUMMARY")
    print("=" * 70)

    for k in metric_keys:
        vals = [all_fold_metrics[f"fold_{f}"][k] for f in range(5)]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        summary_stats[k] = {'mean': mean_val, 'std': std_val}
        print(f"  • {k:<25}: {mean_val:.4f} ± {std_val:.4f}")

    for k in stereo_keys:
        vals = [all_fold_metrics[f"fold_{f}"]['stereotype'][k] for f in range(5)]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        summary_stats[f"stereotype_{k}"] = {'mean': mean_val, 'std': std_val}
        print(f"  • {'stereotype_' + k:<25}: {mean_val:.4f} ± {std_val:.4f}")

    for k in anti_keys:
        vals = [all_fold_metrics[f"fold_{f}"]['anti_stereotype'][k] for f in range(5)]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        summary_stats[f"anti_stereotype_{k}"] = {'mean': mean_val, 'std': std_val}
        print(f"  • {'anti_stereotype_' + k:<25}: {mean_val:.4f} ± {std_val:.4f}")

    report = {
        'model': 'TF-IDF (unigrams + bigrams) + Logistic Regression',
        'per_fold': all_fold_metrics,
        'pooled_confusion_matrix': {
            'tn': int(pooled_cm[0, 0]),
            'fp': int(pooled_cm[0, 1]),
            'fn': int(pooled_cm[1, 0]),
            'tp': int(pooled_cm[1, 1]),
            'matrix': pooled_cm.tolist()
        },
        'summary_statistics': summary_stats
    }

    report_path = out_path / "tfidf_baseline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    mirror_path = Path("data/benchmarks/tfidf_baseline_report.json")
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mirror_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Baseline report saved to: {report_path}")
    print(f"[+] Mirrored report saved to: {mirror_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression Baseline on GroupKFold")
    parser.add_argument("--benchmark", type=str, default="data/benchmarks/indicasa_split_benchmark.json")
    parser.add_argument("--output_dir", type=str, default="./eval_5fold_results/baseline_tfidf")
    args = parser.parse_args()

    run_baseline_5fold(args.benchmark, args.output_dir)
