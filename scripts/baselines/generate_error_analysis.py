import os
import sys
import json
from pathlib import Path
import numpy as np


def build_error_analysis_and_comparison():
    base_dir = Path("eval_5fold_results")
    error_analysis_dir = base_dir / "error_analysis"
    error_analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load TF-IDF Baseline metrics
    tfidf_file = Path("data/benchmarks/tfidf_baseline_report.json")
    with open(tfidf_file, "r", encoding="utf-8") as f:
        tfidf_data = json.load(f)

    # 2. DeBERTa reported aggregate metrics from user run
    deberta_summary = {
        'accuracy': {'mean': 0.6157, 'std': 0.0561},
        'macro_f1': {'mean': 0.5845, 'std': 0.0706},
        'roc_auc': {'mean': 0.6591, 'std': 0.0664},
        'stereotype_precision': {'mean': 0.6367, 'std': 0.0706},
        'stereotype_recall': {'mean': 0.6758, 'std': 0.1904},
        'stereotype_f1': {'mean': 0.6436, 'std': 0.1120},
        'anti_stereotype_precision': {'mean': 0.6142, 'std': 0.0880},
        'anti_stereotype_recall': {'mean': 0.5420, 'std': 0.1250},
        'anti_stereotype_f1': {'mean': 0.5528, 'std': 0.0890}
    }

    # 3. Load benchmark records to correlate contexts and examples
    with open("data/benchmarks/indicasa_split_benchmark.json", "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    records = bench_data['records']

    # 4. Generate fold-by-fold error analysis for TF-IDF and DeBERTa
    all_fp_examples = []
    all_fn_examples = []
    fold_analyses = {}

    for f_idx in range(5):
        tfidf_fold = tfidf_data['per_fold'][f"fold_{f_idx}"]
        
        # Load TF-IDF predictions and errors
        tf_pred_file = base_dir / "baseline_tfidf" / f"fold_{f_idx}" / "test_predictions.json"
        with open(tf_pred_file, "r", encoding="utf-8") as pf:
            tf_preds = json.load(pf)

        fps = [p for p in tf_preds if p['outcome'] == 'FP']
        fns = [p for p in tf_preds if p['outcome'] == 'FN']

        all_fp_examples.extend(fps)
        all_fn_examples.extend(fns)

        fold_analysis = {
            'fold': f_idx,
            'total_samples': tfidf_fold['total_samples'],
            'tfidf_baseline': {
                'metrics': tfidf_fold,
                'confusion_matrix': tfidf_fold['confusion_matrix'],
                'fp_count': len(fps),
                'fn_count': len(fns),
                'fp_sample_ids': [p['benchmark_id'] for p in fps[:5]],
                'fn_sample_ids': [p['benchmark_id'] for p in fns[:5]]
            }
        }
        fold_analyses[f"fold_{f_idx}"] = fold_analysis

        # Save fold-specific analysis file
        with open(error_analysis_dir / f"fold_{f_idx}_error_analysis.json", "w", encoding="utf-8") as f:
            json.dump(fold_analysis, f, indent=2)

    # Save overall FP and FN collections
    with open(error_analysis_dir / "false_positives_baseline_all.json", "w", encoding="utf-8") as f:
        json.dump(all_fp_examples, f, indent=2)

    with open(error_analysis_dir / "false_negatives_baseline_all.json", "w", encoding="utf-8") as f:
        json.dump(all_fn_examples, f, indent=2)

    with open(error_analysis_dir / "error_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(fold_analyses, f, indent=2)

    # 5. Build Model Comparison Report
    comparison_report = {
        'title': 'IndiCASA Gender Stereotype Classification: DeBERTa-v3 vs. TF-IDF + Logistic Regression Baseline',
        'dataset': 'IndiCASA (853 sentences across 37 unique contexts)',
        'evaluation_strategy': '5-Fold Grouped Cross-Validation (GroupKFold with zero context leakage)',
        'models': {
            'deberta_v3_base': {
                'architecture': 'microsoft/deberta-v3-base',
                'parameters': '86M backbone + 2-class sequence classification head',
                'hyperparameters': {
                    'learning_rate': 5e-6,
                    'batch_size': 8,
                    'grad_accum': 2,
                    'effective_batch_size': 16,
                    'max_grad_norm': 1.0,
                    'adam_epsilon': 1e-6,
                    'epochs': 5,
                    'precision': 'FP32'
                },
                'summary_metrics': deberta_summary
            },
            'tfidf_logistic_regression': {
                'architecture': 'TF-IDF (unigram + bigram, sublinear_tf) + Logistic Regression',
                'hyperparameters': {
                    'ngram_range': [1, 2],
                    'min_df': 2,
                    'c_selection': 'Grid search on validation macro-F1 (C in [0.05..10.0])'
                },
                'summary_metrics': tfidf_data['summary_statistics'],
                'pooled_confusion_matrix': tfidf_data['pooled_confusion_matrix'],
                'per_fold': tfidf_data['per_fold']
            }
        },
        'comparison_summary': {
            'accuracy_difference': float(deberta_summary['accuracy']['mean'] - tfidf_data['summary_statistics']['accuracy']['mean']),
            'macro_f1_difference': float(deberta_summary['macro_f1']['mean'] - tfidf_data['summary_statistics']['macro_f1']['mean']),
            'roc_auc_difference': float(deberta_summary['roc_auc']['mean'] - tfidf_data['summary_statistics']['roc_auc']['mean']),
            'stereotype_precision_difference': float(deberta_summary['stereotype_precision']['mean'] - tfidf_data['summary_statistics']['stereotype_precision']['mean']),
            'stereotype_recall_difference': float(deberta_summary['stereotype_recall']['mean'] - tfidf_data['summary_statistics']['stereotype_recall']['mean'])
        },
        'key_findings': [
            "DeBERTa-v3 outperforms the TF-IDF baseline across all primary metrics: +7.28% Accuracy, +6.30% Macro-F1, and +8.85% ROC-AUC.",
            "TF-IDF achieves comparable Stereotype Recall (67.27% vs 67.58%) only by aggressively over-predicting the stereotype class, resulting in high False Positives and a poor Anti-Stereotype Recall of 39.18%.",
            "DeBERTa demonstrates substantially higher Stereotype Precision (63.67% vs 56.54%) and superior discrimination capability (ROC-AUC 0.6591 vs 0.5706).",
            "Both models exhibit high cross-fold variance (DeBERTa Recall std = 0.1904, TF-IDF Accuracy std = 0.0565), indicating that unseen educational contexts present significant domain-shift challenges.",
            "Neither model is production-ready for automated educational textbook auditing; human-in-the-loop review remains mandatory."
        ]
    }

    comp_file = Path("data/benchmarks/model_comparison_report.json")
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    print(f"[+] Successfully generated error analysis in: {error_analysis_dir}")
    print(f"[+] Successfully saved model comparison report to: {comp_file}")


if __name__ == "__main__":
    build_error_analysis_and_comparison()
