# -*- coding: utf-8 -*-
# eval.py — Metrics plots and helpers
# type: ignore

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

plt.rcParams.update({'font.size': 14})


def plot_label_distribution(y_encoded, label_encoder, save_path):
    """
    Plot class distribution after any augmentation steps.
    Args:
        y_encoded: np.ndarray of encoded labels (ints)
        label_encoder: fitted LabelEncoder to map ints->names
        save_path: output png path
    """
    names = dict(enumerate(label_encoder.classes_))
    series = pd.Series([names[i] for i in y_encoded])
    palette = sns.color_palette('tab10', len(series.unique()))
    plt.figure(figsize=(8, 5))
    # sns.countplot(x=series, order=sorted(series.unique()), palette=palette)
    # sns.countplot(x=series, hue=series, order=sorted(series.unique()), palette=palette, legend=False)
    sns.countplot(x=series, hue=series, order=sorted(series.unique()), palette=palette)
    plt.xlabel('Class'); plt.ylabel('Count'); plt.xticks(rotation=45); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()


def plot_all_figures(batch_size, epoch, history, y_true_labels, y_pred_labels, y_pred_probs,
                     categories, result_out, model_name):
    """
    Produce standard set of figures: Accuracy/Loss curves, normalized confusion matrix,
    ROC and Precision-Recall curves (binary or multi-class one-vs-rest).
    """
    os.makedirs(result_out, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_accuracy'], '--o', label='Train')
    plt.plot(history['val_accuracy'], '-o', label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(result_out, f"{model_name}_bs{batch_size}_ep{epoch}_accuracy.png")); plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], '--o', label='Train')
    plt.plot(history['val_loss'], '-o', label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(result_out, f"{model_name}_bs{batch_size}_ep{epoch}_loss.png")); plt.close()

    # Confusion matrix (normalized)
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(os.path.join(result_out, f"{model_name}_cm.png")); plt.close()

    # ROC
    le = LabelEncoder(); y_true_enc = le.fit_transform(y_true_labels)
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 8))
    if len(categories) == 2:
        ybin = (y_true_enc == 1).astype(int)
        fpr, tpr, _ = roc_curve(ybin, y_pred_probs[:, 1]); roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{categories[1]} AUC={roc_auc:.4f}')
        ybin = (y_true_enc == 0).astype(int)
        fpr, tpr, _ = roc_curve(ybin, y_pred_probs[:, 0]); roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{categories[0]} AUC={roc_auc:.4f}')
    else:
        for i, c in enumerate(categories):
            ybin = (y_true_enc == i).astype(int)
            fpr, tpr, _ = roc_curve(ybin, y_pred_probs[:, i]); roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{c} AUC={roc_auc:.4f}', color=colors[i % len(colors)])
    plt.plot([0, 1], [0, 1], 'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(result_out, f"{model_name}_roc.png")); plt.close()

    # PR
    plt.figure(figsize=(10, 8))
    if len(categories) == 2:
        ybin = (y_true_enc == 1).astype(int)
        prec, rec, _ = precision_recall_curve(ybin, y_pred_probs[:, 1]); pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f'{categories[1]} PR-AUC={pr_auc:.4f}')
        ybin = (y_true_enc == 0).astype(int)
        prec, rec, _ = precision_recall_curve(ybin, y_pred_probs[:, 0]); pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f'{categories[0]} PR-AUC={pr_auc:.4f}')
    else:
        for i, c in enumerate(categories):
            ybin = (y_true_enc == i).astype(int)
            prec, rec, _ = precision_recall_curve(ybin, y_pred_probs[:, i]); pr_auc = auc(rec, prec)
            plt.plot(rec, prec, label=f'{c} PR-AUC={pr_auc:.4f}', color=colors[i % len(colors)])
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(result_out, f"{model_name}_pr.png")); plt.close()
