# -*- coding: utf-8 -*-
# train.py — Training utilities for multimodal GNN
# type: ignore

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from eval import plot_all_figures, plot_label_distribution
from data import normalize_data, augment_all_classes_to_balance

from model import GFDHCM_GNN
from model import GNNClassifier
from model import GraphSAGEClassifier
from model import GATClassifier
from model import GINClassifier
from model import GatingFusionGNN, CrossAttentionFusionGNN, AdaptiveFusionGNN, LGC_GNN
from config import get_baselines, DB_PREFIX

# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
                
def create_graph(features, labels, train_idx=None, test_idx=None, k=5):
    from sklearn.neighbors import kneighbors_graph
    edges = kneighbors_graph(features, n_neighbors=k, mode="connectivity", include_self=False)
    edge_index = torch.tensor(np.array(edges.nonzero()), dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)


def train_gnn_model(model, data, optimizer, epochs, device="cpu"):

    model.to(device); data = data.to(device)
    y_np = data.y.cpu().numpy()
    classes = np.unique(y_np)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_np)
    class_weights = torch.tensor(weights, dtype=torch.float, device=device)
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_state, best_val_loss = None, float("inf")
    early_stopping = EarlyStopping(patience=10, min_delta=0.0)
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            out_eval = model(data)
            train_preds = out_eval[data.train_mask].argmax(dim=1)
            train_acc = (train_preds == data.y[data.train_mask]).float().mean().item()
            train_loss = F.nll_loss(out_eval[data.train_mask], data.y[data.train_mask], weight=class_weights).item()
            val_preds = out_eval[data.test_mask].argmax(dim=1)
            val_acc = (val_preds == data.y[data.test_mask]).float().mean().item()
            val_loss = F.nll_loss(out_eval[data.test_mask], data.y[data.test_mask], weight=class_weights).item()
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc); history["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss, best_state = val_loss, model.state_dict()
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    if best_state: model.load_state_dict(best_state)
    return model, history


# def run_experiment(X, y, categories, batch_size, epochs, result_out, model_name):
def run_experiment(X, y, categories, batch_size, epochs, result_out, model_name, gnn_type=None, fusion_type=None, use_lgc=False, X_clinical=None, one_hot_encoder=None, age_scaler=None):
    # Dữ liệu đã được augment và đồng bộ ở main.py, không cần augment lại hoặc dùng indices ở đây
        
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    histories, metrics = [], []
    label_encoder = LabelEncoder(); y_enc = label_encoder.fit_transform(y)
    dist_plot = os.path.join(result_out, "label_distribution.png")
    plot_label_distribution(y_enc, label_encoder, dist_plot)
    # Get desc from config
    try:
        
        baselines = get_baselines(DB_PREFIX)
        desc = None
        for cfg in baselines:
            if cfg["model_name"] == model_name:
                desc = cfg.get("desc", "")
                break
    except Exception:
        desc = ""
    fold = 1
    for train_idx, test_idx in rkf.split(X, y_enc):
        print(f"Fold {fold}")
        # Nếu có dữ liệu clinical, kết hợp với X
        if X_clinical is not None:
            if X_clinical.shape[0] != X.shape[0]:
                raise ValueError("X_clinical và X phải có cùng số mẫu!")
            X_combined = np.concatenate([X, X_clinical], axis=1)
            split_indices = [(0, X.shape[1]), (X.shape[1], X.shape[1] + X_clinical.shape[1])]
        elif X_clinical is not None:
            if X_clinical.shape[0] != X.shape[0]:
                raise ValueError("X_clinical và X phải có cùng số mẫu!")
            X_combined = np.concatenate([X, X_clinical], axis=1)
            split_indices = [(0, X.shape[1]), (X.shape[1], X.shape[1] + X_clinical.shape[1])]
        else:
            X_combined = X
            split_indices = None
        Xn, _ = normalize_data(X_combined, X_combined)
        data = create_graph(Xn, y_enc, train_idx, test_idx, k=5)
        input_dim, num_classes = Xn.shape[1], len(categories)
        # Chọn mô hình đúng với cấu hình
        if use_lgc:
            model = LGC_GNN(input_dim, 64, num_classes)
        elif fusion_type == 'gating':
            model = GatingFusionGNN(input_dim, 64, num_classes)
        elif fusion_type == 'cross_attention':
            model = CrossAttentionFusionGNN(input_dim, 64, num_classes)
        elif fusion_type == 'adaptive':
            if split_indices is not None:
                model = AdaptiveFusionGNN(input_dim, 64, num_classes, split_indices=split_indices)
            else:
                model = AdaptiveFusionGNN(input_dim, 64, num_classes)
            # model = AdaptiveFusionGNN(input_dim, hidden_dim, output_dim, dropout=0.4, split_indices=split_indices)
        elif fusion_type == 'concat' and gnn_type == 'gcn' and model_name == 'GFDHCM':
            model = GFDHCM_GNN(input_dim, 64, num_classes, dropout=0.4)
        elif fusion_type == 'none' and gnn_type == 'gcn' and model_name == 'GFC':
            from model import GFC_GNN
            model = GFC_GNN(input_dim, 64, num_classes, dropout=0.4)
        elif gnn_type == 'gcn':
            model = GNNClassifier(input_dim, 64, num_classes)
        elif gnn_type == 'sage':
            model = GraphSAGEClassifier(input_dim, 64, num_classes)
        elif gnn_type == 'gat':
            model = GATClassifier(input_dim, 64, num_classes)
        elif gnn_type == 'gin':
            model = GINClassifier(input_dim, 64, num_classes)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model, history = train_gnn_model(model, data, optimizer, epochs)
        histories.append(history)
        model.eval()
        with torch.no_grad():
            out = model(data)
            probs = F.softmax(out[data.test_mask], dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            true = y_enc[data.test_mask.cpu().numpy()]
        acc = accuracy_score(true, preds)
        prec = precision_score(true, preds, average="weighted", zero_division=0)
        rec = recall_score(true, preds, average="weighted", zero_division=0)
        f1 = f1_score(true, preds, average="weighted", zero_division=0)
        if len(categories) == 2:
            auc_val = roc_auc_score(true, probs[:, 1])
        else:
            auc_val = roc_auc_score(true, probs, multi_class="ovr")
        report = classification_report(true, preds, target_names=categories)
        cm = confusion_matrix(true, preds)
        sens = cm[1,1]/(cm[1,0]+cm[1,1]) if cm.shape[0]>1 and (cm[1,0]+cm[1,1]) > 0 else 0
        spec = cm[0,0]/(cm[0,0]+cm[0,1]) if cm.shape[0]>1 and (cm[0,0]+cm[0,1]) > 0 else 0
        # Best Epoch và Best Val Acc cho fold này
        if "val_accuracy" in history:
            best_epoch = int(np.argmax(history["val_accuracy"]))
            best_val_acc = float(np.max(history["val_accuracy"]))
        else:
            best_epoch = None
            best_val_acc = None
        # Training Time (s) cho fold này (số epoch)
        train_time = len(history["train_loss"]) if "train_loss" in history else None
        metrics.append({
            "Fold": fold,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Macro AUC": auc_val,
            "Sensitivity": sens,
            "Specificity": spec,
            "Best Epoch": best_epoch,
            "Best Val Acc": best_val_acc,
            "Training Time (s)": train_time
        })
        fold_out = os.path.join(result_out, f"fold_{fold}"); os.makedirs(fold_out, exist_ok=True)
        plot_all_figures(batch_size, epochs, history, true, preds, probs, categories, fold_out, model_name)
        with open(os.path.join(fold_out, "report.txt"), "w") as f: f.write(report)
        fold += 1
    # Lưu bảng chỉ số từng fold
    df_metrics = pd.DataFrame(metrics)
    df_metrics["Model"] = model_name
    df_metrics["Desc"] = desc
    df_metrics.to_csv(os.path.join(result_out, f"{model_name}_all_fold_metrics.csv"), index=False)

    # Tính trung bình và độ lệch chuẩn cho các chỉ số hiệu năng
    summary = {}
    for col in ["Accuracy", "Precision", "Recall", "F1", "AUC", "Sensitivity", "Specificity"]:
        if col in df_metrics:
            summary[f"{col} Mean"] = df_metrics[col].mean()
            summary[f"{col} Std"] = df_metrics[col].std()

    # Tính Best Epoch và Best Val Acc từ histories
    best_epochs = []
    best_val_accs = []
    train_times = []
    for hist in histories:
        if "val_accuracy" in hist:
            best_epoch = int(np.argmax(hist["val_accuracy"]))
            best_val_acc = float(np.max(hist["val_accuracy"]))
            best_epochs.append(best_epoch)
            best_val_accs.append(best_val_acc)
        if "train_loss" in hist and "val_loss" in hist:
            # Giả sử mỗi epoch mất 1 đơn vị thời gian, hoặc có thể đo thời gian thực tế nếu có
            train_times.append(len(hist["train_loss"]))
    summary["Best Epoch Mean"] = np.mean(best_epochs) if best_epochs else None
    summary["Best Epoch Std"] = np.std(best_epochs) if best_epochs else None
    summary["Best Val Acc Mean"] = np.mean(best_val_accs) if best_val_accs else None
    summary["Best Val Acc Std"] = np.std(best_val_accs) if best_val_accs else None
    summary["Training Time Mean (s)"] = np.mean(train_times) if train_times else None
    summary["Training Time Std (s)"] = np.std(train_times) if train_times else None
    # Có thể bổ sung các chỉ số khác nếu cần
    # Lưu bảng tổng hợp hiệu năng
    summary["Model"] = model_name
    summary["Desc"] = desc
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(os.path.join(result_out, f"{model_name}_metrics_summary.csv"), index=False)

    return histories, metrics
