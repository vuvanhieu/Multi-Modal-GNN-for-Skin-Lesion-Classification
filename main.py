# -*- coding: utf-8 -*-
# main_ham10000.py — Runner for HAM10000 (7-class)
# type: ignore
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from config import set_global_seed, SEED
set_global_seed(SEED)

from config import (
    CATEGORIES,            # ['VASC','DF','BKL','AKIEC','BCC','NV','MEL']
    FEATURE_DIR,           # e.g., .../data2/data2_HAM10000_FEATURES/...
    PROJECT_DIR,
    TRAIN_METADATA_PATH,   # HAM10000 metadata (có 'dx')
    BATCH_SIZE_LIST, EPOCH_VALUES,
    FEATURE_TYPES_DEEP, FEATURE_TYPES_HAND,
    DB_PREFIX, get_baselines
)

from data import (
    ham_generate_paths,                 # dùng cho HAM (chỉ train folders)
    load_features_without_smote,        # -> (X, y, ids)
    prepare_clinical_encoder,
    extract_clinical_features_from_list
)
from train import run_experiment


def main():
    # ===== 1) Metadata & encoders =====
    meta = pd.read_csv(TRAIN_METADATA_PATH)
    if "dx" not in meta.columns:
        raise ValueError("HAM10000 metadata cần cột 'dx'.")
    id_col = "image_name" if "image_name" in meta.columns else ("image_id" if "image_id" in meta.columns else None)
    if id_col is None:
        raise ValueError("Không tìm thấy cột image id (image_name/image_id).")

    one_hot_encoder, age_scaler = prepare_clinical_encoder(meta)

    # ===== 2) Baselines =====
    baselines = get_baselines(DB_PREFIX)
    all_fold_metrics = []

    for cfg in baselines:
        model_name = cfg["model_name"]
        use_deep  = cfg["use_deep_feature"]
        use_hand  = cfg["use_handcraft_feature"]
        use_cli   = cfg["use_clinical_feature"]
        gnn_type  = cfg["gnn_type"]
        fusion    = cfg["fusion_type"]
        use_lgc   = cfg.get("use_lgc", False)

        feature_types = []
        if use_deep: feature_types += FEATURE_TYPES_DEEP
        if use_hand: feature_types += FEATURE_TYPES_HAND

        result_dir = os.path.join(PROJECT_DIR, model_name)
        os.makedirs(result_dir, exist_ok=True)

        print(f"\n=== Running: {model_name} | GNN={gnn_type} | Fusion={fusion} | LGC={use_lgc} ===")
        print(f"Features: {feature_types} | Clinical={use_cli}")

        batch_size = BATCH_SIZE_LIST[0] if BATCH_SIZE_LIST else 32
        epochs     = EPOCH_VALUES[0]     if EPOCH_VALUES     else 10

        # ===== Clinical-only (GFC) =====
        if (not feature_types) and use_cli:
            # X = clinical cho toàn bộ ids có trong metadata
            train_ids = meta[id_col].astype(str).tolist()
            X_cli = extract_clinical_features_from_list(train_ids, meta, one_hot_encoder, age_scaler)

            # Map nhãn 7 lớp theo CATEGORIES
            cmap = {c.lower(): i for i, c in enumerate([c.lower() for c in CATEGORIES])}
            y_raw = meta[meta[id_col].astype(str).isin(train_ids)]["dx"].astype(str).str.lower()
            y_train = y_raw.map(cmap).values

            print(f"[DEBUG][GFC] X_cli: {X_cli.shape}, y: {y_train.shape}")
            # Clinical-only → KHÔNG truyền X_clinical (tránh ghép đúp)
            _, fold_metrics = run_experiment(
                X_cli, y_train, CATEGORIES, batch_size, epochs, result_dir, model_name,
                gnn_type=gnn_type, fusion_type=fusion, use_lgc=use_lgc,
                X_clinical=None, one_hot_encoder=one_hot_encoder, age_scaler=age_scaler
            )
            all_fold_metrics.append(fold_metrics)
            continue

        # ===== Deep/Hand ± Clinical =====
        if feature_types:
            driver_ft = feature_types[0]
            train_paths = ham_generate_paths(FEATURE_DIR, feature_types, CATEGORIES)
            print(f"[DEBUG] train_paths: {train_paths}")
            print(f"[DEBUG] Loading features for feature_type: {driver_ft}")

            X_base, y_train, ids_train = load_features_without_smote(train_paths, driver_ft, CATEGORIES, return_ids=True)
            print(f"[DEBUG] X_base: {X_base.shape}, y_train: {y_train.shape}")

            # Clinical RIÊNG
            X_clin = None
            if use_cli:
                X_clin = extract_clinical_features_from_list(ids_train, meta, one_hot_encoder, age_scaler)
                print(f"[DEBUG] Pre-check rows -> X_base: {X_base.shape[0]}, X_clinical: {X_clin.shape[0]}")

            # Augment dữ liệu và đồng bộ lại X_clinical
            X_base_aug, y_train_aug, indices_aug = X_base, y_train, None
            try:
                from data import augment_all_classes_to_balance
                X_base_aug, y_train_aug, indices_aug = augment_all_classes_to_balance(X_base, y_train)
            except ImportError:
                pass
            if use_cli and X_clin is not None and indices_aug is not None:
                X_clin = X_clin[indices_aug[indices_aug != -1]]
                print(f"[DEBUG] After augment: X_base: {X_base_aug.shape[0]}, X_clinical: {X_clin.shape[0]}")

            # Train
            _, fold_metrics = run_experiment(
                X_base_aug, y_train_aug, CATEGORIES, batch_size, epochs, result_dir, model_name,
                gnn_type=gnn_type, fusion_type=fusion, use_lgc=use_lgc,
                X_clinical=X_clin, one_hot_encoder=one_hot_encoder, age_scaler=age_scaler
            )
            all_fold_metrics.append(fold_metrics)
        else:
            raise ValueError("Cấu hình baseline không hợp lệ: không có non-clinical features và cũng không phải GFC.")

    print("\n=== All Done ===")

    # ===== 4) Summarize results across baselines =====
    rows = []
    for idx, metrics in enumerate(all_fold_metrics):
        df = pd.DataFrame(metrics)
        row = {}
        if idx < len(baselines):
            row["Model"] = baselines[idx]["model_name"]
            row["Desc"]  = baselines[idx].get("desc", "")
        else:
            row["Model"] = f"Model {idx+1}"
            row["Desc"]  = ""
        for k in ["Accuracy","Precision","Recall","F1","Macro AUC","Sensitivity","Specificity","Best Epoch","Best Val Acc","Training Time (s)"]:
            if k in df:
                row[f"{k} Mean"] = df[k].mean()
                row[f"{k} Std"]  = df[k].std()
            else:
                row[f"{k} Mean"] = None
                row[f"{k} Std"]  = None
        rows.append(row)
    summary = pd.DataFrame(rows)
    out_csv = os.path.join(PROJECT_DIR, f"{DB_PREFIX}_baseline_metrics_summary.csv")
    summary.to_csv(out_csv, index=False)
    print("\n===== SUMMARY =====")
    print(summary)


if __name__ == "__main__":
    main()
