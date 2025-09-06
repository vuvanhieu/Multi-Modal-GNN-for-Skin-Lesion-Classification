# -*- coding: utf-8 -*-
# data.py — Data loading, preprocessing and path helpers
# type: ignore

import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# -----------------
# Metadata helpers
# -----------------

def convert_isic_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ISIC metadata column names to: image_id, sex, localization, age.
    Accepts ISIC 2020 CSV variants.
    """
    df = df.rename(columns={
        'image': 'image_id',
        'age_approx': 'age',
        'anatom_site_general': 'localization'
    })
    for col in ['image_id', 'sex', 'localization', 'age']:
        if col not in df.columns:
            df[col] = np.nan
    return df


def prepare_clinical_encoder(metadata_df: pd.DataFrame):
    """
    Fit OneHotEncoder for (sex, localization) and StandardScaler for (age).
    Handles sklearn version differences for OneHotEncoder's `sparse` vs `sparse_output`.
    """
    metadata_df = metadata_df.copy()
    metadata_df['sex'] = metadata_df['sex'].fillna('unknown')
    metadata_df['localization'] = metadata_df['localization'].fillna('unknown')
    metadata_df['age'] = metadata_df['age'].fillna(metadata_df['age'].mean())

    # Handle sklearn versions (sparse vs sparse_output)
    try:
        import inspect
        sig = inspect.signature(OneHotEncoder)
        if 'sparse' in sig.parameters:
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        else:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except Exception:
        try:
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        except TypeError:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    ohe.fit(metadata_df[['sex', 'localization']])

    scaler = StandardScaler()
    scaler.fit(metadata_df[['age']])
    return ohe, scaler


def extract_clinical_features_from_list(file_names, metadata_df, one_hot_encoder, age_scaler):
    """
    Given a list of feature file names (containing ISIC IDs), return concatenated
    one-hot (sex, localization) and scaled age features aligned to those files.
    If a file name has no matching metadata, fall back to zeros of correct width.
    """
    image_ids = [
        re.search(r'(ISIC_\d+)', f).group(1) if re.search(r'(ISIC_\d+)', f) else None
        for f in file_names
    ]
    image_ids_clean = [i for i in image_ids if i is not None]
    matched = metadata_df[metadata_df['image_id'].isin(image_ids_clean)].copy()
    if matched.empty:
        return np.zeros((
            len(file_names),
            one_hot_encoder.transform([['unknown', 'unknown']]).shape[1] + 1
        ))
    matched = matched.set_index('image_id').reindex(image_ids_clean)
    cat = one_hot_encoder.transform(matched[['sex', 'localization']].fillna('unknown'))
    age = age_scaler.transform(matched[['age']].fillna(age_scaler.mean_[0]))
    return np.hstack([cat, age])


# -----------------
# Feature loading
# -----------------

def load_features_without_smote(feature_paths, feature_type, categories, image_ids=None, return_ids=False, return_filenames=False):
    """
    Load per-category .npy feature matrices from directory structure:
      feature_dir/dataset_type/<feature_type>/<category>/*.npy
    Returns X, y (encoded with LabelEncoder). Optionally returns list of filenames.
    """
    all_features, all_labels, all_fns, all_ids = [], [], [], []
    le = LabelEncoder()
    for cat in categories:
        if cat not in feature_paths or feature_type not in feature_paths[cat]:
            continue
        folder = feature_paths[cat][feature_type]
        if not os.path.isdir(folder):
            continue
        feats = []
        files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
        for fn in files:
            arr = np.load(os.path.join(folder, fn))
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            feats.append(arr)
            # Lấy id từ tên file (bỏ phần mở rộng)
            img_id = os.path.splitext(fn)[0]
            all_ids.append(img_id)
            if return_filenames:
                all_fns.append(fn)
        if len(feats):
            mat = np.vstack(feats)
            all_features.append(mat)
            all_labels += [cat] * mat.shape[0]
    if len(all_features) == 0:
        if return_ids:
            return np.array([]), np.array([]), []
        elif return_filenames:
            return np.array([]), np.array([]), []
        else:
            return np.array([]), np.array([])
    X = np.vstack(all_features)
    y = le.fit_transform(np.array(all_labels))
    # Nếu truyền image_ids, chỉ giữ lại các mẫu có id trong image_ids và đúng thứ tự
    if image_ids is not None:
        id_to_idx = {img_id: idx for idx, img_id in enumerate(all_ids)}
        valid_idxs = [id_to_idx[img_id] for img_id in image_ids if img_id in id_to_idx]
        X = X[valid_idxs]
        y = y[valid_idxs]
        all_ids = [image_ids[i] for i in range(len(image_ids)) if image_ids[i] in id_to_idx]
    if return_ids:
        return X, y, all_ids
    if return_filenames:
        return X, y, all_fns
    return X, y


# -----------------
# Utilities
# -----------------

def normalize_data(train_data, test_data):
    """
    Standardize features; impute NaNs with column means first.
    Returns (train_scaled, test_scaled).
    """
    scaler = StandardScaler()
    if np.isnan(train_data).any():
        col_mean = np.nanmean(train_data, axis=0)
        train_data = np.where(np.isnan(train_data), col_mean, train_data)
    if np.isnan(test_data).any():
        col_mean = np.nanmean(test_data, axis=0)
        test_data = np.where(np.isnan(test_data), col_mean, test_data)
    return scaler.fit_transform(train_data), scaler.transform(test_data)


def augment_all_classes_to_balance(X, y, noise_std=0.01, random_state=42):
    import numpy as np
    rng = np.random.RandomState(random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_list, y_list, rep_idx_list = [], [], []

    for c in classes:
        idx_c = np.where(y == c)[0]       # chỉ số mẫu gốc thuộc lớp c
        Xc = X[idx_c]
        yc = y[idx_c]
        need = max_count - len(idx_c)

        if need > 0:
            add_src = rng.choice(idx_c, size=need, replace=True)
            X_add = X[add_src] + rng.normal(0, noise_std, size=(need, X.shape[1]))
            y_add = np.full(need, c, dtype=y.dtype)

            X_bal_c = np.vstack([Xc, X_add])
            y_bal_c = np.concatenate([yc, y_add])
            rep_idx_c = np.concatenate([idx_c, add_src])
        else:
            X_bal_c = Xc
            y_bal_c = yc
            rep_idx_c = idx_c

        X_list.append(X_bal_c)
        y_list.append(y_bal_c)
        rep_idx_list.append(rep_idx_c)

    X_bal = np.vstack(X_list)
    y_bal = np.concatenate(y_list)
    rep_idx = np.concatenate(rep_idx_list).astype(int)  # KHÔNG có -1

    return X_bal, y_bal, rep_idx


def ham_generate_paths(feature_dir, feature_types, categories):
    """
    Tạo dictionary paths[category][feature_type] = đường dẫn thư mục
    """
    import os
    paths = {}
    for category in categories:
        paths[category] = {}
        for feature_type in feature_types:
            feature_path = os.path.join(feature_dir, feature_type, category)
            if os.path.exists(feature_path) and os.path.isdir(feature_path):
                paths[category][feature_type] = feature_path
    # Nếu có tập test không nhãn
    unlabeled_category = 'unlabeled'
    paths[unlabeled_category] = {}
    for feature_type in feature_types:
        unlabeled_path = os.path.join(feature_dir, feature_type, unlabeled_category)
        if os.path.exists(unlabeled_path) and os.path.isdir(unlabeled_path):
            paths[unlabeled_category][feature_type] = unlabeled_path
    return paths

def generate_paths(feature_dir, dataset_type, feature_types, categories):
    """
    Discover existing feature folders for a given dataset split ('train' or 'test').
    Returns a nested dict: paths[category][feature_type] -> directory path
    """
    paths = {}
    for cat in categories:
        paths[cat] = {}
        for ft in feature_types:
            p = os.path.join(feature_dir, dataset_type, ft, cat)
            if os.path.isdir(p):
                paths[cat][ft] = p
    if dataset_type == 'test':
        unl = 'unlabeled'
        paths[unl] = {}
        for ft in feature_types:
            p = os.path.join(feature_dir, dataset_type, ft, unl)
            if os.path.isdir(p):
                paths[unl][ft] = p
    return paths
