# -*- coding: utf-8 -*-
# config.py — Configuration for Multi-Modal GNN
# type: ignore
import os
import numpy as np
import torch
import random

# ====== Reproducibility ======
SEED = 42
def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====== Training hyperparameters ======
BATCH_SIZE_LIST = [32]
EPOCH_VALUES = [300]
HIDDEN_DIM = 64
DROPOUT = 0.4
K_NEIGHBORS = 5

# ====== HAM SETTING ======
Code_Dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = "OUTPUT"
PROJECT_DIR = os.path.join(Code_Dir, PROJECT_NAME)
os.makedirs(PROJECT_DIR, exist_ok=True)
# =============== Categories ==============
CATEGORIES =['VASC', 'DF', 'BKL', 'AKIEC', 'BCC', 'NV', 'MEL']
DB_PREFIX = "ham"
# =============== Features ==============
# # Top features (từ chi-square ranking trong bài báo)
FEATURE_TYPES_DEEP = ["densenet121_features", "densenet169_features", "densenet201_features"]
# FEATURE_TYPES_HAND = ["densenet121_features"]
FEATURE_TYPES_HAND = ["lbp_features", "color_histograms_features", "hsv_histograms_features"]

# Đường dẫn metadata và đặc trưng môi trường HPC
FEATURE_DIR = "/data2/cmdir/home/hieuvv/IMAGES/data2/data2_HAM10000_FEATURES"
TRAIN_METADATA_PATH = os.path.join(FEATURE_DIR, "HAM10000_metadata.csv")

# dữ liệu thử nghiệm nhỏ trên máy tính cá nhân
# FEATURE_DIR = r"E:\Backup\01_KetQuaNghienCuu\14-Multi-modal GNN_for Skin Lesion Classification Using Dermoscopic Images\14_code\test_HAM10000"
# TRAIN_METADATA_PATH = os.path.join(FEATURE_DIR, "HAM10000_metadata.csv")

# ====== Baselines definition ======
# Model configurations
def get_baselines(prefix=DB_PREFIX):
    """
    - Gating/Cross-attention: deep + clinical (2 modality)
    - Adaptive: deep + hand + clinical (3 modality)
    - Ours: 'multi_modal_GNN_tri_adaptive_gcn' = adaptive + GCN + LGC=True
    """
    return [
        {"model_name": "GFC", "desc": "GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using only clinical features (age, sex, localization). No deep or handcrafted features. No fusion, no LGC.", "use_deep_feature": False, "use_handcraft_feature": False, "use_clinical_feature": True, "gnn_type": "gcn", "fusion_type": "none", "use_lgc": False},
        {"model_name": "GFH", "desc": "GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using only handcrafted features (color histograms, fractal, hsv histograms). Features are fused by concatenation. No deep or clinical features included.", "use_deep_feature": False, "use_handcraft_feature": True,  "use_clinical_feature": False, "gnn_type": "gcn",  "fusion_type": "concat",          "use_lgc": False},
        {"model_name": "GFD", "desc": "GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using only deep features (VGG19, DenseNet121, MobileNet). Features are fused by concatenation. No handcrafted or clinical features included.", "use_deep_feature": True,  "use_handcraft_feature": False, "use_clinical_feature": False, "gnn_type": "gcn",  "fusion_type": "concat",          "use_lgc": False},
        {"model_name": "GFDH", "desc": "GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using deep and handcrafted features, fused by concatenation. No clinical features included.", "use_deep_feature": True,  "use_handcraft_feature": True,  "use_clinical_feature": False, "gnn_type": "gcn",  "fusion_type": "concat",          "use_lgc": False},
        {"model_name": "GFDCG", "desc": "GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using deep and clinical features. Fusion is performed by gating mechanism (gating fusion) between modalities.", "use_deep_feature": True,  "use_handcraft_feature": False, "use_clinical_feature": True,  "gnn_type": "gcn",  "fusion_type": "gating",          "use_lgc": False},
        {"model_name": "GFDCX", "desc": "GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using deep and clinical features. Fusion is performed by cross-attention mechanism between modalities.", "use_deep_feature": True,  "use_handcraft_feature": False, "use_clinical_feature": True,  "gnn_type": "gcn",  "fusion_type": "cross_attention", "use_lgc": False},
        {"model_name": "GFDHCA", "desc": "GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using deep, handcrafted, and clinical features. Fusion is performed by adaptive fusion (dynamic weighting of modalities).", "use_deep_feature": True,  "use_handcraft_feature": True,  "use_clinical_feature": True,  "gnn_type": "gcn",  "fusion_type": "adaptive",        "use_lgc": False},
        {"model_name": "GSFDHC", "desc": "GraphSAGE (2 layers + FC, hidden_dim=64, dropout=0.4) using deep, handcrafted, and clinical features. Features are fused by concatenation.", "use_deep_feature": True,  "use_handcraft_feature": True,  "use_clinical_feature": True,  "gnn_type": "sage", "fusion_type": "concat",          "use_lgc": False},
        {"model_name": "GAFDHC", "desc": "GAT (2 layers + FC, hidden_dim=64, heads=4, dropout=0.4) using deep, handcrafted, and clinical features. Features are fused by concatenation.", "use_deep_feature": True,  "use_handcraft_feature": True,  "use_clinical_feature": True,  "gnn_type": "gat",  "fusion_type": "concat",          "use_lgc": False},
        {"model_name": "GIFDHC", "desc": "GIN (2 layers + FC, hidden_dim=64, dropout=0.4) using deep, handcrafted, and clinical features. Features are fused by concatenation.", "use_deep_feature": True,  "use_handcraft_feature": True,  "use_clinical_feature": True,  "gnn_type": "gin",  "fusion_type": "concat",          "use_lgc": False},
        {"model_name": "GFDC", "desc": "GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using deep and clinical features, fused by concatenation. No handcrafted features included.", "use_deep_feature": True,  "use_handcraft_feature": False, "use_clinical_feature": True,  "gnn_type": "gcn",  "fusion_type": "concat",          "use_lgc": False},
        {"model_name": "GFDHCM", "desc": "GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using deep, handcrafted, and clinical features. Features are fused by concatenation. No adaptive fusion, no LGC.", "use_deep_feature": True, "use_handcraft_feature": True, "use_clinical_feature": True, "gnn_type": "gcn", "fusion_type": "concat", "use_lgc": False},
        {"model_name": "GFHCM", "desc": "GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using handcrafted and clinical features, fused by concatenation. No deep features included.", "use_deep_feature": False, "use_handcraft_feature": True,  "use_clinical_feature": True,  "gnn_type": "gcn",  "fusion_type": "concat",          "use_lgc": False},
        {"model_name": "GFDHCA-LGC", "desc": "GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using deep, handcrafted, and clinical features. Fusion is performed by adaptive fusion and includes Local Graph Convolution (LGC) for enhanced local structure learning.", "use_deep_feature": True,  "use_handcraft_feature": True,  "use_clinical_feature": True,  "gnn_type": "gcn",  "fusion_type": "adaptive",        "use_lgc": True},
    ]
    # ...existing code...
