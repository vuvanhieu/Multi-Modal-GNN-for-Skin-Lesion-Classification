# Multi-Modal GNN for Skin Lesion Classification (HAM10000)

This project implements multi-modal Graph Neural Networks (GNNs) for skin lesion classification using the HAM10000 dataset. It supports various feature types (deep, handcrafted, clinical) and fusion strategies (concat, gating, cross-attention, adaptive fusion).

## Features
- **Deep features**: Extracted from CNNs (DenseNet121, DenseNet169, DenseNet201)
- **Handcrafted features**: Color histograms, LBP, HSV histograms
- **Clinical features**: Age, sex, localization
- **Fusion methods**: Concatenation, gating, cross-attention, adaptive fusion
- **GNN architectures**: GCN, GraphSAGE, GAT, GIN
- **Data balancing**: Augmentation to balance class distribution

## Project Structure
```
├── config.py         # Configuration and baseline definitions
├── data.py           # Data loading, preprocessing, augmentation
├── model.py          # GNN model definitions and fusion modules
├── train.py          # Training and evaluation utilities
├── eval.py           # Evaluation and plotting functions
├── main.py           # Main runner for experiments and aggregation
├── HAM10000_metadata.csv # Metadata for HAM10000
```

## Usage
1. **Configure paths and baselines** in `config.py`.
2. **Prepare features** in the specified directories.
3. **Run experiments**:
   ```bash
   python main.py
   ```
4. **Results** and summary metrics will be saved in the project directory.

## Requirements
- Python 3.7+
- PyTorch
- torch_geometric
- scikit-learn
- pandas
- numpy

## Citation
If you use this code, please cite the original HAM10000 dataset and relevant papers.

## Contact
For questions or contributions, please contact the project maintainer.
