# VGAE for Biological Networks

This repo contains an end-to-end implementation of a **Variational Graph Autoencoder (VGAE)** for **link prediction** on biological interaction networks. It is tailored for square interaction matrices (e.g., protein-protein, gene-gene, or cell-cell networks) and evaluates the model's ability to infer missing or potential edges.

---

## Overview

- **Input:** A square `n×n` numeric interaction matrix
  - Represents presence (or count) of interactions between biological entities
  - Interpreted as an undirected graph: entries > 0 imply an edge between node *i* and node *j*
  - Format: `.xlsx` file (Excel), optionally with row/column labels (auto-detected and stripped)
  - Diagonal entries must be numeric but are not used in edge construction
- **Output:**
  - `embeddings.csv` — node embeddings learned via VGAE
  - `embeddings_umap.png` — UMAP-based 2D projection of the learned embeddings
  - `link_prob(i, j)` — estimated probability of interaction between any two nodes based on their embeddings
- **Evaluation:**
  - Link prediction performance evaluated using ROC-AUC and Average Precision (AP)
  - Dataset is split using random link sampling (train/validation/test)

---

## Model Assumptions

- **Graph Assumptions:**
  - Input matrix must be square and fully numeric
  - Treated as an undirected graph (symmetric adjacency assumed)
  - Only binary edge presence is used (any value > 0 is treated as a link)
- **Node Features:**
  - Uses an identity matrix (`I_n`) as node features (i.e., each node has a one-hot input vector)
- **Learning Objective:**
  - VGAE reconstructs the observed edges and infers likely missing ones
  - The training is unsupervised; no class or label annotations are needed
- **Evaluation Metrics:**
  - ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)
  - AP (Average Precision)

## Input Data Format

| Requirement          | Description                                      |
|----------------------|--------------------------------------------------|
| File format          | Excel `.xlsx`                                    |
| Matrix shape         | Must be square (`n×n`)                           |
| Data type            | All values must be numeric (floats or integers) |
| Labels (optional)    | Auto-removed if present in top row and column    |
| Symmetry             | Not enforced, but assumed for undirected graphs |
| Edge interpretation  | Matrix entry > 0 implies an edge                |

## Use Cases

- Any biological network represented by a symmetric interaction matrix

## Output Files

- **`embeddings.csv`**: Node embeddings with shape `n × latent_dim` (default: 32)
- **`embeddings_umap.png`**: 2D projection using UMAP for visualization
- **KMeans clustering**: Included in the script for basic unsupervised grouping
- **`link_prob(i, j)`**: Callable function to estimate the probability of an edge between node `i` and node `j`

## Notes

- Designed for small to moderately sized graphs (typically up to a few thousand nodes)
- Larger matrices may require GPU acceleration and sufficient memory
- Not suitable for highly sparse datasets with extremely low edge density without modification
