# VGAE for Biological Networks

This repo contains an end-to-end implementation of a **Variational Graph Autoencoder (VGAE)** for **link prediction** on biological interaction networks. It is tailored for square interaction matrices (e.g., protein-protein, gene-gene, or cell-cell networks) and evaluates the model's ability to infer missing or potential edges.

---

## 🔍 Overview

- Input: `60×60` interaction matrix (e.g., adjacency matrix from experiments or databases)
- Output:
  - **Learned node embeddings** (`embeddings.csv`)
  - **UMAP visualization** (`embeddings_umap.png`)
  - **Link probability estimator** between any two nodes
- Evaluates using **ROC-AUC** and **Average Precision (AP)**
- Unsupervised: no labels required beyond the adjacency matrix
