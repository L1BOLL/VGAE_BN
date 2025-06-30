## Quick Start Tutorial

### 1. Setup

Clone the repository or download the two files:

- `vgae.py` — the main script
- `requirements.txt` — list of required Python packages

Place your input file in the same folder:

- `data.xlsx` — your square numeric interaction matrix

### 2. Install Dependencies

Run this in your terminal:

```bash
pip install -r requirements.txt
```
If you run into issues with torch-geometric, follow the instructions at:
https://pytorch-geometric.com/quick-start/

### 3. Run the Model

In the same folder as the files, run:

```bash
python vgae.py data.xlsx
```

This will:
- Train a VGAE model on your interaction matrix
- Output evaluation metrics
- Generate:
  - embeddings.csv — node embeddings
  - embeddings_umap.png — 2D visualization of the embeddings

### 4. Estimate Link Probability

To estimate the link probability between any two nodes after training:
Write in you terminal:
```bash
python
```
Then enter:
```python
import numpy as np
z = np.loadtxt("embeddings.csv", delimiter=",")

def link_prob(i, j):
    logit = np.dot(z[i], z[j])
    return 1 / (1 + np.exp(-logit))

print(link_prob(3, 17))  # change 3 and 17 to the 2 nodes you want
```
 You now have a working VGAE pipeline for link prediction on your data.

 9te3
``
