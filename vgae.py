#!/usr/bin/env python3

"""
End-to-end VGAE for interaction matrix
"""

# imports
import sys, warnings
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

# silence
warnings.filterwarnings(
    "ignore",
    message="Signature .* for <class 'numpy.longdouble'> does not match any known type",
)

# hyper-params
LATENT = 32      # embedding size
EPOCHS = 200
LR = 0.005
VAL_RATIO = 0.10
TEST_RATIO = 0.10


#  data loader
def read_matrix(xlsx_path: Path) -> torch.Tensor:
    df = pd.read_excel(xlsx_path, header=None)
    # auto-detect & strip label row/col
    if isinstance(df.iat[0, 0], str):
        df = df.iloc[1:, 1:]  # drop top-left labels
    # final sanity check
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Need a 60×60 numeric block, got {df.shape}")
    df = df.apply(pd.to_numeric, errors="raise")  # abort on non-numbers
    return torch.tensor(df.values, dtype=torch.float32)


# model definition 
class Encoder(torch.nn.Module):
    def __init__(self, num_nodes: int):
        super().__init__()
        self.conv_mu  = GCNConv(num_nodes, LATENT)
        self.conv_log = GCNConv(num_nodes, LATENT)
    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_log(x, edge_index)



# helpers
@torch.no_grad()
def evaluate(z, pos_edge_index, num_nodes):
    neg_edge_index = negative_sampling(
        pos_edge_index, num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1),
        method="sparse",
    )
    def probs(edge_ix):
        logits = (z[edge_ix[0]] * z[edge_ix[1]]).sum(dim=-1)
        return torch.sigmoid(logits).cpu().numpy()
    pos, neg = probs(pos_edge_index), probs(neg_edge_index)
    y_true  = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    y_score = np.concatenate([pos,               neg])
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)

# main 
def main(xlsx_file: str):
    A = read_matrix(Path(xlsx_file))
    num_nodes = A.size(0)
    # build PyG Data object
    edge_index = (A > 0).nonzero(as_tuple=False).t().contiguous()
    x = torch.eye(num_nodes)
    data = Data(x=x, edge_index=edge_index)
    # train/val/test split
    splitter = RandomLinkSplit(
        num_val=VAL_RATIO, num_test=TEST_RATIO, is_undirected=True)
    train_data, val_data, test_data = splitter(data)
    # model
    model = VGAE(Encoder(num_nodes))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.edge_index) + model.kl_loss() / num_nodes
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(train_data.x, train_data.edge_index)
                val_auc, val_ap = evaluate(z, val_data.edge_index, num_nodes)
            print(f"Epoch {epoch:03d}  loss {loss:.4f}  val AUC {val_auc:.3f}  AP {val_ap:.3f}")
    # final test score
    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)
        test_auc, test_ap = evaluate(z, test_data.edge_index, num_nodes)
    print("\n===== FINAL METRICS =====")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test AP : {test_ap:.4f}")
    # save embeddings
    pd.DataFrame(z.cpu().numpy()).to_csv("embeddings.csv", index=False)
    print("Embeddings written to embeddings.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage:  python vgae.py  data.xlsx")
    main(sys.argv[1])
import pandas as pd, umap
import matplotlib.pyplot as plt
z = pd.read_csv("embeddings.csv").values
coords = umap.UMAP(n_neighbors=10, min_dist=0.3).fit_transform(z)
plt.scatter(coords[:,0], coords[:,1]);

plt.scatter(coords[:, 0], coords[:, 1])
plt.tight_layout()
plt.savefig("embeddings_umap.png",
            dpi=300,
            bbox_inches="tight")


from sklearn.cluster import KMeans
labels = KMeans(n_clusters=4, random_state=0).fit_predict(z)

import numpy as np
z = np.loadtxt("embeddings.csv", delimiter=",")

def link_prob(i: int, j: int) -> float:
    """Return model-estimated interaction probability between nodes i and j."""
    logit = np.dot(z[i], z[j])          # inner product of their embeddings
    return 1 / (1 + np.exp(-logit))     # sigmoid

# example: probability that node 3 interacts with node 17
print(link_prob(3, 17))
