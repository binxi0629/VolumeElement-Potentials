# Volume-Element-Project
This repo aims to study on total potential energy fitting with area elements (of BN) and volume elements (of GaN). Different from conventional Chemical bonding model, we propose a new approach that starts from volume element as a basis, which also incroporates interactions with nearby volume elements (MORE THEORY TO COME).

---
# Data Generation

See [README.md](generation#readme)

---

# Label and features generation process
Includes:
- symmetry functions of structures
- energy labels of structures
- volume elements vertices coordinates
  - method 1: NN search on all structures.
  - method 2: NN search on one reference structure and remember the indices of NNs, then map the indices for all other structures and compute vertices directly. Requires reference structure having same file format as all other data.
- pca features

See tom_datagen_demo.ipynb for instructions and details.

The csv files are not provided here, please ask me for it if you want my generated files.

# NN Modelling
Includes:
- Parrinello's symmetry functions model
- Volume elements PCA model
- Volume elements coordinates model

See tom_model_demo.ipynb for instructions and details
