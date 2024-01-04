# Learning Structural Dynamics with Graph Neural Network



## Introduction

Graph Neural Networks (GNNs) / Message Passing Neural Networks  (MPNNs) have shown impressive performance in various tasks ranging from  molecule generation to computer graphics or even physics-based  simulations. On the other hand, operations on structural meshes, such as condensation, are well established in the FEM community. Yet, the  relationship between MPNNs and these structural operations remains  largely unexplored. This thesis aims to formalize the equivalence  between the message-passing scheme and operations on FEM matrices.

**Keywords**: graphs, geometric deep learning, structures, message passing



## Description

This thesis aims to formalize the equivalence between message-passing neural networks and operations on structural matrices. The proposed research  will contribute to the understanding of the underlying mechanisms of  MPNNs and help develop more efficient and interpretable MPNNs as  alternatives to structural simulators.

Desired competencies: Solid knowledge of Python and PyTorch. Experience and/or good knowledge about ML and structures (FEM).



## Goal

This thesis aims to formalize the equivalence between message passing neural networks and operations on structural matrices. Specifically, the  student will have the following objectives:

- Review the existing methods and literature.
- Become familiar with the PyTorch Geometric framework.
- Implement a first prototype, based on existing work from the literature.
- Develop a theoretical framework for formalizing the equivalence between MPNNs and structural operations.
- Formulate methods for sub-structuring/condensation schemes which exploit this framework.
- Demonstrate the method on a simple case study, exploring its advantages and limitations.


## Usage 

train SIGN
```bash 
cd src 
python main.py -c config/hyperparameter/sign.toml
```




## Involvements

| Person        | Role | Organization                                 |
| ------------- | ---- | -------------------------------------------- |
| Duthé Gregory | Host | Structural Mechanics (Prof. Chatzi) (ETHZ)   |
| Duthé Gregory | Host | ETH Competence Center - ETH AI Center (ETHZ) |

