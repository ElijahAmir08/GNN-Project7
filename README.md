# Symmetry-Consistent GNN Explanations for Graph Rigidity

**Course:** CMPSC 292F — Machine Learning on Graphs  
**University:** University of California, Santa Barbara  
**Team:** Elijah-Amir Hodges & Salaar Saraj  
**Final Deadline:** December 11, 2026  

## Project Overview

Graph Neural Networks (GNNs) can produce consistent predictions on equivalent graph representations while their explanation methods may assign different importance scores to structurally equivalent edges.

This project investigates whether making GNN explanations respect existing graph symmetries improves explanation consistency without sacrificing predictive faithfulness.

We focus on **generic two-dimensional bar-and-joint rigidity**, where nodes represent joints and edges represent fixed-length bars.

Our prediction task is to determine whether adding a proposed edge introduces an independent constraint or a redundant (dependent) constraint.

For redundant edges, **rigidity circuits** provide mathematically verifiable dependency certificates. These allow us to compare GNN explanations against known structural dependencies while separately evaluating predictive faithfulness.

## Research Goals

Our primary research question is:

**Can enforcing graph symmetry improve explanation consistency and agreement with valid rigidity dependencies without sacrificing predictive faithfulness?**

We aim to:

1. Train a Graph Neural Network, initially a Graph Isomorphism Network (GIN), to classify proposed edges as independent or redundant.
2. Generate explanations using GNNExplainer and PGExplainer.
3. Compare three explanation strategies:
   - **Raw Explanations:** Original edge-importance scores.
   - **Orbit Averaging:** Average importance scores among edges equivalent under task-preserving graph automorphisms.
   - **Orbit-Consistent Selection:** Select complete groups of symmetry-equivalent edges.
4. Evaluate explanation consistency under node relabeling and graph automorphisms.
5. Measure predictive faithfulness, explanation stability, and agreement with rigidity circuits.

## Development Progress

### November 2, 2026: Rigidity Labels and Circuit Verification

Implemented and tested an initial rigidity-labeling pipeline using PyRigi.

- Set up Python environment.
- Implement independence/redundancy checks.
- Verify an independent proposed edge.
- Verify a redundant proposed edge.
- Identify a valid K4 rigidity circuit.
- Pass all three automated tests.



## Installation and Setup

### Requirements

- Python 3.11 (recommended)
- Git
- VS Code or another Python-compatible IDE

Additional dependencies will be added as development progresses.

### Windows (PowerShell)

Clone the repository or open your existing local project folder in VS Code.

**1. Create a virtual environment:**

```powershell
py -3.11 -m venv .venv
```

**2. Install dependencies:**

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data Representation

Each rigidity example currently contains:

| Field | Description |
|---|---|
| `num_nodes` | Number of joints |
| `existing_edges` | Bars already present in the framework |
| `candidate_edge` | Proposed edge being classified |
| `label` | `0` = independent, `1` = redundant |
| `circuit_edges` | Verified rigidity circuit for redundant examples; `null` otherwise |

The candidate edge is stored separately from the existing edges.

The initial implementation assumes that existing edges form an independent set. Under this assumption, adding a redundant edge produces a unique fundamental rigidity circuit.

Future datasets may include more complex graphs with multiple possible circuits.


## References

### Software and Datasets

- [PyRigi Documentation](https://pyrigi.github.io/PyRigi/)
- [PyRigi Graph API](https://pyrigi.github.io/PyRigi/userguide/api/graph.html)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [NetworkX](https://networkx.org/documentation/stable/)
- [Laman Graph Dataset — Zenodo](https://zenodo.org/records/1245517)

### Research Papers

- Ying et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks.*
- Luo et al. (2020). *Parameterized Explainer for Graph Neural Network.*
- Xu et al. (2019). *How Powerful Are Graph Neural Networks?*
- Jacobs & Hendrickson (1997). *An Algorithm for Two-Dimensional Rigidity Percolation: The Pebble Game.*
