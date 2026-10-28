# RPSGH: Reliability-Guided Prototype Semantic Graph Hashing

**MATLAB implementation for unsupervised domain adaptive image retrieval.**

RPSGH transfers category-level knowledge from a labeled source domain to an unlabeled target domain and learns compact binary codes for efficient retrieval. The framework combines prototype-enhanced pseudo-labeling, reliability-guided alignment, and semantic graph-based hash learning.

## Overview

RPSGH consists of three components:

1. **Prototype-enhanced pseudo-labeling.** Orthogonality-constrained class prototypes refine labeled source class centers. The refined centers initialize target-domain clustering, and the resulting source-semantic and target-geometric class probabilities are fused to assign target pseudo-labels.
2. **Reliability-guided alignment.** Neighborhood reliability weights cross-domain local alignment, while sample reliability weights prototype alignment to reduce the influence of uncertain associations and representations.
3. **Unified semantic graph hashing.** A graph combines source-domain supervision, source–target semantic associations, and target-domain neighborhood structure. A semantic-guided scale factor adjusts the overall strength of the target–target relations before asymmetric similarity alignment in Hamming space.

The implementation uses **pre-extracted features**; it does not train a deep feature extractor.

## Requirements

- MATLAB (the minimum supported release has not been established).
- **Statistics and Machine Learning Toolbox** (`kmeans`, `pdist2`, and `dummyvar`).
- Sufficient memory for the dataset; the implementation constructs dense pairwise matrices.

Run the scripts from the repository root, because dataset loading relies on relative paths. Compatibility with GNU Octave has not been verified.

## Repository Structure

```text
RPSGH/
├── README.md
├── RPSGH_Demo.m          # Experiment entry point
├── RPSGH_main.m          # Training, query encoding, and MAP evaluation
├── construct_dataset.m  # Dataset loading and preprocessing
├── improve_P.m          # Shared projection initialization
├── feature_confidence.m # Sample reliability estimation
├── Init_S_Structure.m   # Unified graph construction
├── normalize1.m         # Feature normalization
├── compactbit.m         # Binary-code packing
├── hammingDist.m        # Hamming distance computation
├── ...                  # Additional MATLAB utilities
└── data/                 # Locally prepared datasets 
```

Auxiliary scripts not used by the main training path are also present in the original project. Ensure all functions called by `RPSGH_main.m` and `construct_dataset.m` are available on the MATLAB path.

## Datasets and Protocol

The dataset loader supports adaptation tasks from **MNIST–USPS**, **COIL20**, **Office-31**, **Office-Home**, and other cross-dataset image retrieval benchmarks implemented in `construct_dataset.m`.

Prepare the corresponding pre-extracted feature files under `data/`, preserving the relative paths and variable names expected by `construct_dataset.m`. Representative directories include:

```text
data/
├── MNIST_vs_USPS.mat
├── USPS_vs_MNIST.mat
├── COIL_1.mat
├── COIL_2.mat
├── Office31/
├── Office-Home(vgg)/
├── cross-dataset/
└── VLSC/
```

**Dataset files are not provided in this README.** Obtain them from their original distributors and follow the applicable licenses. The supported task identifiers and required `.mat` fields are defined in `construct_dataset.m`.

For each task, the source domain provides labeled training samples, and the target domain is divided into training and query subsets. The example below uses a **class-stratified 10% target query split**. Target training labels are not passed into the model updates. However, the dataset preparation code uses target labels for stratified splitting, determining the class count, and constructing retrieval relevance ground truth. This distinction should be preserved when reporting the experimental protocol.

Features are L2-normalized per sample and then centered using the joint mean of the source and target **training** features.

## Quick Start

Set MATLAB's current folder to the project root and run:

```matlab
clear; clc;
addpath(pwd);
addpath(genpath(fullfile(pwd, 'data')));

rng(0, 'twister');

param = struct();
param.maxIter  = 30;
param.k        = 25;
param.bit      = 64;
param.lambda1  = 1;
param.lambda2  = 10;
param.beta     = 0.1;
param.lambda_h = 0.1;

DB_name  = 'MNIST-USPS';
test_num = 0.10;

dataset = construct_dataset(DB_name, test_num, param);
[single_MAP, cross_MAP] = RPSGH_main(dataset, param);

fprintf('%s | %d bits | Single-MAP: %.2f%% | Cross-MAP: %.2f%%\n', ...
    DB_name, param.bit, single_MAP, cross_MAP);
```

The output MAP values are already expressed as **percentages**; do not multiply them by 100 again.

The included `RPSGH_Demo.m` may also be run from the repository root. In the version described by the supplied project documentation, its active sections perform a 64-bit single-run `MNIST-USPS` experiment followed by ten 64-bit `D-A` runs; the multi-bit benchmark loop is not enabled by default. Review the demo settings before treating its output as a reproduction of every table in the paper.

## Parameters

The following values match the sample configuration in the current project documentation; they are **example settings**, not automatic defaults of `RPSGH_main.m`.

| Parameter | Example | Description |
| --- | ---: | --- |
| `maxIter` | 30 | Number of outer optimization iterations |
| `k` | 25 | Neighborhood size for cross-domain alignment and target graph construction |
| `bit` | 64 | Binary code length and projection dimension |
| `lambda1` | 1 | Cross-domain neighborhood alignment weight |
| `lambda2` | 10 | Reliability-related code alignment weight |
| `beta` | 0.1 | Semantic graph alignment weight |
| `lambda_h` | 0.1 | Regularization for the out-of-sample ridge hash mapping |

The projection update also uses an internal regularization coefficient, `gamma = 5`, in the documented implementation. To change it, inspect the main function rather than adding an unused configuration field.

## Retrieval Evaluation

Target-domain test samples serve as queries in both settings:

| Setting | Query set | Retrieval database |
| --- | --- | --- |
| **Cross-domain retrieval (Cross-MAP)** | Target test samples | Labeled source training samples |
| **Single-domain retrieval (Single-MAP)** | Target test samples | Unlabeled target training samples |

The final training codes are divided into source and target database codes. A ridge-regression mapping from centered original training features to learned binary codes encodes unseen queries. Retrieval ranks all database items by Hamming distance; MAP is computed from class-label relevance and reported on a 0–100 scale. The reported metric is full-database MAP, not MAP@K.

For multi-bit studies, configure `param.bit` using `[16, 32, 48, 64, 96, 128]` and repeat experiments with documented random seeds and split settings. The current demo does not automatically save complete multi-task, multi-bit results.

## Reproducibility Notes

- Use the **exact code revision, dataset feature files, train/test splits, and evaluation procedure** associated with the results reported in the paper.
- Set and report random seeds for both target-query splitting and model initialization. Fixing seeds does not guarantee identical floating-point results across MATLAB releases or hardware.
- Confirm that the main MATLAB function name matches its filename (`RPSGH_main.m`) and that all auxiliary functions resolve correctly.
- **Verify the Hamming-distance lookup before publishing benchmark results.** The implementation described in the supplied documentation uses a `uint8` XOR result as a lookup index; adding one without converting to a wider type can mishandle a byte value of `255`. Check the index operation (e.g., `double(y) + 1`), test the eight-opposite-bits case, and recompute evaluation metrics if a correction is needed.
- The provided documentation has not established an end-to-end execution test for this release. No unverified benchmark numbers or MATLAB version claims are included here.

## Citation

If you use this implementation, please cite the corresponding RPSGH paper when its bibliographic information becomes available. The paper title, author list, and publication details will be added after confirmation.

## License and Data Availability

No open-source license has been specified for this release. Confirm permission from all code owners before adding one. Dataset usage and redistribution are governed by the respective dataset providers' terms.
