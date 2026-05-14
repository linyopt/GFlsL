# Change-point detection in variance-covariance matrix

This repository provides MATLAB code for **change-point recovery** in **variance-covariance matrices**
using **GFlsL** (Group Fused least squares LASSO). The estimators solve the optimization problem

```math
\min_{\substack{\Theta_t \succeq \epsilon I_p, \\ 1 \leq t \leq T}} \left\{ \sum_{t=1}^T \frac{1}{2T} \| X_t X_t^{\top} - \Theta_t \|_F^2 + \lambda_1\sum_{t=1}^T \sum_{u \neq v} \xi_{uv, 1t} | \Theta_{uv, t} | + \lambda_2 \sum_{t=2}^T \xi_{2t} \| \Theta_t - \Theta_{t-1} \|_F \right\}.
```

It contains:

- The main estimator from the paper: **GFlsL** (non-adaptive and adaptive versions)
- Three competing methods (TBFL, DCDP, BSOP/WBSIP)
- Simulation utilities (`simu/`) and experiment scripts (`runcode_*.m`)

For further details on the problem and the algorithm, please refer to [our paper](https://arxiv.org/abs/2605.12881):

```bibtex
@article{LP2026,
      title={Change-point detection in variance-covariance matrix}, 
      author={Ying Lin, Benjamin Poignard},
      year={2026},
      eprint={2605.12881},
      archivePrefix={arXiv},
      primaryClass={math.ME},
      url={https://arxiv.org/abs/2605.12881},
}
```


## Requirements

1. MATLAB R2023a or newer is recommended to run the experiment scripts (`runcode_*.m`).

2. Some workflows use the Parallel Computing Toolbox (`parfor`,
   `parallel.pool.DataQueue`) for speed, especially in simulation runs.
   The code can still be read without it, but full experiments may depend
   on it.

## Project Layout (Implementations)

Main method:

- `@GFlsL/`: **GFlsL** (Group Fused least squares LASSO), the main estimator studied in the paper.

Competitors:

- `@TBFL/`: **TBFL** (Threshold Block Fused Lasso), see `@TBFL/readme.md`.
- `@DCDP/`: **DCDP** (Divide-and-Conquer Dynamic Programming).
- `BS/`: **BSOP / WBSIP** covariance change-point baselines.

Utilities / data:

- `simu/`: synthetic data generation for iid, banded, and factor-model settings
- `utils/`: helper functions for evaluation, reporting, and plotting
- `utils/post_covariance/`: post-segmentation covariance refits for baseline break detectors
- `data/`: bundled input datasets used by the real-data scripts
- `results/`: default output location for experiment logs and saved results

## Running Experiments

All entry points follow the `runcode_*.m` naming convention. Common ones:

- `runcode_simu.m`: main simulation pipeline across DGPs and tuning grids
- `runcode_realdata.m`: real-data estimation and portfolio evaluation
- `runcode_sensitivity.m`: sensitivity analysis for adaptive exponents
- `runcode_computational_complexity.m`: empirical computational complexity analysis

Typical MATLAB setup:

```matlab
addpath(genpath(pwd));

% Pick one of the runcodes:
runcode_simu
% runcode_realdata
```

## Supplementary Documentation

Detailed analysis and discussion from the paper:

- [Sensitivity to Adaptive Weight Exponents](docs/sensitivity_adaptive_weights.md) — sensitivity of the adaptive estimator to $\mu_1$ and $\mu_2$ across all three DGPs
- [Empirical Computational Complexity](docs/empirical_computational_complexity.md) — empirical computational complexity with respect to tuning parameters, $T$, and $p$

## Notes

- The repository includes `utils/boundedline/` as a vendored external
  plotting package.
- Result files are written under `results/` by default.
- Due to numerical sensitivity, **TBFL** may produce slightly different results
  across different machines. TBFL tends to over-segment, and the subsequent
  per-segment covariance estimation frequently involves eigendecomposition of
  ill-conditioned matrices, which can vary across hardware and software
  environments.

## License

This project is released under the MIT License (see [`LICENSE`](LICENSE)).
