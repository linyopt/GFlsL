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

For further details on the problem and the algorithm, please refer to our paper:

```bibtex
@article{LP2026,
      title={Change-point detection in variance-covariance matrix},
      author={Ying Lin and Benjamin Poignard},
      year={2026},
      eprint={2605.12881},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      doi={10.48550/arXiv.2605.12881},
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
- `data/`: the bundled SP100 panel used by the real-data script: daily closing
  prices of the 94 constituents, the S&P 100 index level and the trading
  calendar, in `data/SP100.xls`
- `results/`: default output location for experiment logs and saved results

## Running Experiments

All entry points follow the `runcode_*.m` naming convention. Common ones:

- `runcode_simu.m`: main simulation pipeline across DGPs and tuning grids
- `runcode_realdata.m`: real-data estimation and break diagnostics
- `runcode_sensitivity.m`: sensitivity analysis for adaptive exponents
- `runcode_computational_complexity.m`: empirical computational complexity analysis

Typical MATLAB setup:

```matlab
addpath(genpath(pwd));

% Pick one of the runcodes:
runcode_simu
% runcode_realdata
```

## Notes

- `utils/boundedline/` is a vendored subset of the `boundedline` package by
  Kelly Kearney (<https://github.com/kakearney/boundedline-pkg/>), distributed
  under its own MIT license (`utils/boundedline/LICENSE.txt`).
  Only the four functions the computation-time figures need are kept —
  `boundedline.m`, `singlepatch.m`, `catuneven.m` and the `Inpaint_nans`
  dependency of John D'Errico, whose license sits in
  `utils/boundedline/Inpaint_nans/license.txt`. The other files the package
  README describes, such as `outlinebounds.m` and the demonstrations, are not
  shipped here.
- Result files are written under `results/` by default.
- Due to numerical sensitivity, **TBFL** may produce slightly different results
  across different machines. TBFL tends to over-segment, and the subsequent
  per-segment covariance estimation frequently involves eigendecomposition of
  ill-conditioned matrices, which can vary across hardware and software
  environments.

## License

The source code of this project is released under the MIT License (see
[`LICENSE`](LICENSE)).

That license does not extend to the market data bundled in
`data/SP100.xls`. Those are third-party data obtained from Yahoo Finance
(<https://finance.yahoo.com>) and are included solely so that the real-data
experiment can be reproduced. They remain subject to the terms of their
original provider; reuse and redistribution of the data must comply with
those terms. See [`LICENSE`](LICENSE) for the full notice.
