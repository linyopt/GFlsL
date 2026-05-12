# TBFL: Threshold Block Fused Lasso (GGM)

MATLAB implementation of the **Threshold Block Fused Lasso (TBFL)** algorithm
for change point detection in **Gaussian Graphical Models (GGM)**.

## Reference

This implementation is based on:

```bibtex
@Article{bai2024unified,
  author          = {Bai, Yue and Safikhani, Abolfazl},
  title           = {A Unified Framework for Change Point Detection in
                  High-Dimensional Linear Models},
  journal         = {Statistica Sinica},
  year            = 2024,
  publisher       = {Statistica Sinica (Institute of Statistical Science)}
}
```

The paper notes the algorithm is implemented in the R package
`LinearDetect`.

## Algorithm Overview

TBFL is a three-step procedure:

1. **Step I (Block fused lasso / neighborhood selection)**:
   fit block-wise regressions for each node in a GGM using fused penalties.
2. **Step II (Hard-thresholding + BIC/HBIC recursion)**:
   screen block-end candidates by clustering jump sizes into "small" vs
   "large" groups and accepting blocks while BIC/HBIC improves.
3. **Step III (Block clustering + local exhaustive search)**:
   cluster candidates using a Gap-statistic style rule and refine each
   cluster by exhaustive local SSE search.

## Code Structure

| File | Description |
|------|-------------|
| `TBFL.m` | Main class definition, constructor, and public properties |
| `run.m` | Entry point: block-size selection (HBIC), CV for lambdas, Step II/III |
| `BuildBlocks.m` | Construct boundary-aware block partition (LinearDetect style) |
| `ComputeBlockSizeGrid.m` | Compute the default block-size grid (LinearDetect) |
| `ComputeLambda1Grid.m` | Compute lambda1 grid (warm-up / lambda-max) |
| `SelectLambdasForBlocks.m` | CV over `(lambda1, lambda2)` grid for GGM |
| `SolveNeighborhoodSelection.m` | Step I loop over nodes (GGM) |
| `SolveFusedLassoNode.m` | Step I solver for one node (LinearDetect C++ port) |
| `HardThreshold.m` | Step II: jump selection + candidate clustering |
| `LocalRefinement.m` | Step III: local exhaustive refinement |
| `ComputeGOF.m` | Segment goodness-of-fit helper (used by legacy code paths) |
| `ComputeHBIC.m` | Legacy HBIC helper (not used by the current `run.m`) |
| `SelectBlockSize.m` | Legacy block-size selection (not used by current `run.m`) |

## Usage

```matlab
addpath(genpath(pwd));

% Example: synthetic GGM data (see repo utilities).
T = 200; p = 10; m = 1; prob = 0.8; Nsim = 1;
[X, ~, ~, trueBreaks] = DGP_iid(T, p, m, prob, Nsim);

% Run TBFL (auto-selects block size and penalties).
Est = TBFL(X=X, disp_freq=inf);
Est.run();

cp_hat = Est.breaks;        % Final change points after Step III
candidates = Est.candidates; % Step II candidates (before refinement)
```

### Fix Block Size / Provide Custom Lambda Grids

```matlab
Est = TBFL(X=X, block_size=5, lambda1_cv=[], lambda2_cv=[], disp_freq=inf);
Est.run();
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `block_size` | Block size (scalar). If empty, select via HBIC. | `[]` |
| `block_size_grid` | Grid for HBIC selection when `block_size=[]`. | Auto |
| `lambda1_cv` | Lambda1 grid for CV. If empty, computed from data. | `[]` |
| `lambda2_cv` | Lambda2 grid for CV. | `[10,1,0.1]*sqrt(2*log(p)/T)` |
| `gamma_val` | HBIC strength for block-size selection. | `1.5` |
| `HBIC_step2` | Use HBIC (vs BIC) inside Step II recursion. | `true` |
| `gamma_step2` | HBIC gamma used in Step II (LinearDetect default is 1). | `1` |
| `tol` | Step I solver tolerance. | `1e-2` |
| `maxiter` | Step I solver maximum iterations. | `100` |
| `disp_freq` | Display frequency; `inf` disables prints. | `inf` |

## Outputs

After `Est.run()`, the main result fields are:

- `Est.breaks`: final estimated change points (Step III)
- `Est.candidates`: candidate change points after Step II
- `Est.cp_first`: clustered candidates (cell array; Step III input)
- `Est.selected_block_size`, `Est.selected_lambda1`, `Est.selected_lambda2`:
  selected tuning parameters
- `Est.phi_hat_full`: Step I increment estimates (LinearDetect: `phi.hat.full`)
- `Est.beta_full`: Step I cumulative block coefficients (LinearDetect: `beta.full`)
- `Est.beta_hat_list`: Step III segment coefficients (LinearDetect: `beta.hat.list`)

### Precision Matrix Estimation (Omega)

TBFL is implemented via neighborhood selection. After running TBFL you can
obtain a segment-wise precision matrix estimate Omega using:

```matlab
[Omega_hat, info] = Est.EstimateOmega();
Omega_1 = Omega_hat(:, :, 1);
```

`Omega_hat` is a `p-by-p-by-m` array where `m = length(Est.breaks) + 1`.
The struct `info` includes the segment boundaries and the estimated
residual variances used to construct the diagonal of Omega.
