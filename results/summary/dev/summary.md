# Summary: dog

Run names: A = per-frame (A@<n> = stopped after n iterations), B = warm start, BL_lam<l>_K<K> = propagation without flow, C_lam<l>_K<K> = flow propagation, BF = warm start + flow (exploratory).

Ratios to per-frame drawing (A = 1). Flicker: lower is smoother. LPIPS lower and CLIP higher is better. Mean over (clip, seed) units, 95% bootstrap CI in brackets when there are 2 or more units.

Main flicker metric: chamfer_eval.

| Run | n | Flicker: chamfer | E_warp | E_warp per ink | E_warp with RAFT | LPIPS | CLIP | Iters/frame | Drift slope (LPIPS gap per frame since keyframe) | Fidelity limits met |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2001 | nan | fidelity ok |
| A@1000 | 1 | 1.192 | 0.978 | 1.039 | 0.990 | 1.004 | 0.991 | 1000 | nan | fidelity ok |
| A@150 | 1 | 2.008 | 0.836 | 1.302 | 0.829 | 1.080 | 0.942 | 150 | nan |  |
| A@1500 | 1 | 1.104 | 1.007 | 1.042 | 1.011 | 1.003 | 0.994 | 1500 | nan | fidelity ok |
| A@300 | 1 | 1.602 | 0.931 | 1.194 | 0.928 | 1.043 | 0.966 | 300 | nan | fidelity ok |
| A@500 | 1 | 1.397 | 0.979 | 1.134 | 0.978 | 1.022 | 0.985 | 500 | nan | fidelity ok |
| A@750 | 1 | 1.272 | 1.011 | 1.096 | 1.020 | 1.012 | 0.990 | 750 | nan | fidelity ok |
| B | 1 | 1.021 | 1.125 | 1.047 | 1.125 | 1.059 | 0.967 | 150 | 0.0009 | fidelity ok |
| BF | 1 | 0.886 | 1.139 | 1.008 | 1.135 | 1.143 | 0.987 | 150 | 0.0011 |  |
| BL_lam0.0001_K6 | 1 | 1.046 | 0.931 | 1.014 | 0.939 | 1.068 | 0.970 | 196 | 0.0028 | fidelity ok |
| BL_lam0.0004_K6 | 1 | 1.215 | 1.070 | 1.146 | 1.077 | 1.122 | 0.965 | 196 | 0.0065 |  |
| BL_lam0.0016_K6 | 1 | 2.061 | 1.141 | 1.345 | 1.139 | 1.384 | 0.948 | 196 | 0.0281 |  |
| BL_lam0.0064_K6 | 1 | 2.256 | 1.215 | 1.385 | 1.201 | 1.476 | 0.947 | 196 | 0.0340 |  |
| BL_lam0_K12 | 1 | 1.051 | 1.067 | 1.064 | 1.055 | 1.073 | 0.960 | 165 | 0.0041 | fidelity ok |
| BL_lam0_K6 | 1 | 1.035 | 0.906 | 0.990 | 0.914 | 1.032 | 0.978 | 196 | 0.0022 | fidelity ok |
| BL_lam0_Kinf | 1 | 1.109 | 1.111 | 1.107 | 1.097 | 1.102 | 0.957 | 150 | 0.0018 |  |
| C_lam0.0001_K6 | 1 | 0.958 | 0.938 | 0.984 | 0.943 | 1.023 | 0.982 | 196 | 0.0016 | fidelity ok |
| C_lam0.0001_K6_curve | 1 | 0.775 | 1.014 | 0.958 | 1.011 | 1.048 | 0.966 | 196 | 0.0021 | fidelity ok |
| C_lam0.0004_K6 | 1 | 0.752 | 0.935 | 0.934 | 0.940 | 1.064 | 0.946 | 196 | 0.0011 |  |
| C_lam0.0004_K6_curve | 1 | 0.670 | 0.842 | 0.881 | 0.857 | 1.057 | 0.964 | 196 | 0.0026 | fidelity ok |
| C_lam0.0016_K12 | 1 | 0.637 | 0.727 | 0.829 | 0.728 | 1.118 | 0.968 | 165 | 0.0038 |  |
| C_lam0.0016_K6 | 1 | 0.637 | 0.788 | 0.836 | 0.789 | 1.070 | 0.969 | 196 | 0.0035 | fidelity ok |
| C_lam0.0016_K6_curve | 1 | 0.614 | 0.851 | 0.798 | 0.839 | 1.199 | 0.967 | 196 | 0.0048 |  |
| C_lam0.0016_Kinf | 1 | 0.509 | 0.745 | 0.821 | 0.764 | 1.143 | 0.949 | 150 | 0.0026 |  |
| C_lam0.0064_K6 | 1 | 0.683 | 0.627 | 0.731 | 0.630 | 1.137 | 0.949 | 196 | 0.0093 |  |
| C_lam0.0064_K6_curve | 1 | 0.347 | 0.717 | 0.598 | 0.715 | 1.149 | 0.932 | 196 | 0.0042 |  |
| C_lam0_K6 | 1 | 0.871 | 1.027 | 0.985 | 1.030 | 1.038 | 0.971 | 196 | 0.0020 | fidelity ok |
| C_lam0_K6_curve | 1 | 0.882 | 1.020 | 0.999 | 1.019 | 1.055 | 0.973 | 196 | 0.0000 | fidelity ok |

## Paired comparisons (by clip and seed): flow propagation vs propagation without flow at equal settings, then other pairs

| Pair (X vs Y) | Chamfer flicker change (X / Y - 1) | E_warp change | LPIPS ratio difference (X - Y) |
| --- | --- | --- | --- |
| C_lam0.0001_K6 vs BL_lam0.0001_K6 | -0.084 | 0.007 | -0.044 |
| C_lam0.0004_K6 vs BL_lam0.0004_K6 | -0.381 | -0.126 | -0.058 |
| C_lam0.0016_K6 vs BL_lam0.0016_K6 | -0.691 | -0.309 | -0.314 |
| C_lam0.0064_K6 vs BL_lam0.0064_K6 | -0.697 | -0.484 | -0.339 |
| C_lam0_K6 vs BL_lam0_K6 | -0.159 | 0.133 | 0.007 |
| C_lam0.0016_K6 vs BL_lam0_K6 | -0.385 | -0.131 | 0.038 |
| C_lam0.0016_K6 vs B | -0.376 | -0.300 | 0.010 |
| BF vs B | -0.132 | 0.012 | 0.084 |
| BF vs C_lam0.0016_K6 | 0.391 | 0.445 | 0.073 |
| BF vs BL_lam0_K6 | -0.144 | 0.256 | 0.111 |

Selection rule applied to these clips (used only on the development clip; the settings are in configs/settings.yaml): flow propagation = `C_lam0.0016_K6`, without flow = `BL_lam0_K6`

Per-frame seed-to-seed chamfer (px), the same frame drawn twice: {"dog": null}
