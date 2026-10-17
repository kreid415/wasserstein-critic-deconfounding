## Regression check: float64 re-run vs committed validation_12mb.csv

23 shared columns, 3076 rows; columns that differ: none (bit-identical)

## Accuracy against the converged float64 reference (SINKHORN_TOL 1e-4, cap 500,000)

```
                     minibatches  cap_hits  worst_cosine  1-worst_cosine  worst_rel_grad_err  norm_ratio_min  norm_ratio_max  rows_ref_gt_1e-6  worst_rel_value_err  worst_abs_value_err_ref_le_1e-6  iters_median  iters_max
598 hardest float64          598         0             1       2.789e-07            0.000749          0.9999               1               598            3.079e-06                              NaN           154       5273
            float32          598         0             1       2.788e-07           0.0007489          0.9999               1               598            2.974e-06                              NaN           154       5273
all 3,076   float64         3076         0             1       2.789e-07            0.000749          0.9999               1              3064            6.667e-05                        2.541e-16             7       5273
            float32         3076         0             1       2.788e-07           0.0007489          0.9999               1              3064             0.003168                        8.885e-09             7       5273
```

SI-58 accuracy criterion met by float32: 598 hardest: yes, all 3,076: NO

## Worst relative value error by regime (|reference| > 1e-6)

```
                          n  f32_value_err_max  f64_value_err_max
regime           zstd                                            
A1               0     2160           0.000269           1.55e-05
A1_ext_r1_round1 0      204            0.00317           6.67e-05
A2A3             1      432           2.97e-06           3.08e-06
early            0      160           1.07e-06           5.72e-08
                 1       32           1.96e-07           4.97e-08
trainmode        0       60           0.000183           1.37e-05
                 1       16           3.64e-07           2.56e-07
```

## GPU ms per loss forward + backward, median of 20 calls x 12 minibatches (RTX 3080, alone)

```
                   tag_ms  float64_ms  float32_ms  f32_over_f64_median  f32_over_f64_min  gpu_cos_32_64_min  gpu_value_rel_32_64_absmax
regime                                                                                                                                 
A1_l10               7.84       11.63       11.81                1.003            0.9992                  1                   2.079e-07
A1_shrunk_l3000     3.989       6.734       6.667               0.9919            0.9183                  1                    3.14e-05
A1_wide_l0.1        11.81       19.74       19.46                0.989            0.8778                  1                   9.486e-08
A3_zstd_l30         12.68       117.8       117.1                1.014            0.9612                  1                   1.771e-07
A3_zstd_l3_immune   12.73       142.7       140.3                1.006             0.953                  1                   8.543e-08
```
