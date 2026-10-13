## Validation at the committed constants (3076 minibatches; cap reached: 0)

```
                         minibatches  branch_cos_min  branch_norm_ratio_min  branch_norm_ratio_max  branch_value_err_median  iters_median  iters_max  tag_cos_median  tag_norm_ratio_median
regime                                                                                                                                                                                     
A1 zstd=0                       2160               1                      1                      1                3.352e-08             7        110          0.9963                      2
A1_ext_r1_round1 zstd=0          216               1                 0.9999                      1                4.317e-06             4          7          0.2813                  14.44
A2A3 zstd=1                      432               1                 0.9999                      1                2.495e-08         210.5       5273          0.8685                  2.273
early zstd=0                     160               1                      1                      1                1.886e-08             8        290           0.982                  1.995
early zstd=1                      32               1                      1                      1                1.646e-08            21         55          0.8943                   2.27
trainmode zstd=0                  60               1                      1                      1                8.082e-07           5.5        115          0.9658                  2.005
trainmode zstd=1                  16               1                      1                      1                 2.16e-08           426       1360          0.8483                   2.33
```

## Tolerance sweep (cap 200,000)

```
               minibatches  rel_grad_err_max  iters_median  iters_p99  iters_max
zstd tol                                                                        
0    0.000001          118          1.82e-06         132.5      450.7        480
     0.000010          118         1.853e-05           102      358.2        385
     0.000100          118         0.0001874            70      265.7        290
     0.001000          118          0.002226            42      172.5        195
1    0.000001          480         6.042e-06           334       3636     171388
     0.000010          480         6.042e-05           274       2551      29190
     0.000100          480          0.000749         200.5       1482       5273
     0.001000          480          0.007179           141       1077       4426
```

## GPU ms per loss forward + backward, median (RTX 3080, one process)

```
                   tag_ms  new_ms
regime                           
A1_l10              8.103   12.14
A1_shrunk_l3000     4.011   6.736
A1_wide_l0.1        11.69   19.47
A3_zstd_l30         12.64   117.2
A3_zstd_l3_immune   12.76   138.7
```
