# Wall time for the Tier 1+2 rerun (regenerate: python scripts/wall_time_report.py)

Measured on the local RTX 3080 (scvi-tools 1.4.2, scIB scVI backbone, batch 128): ms per training step per arm (docs/throughput_rtx3080.csv); 8 concurrent lanes give 4.02x aggregate (minimum of the measured pooled 4.02x, barycenter 4.68x); scIB-native scoring 16.2 ms per cell per config, single thread (docs/scoring_time.csv).

| lambda design | uncond seeds | barycenter | fits | GPU lane-h | critic share | scoring CPU-h | local wall (8 lanes) |
|---|---|---|---|---|---|---|---|
| pilot | 5 | warm | 5,989 | 2,215 | 74% | 779 | 23.0 d |
| pilot | 5 | cold | 5,989 | 2,376 | 75% | 779 | 24.6 d |
| pilot | 3 | warm | 5,397 | 1,965 | 73% | 677 | 20.4 d |
| pilot | 3 | cold | 5,397 | 2,099 | 75% | 677 | 21.8 d |
| shared | 5 | warm | 6,829 | 2,659 | 74% | 1,020 | 27.6 d |
| shared | 5 | cold | 6,829 | 2,876 | 76% | 1,020 | 29.8 d |
| shared | 3 | warm | 5,853 | 2,242 | 74% | 851 | 23.2 d |
| shared | 3 | cold | 5,853 | 2,416 | 76% | 851 | 25.0 d |

ms/step (min-max over tasks): barycenter 49.4-72.0; barycenter_warm3 42.2-46.1; discriminator 11.2-11.8; discriminator_sn 14.7-14.7; mmd 12.0-12.0; mmd_ref 12.2-12.2; none 7.2-7.9; pooled 15.6-44.2; pooled_sn 40.1-40.1; reference 42.9-42.9; reference_fixed 33.6-33.6; scanvi 7.6-7.6; scvi_adv 10.9-10.9; sinkhorn 16.9-19.3; sysvi 15.8-15.8

Assumptions: fit cost = steps x measured ms/step (+5 s setup); steps = epochs x ceil(n x train_size / batch); the 8-lane factor measured on immune is applied to every arm and task; scoring runs on CPU cores not used by the fit lanes (its effect on fit throughput is not measured).
