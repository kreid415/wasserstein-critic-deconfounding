# Wall time for the Tier 1+2 rerun, scib backbone (regenerate: python scripts/wall_time_report.py --backbone scib)

Measured on the local RTX 3080 (scvi-tools 1.4.2, scIB scVI backbone (n_latent 30, 2 layers, NB, all cells), batch 128): ms per training step per arm (docs/throughput_rtx3080.csv); 8 concurrent lanes: 4.02x aggregate (minimum identical-arm factor) (barycenter 4.68x, pooled 4.02x); scIB-native scoring 16.2 ms per cell per config, single thread (docs/scoring_time.csv).

| lambda design | uncond seeds | barycenter iterations | fits | GPU lane-h | critic share | scoring CPU-h | local wall (8 lanes) | conservative |
|---|---|---|---|---|---|---|---|---|
| pilot | 5 | 10 | 5,989 | 2,381 | 75% | 779 | 24.7 d | 24.7 d |
| pilot | 5 | 5 | 5,989 | 2,266 | 74% | 779 | 23.5 d | 23.5 d |
| pilot | 5 | 3 | 5,989 | 2,220 | 73% | 779 | 23.0 d | 23.0 d |
| pilot | 3 | 10 | 5,397 | 2,104 | 75% | 677 | 21.8 d | 21.8 d |
| pilot | 3 | 5 | 5,397 | 2,008 | 73% | 677 | 20.8 d | 20.8 d |
| pilot | 3 | 3 | 5,397 | 1,969 | 73% | 677 | 20.4 d | 20.4 d |
| shared | 5 | 10 | 6,829 | 2,881 | 76% | 1,020 | 29.9 d | 29.9 d |
| shared | 5 | 5 | 6,829 | 2,726 | 74% | 1,020 | 28.3 d | 28.3 d |
| shared | 5 | 3 | 6,829 | 2,664 | 74% | 1,020 | 27.6 d | 27.6 d |
| shared | 3 | 10 | 5,853 | 2,420 | 75% | 851 | 25.1 d | 25.1 d |
| shared | 3 | 5 | 5,853 | 2,296 | 74% | 851 | 23.8 d | 23.8 d |
| shared | 3 | 3 | 5,853 | 2,246 | 73% | 851 | 23.3 d | 23.3 d |

ms/step (min-max over tasks): barycenter 49.4-72.0; barycenter_warm3 42.2-46.1; discriminator 11.2-11.8; discriminator_sn 14.7-14.7; mmd 12.0-12.0; mmd_ref 12.2-12.2; none 7.2-7.9; pooled 15.6-44.2; pooled_sn 40.1-40.1; reference 42.9-42.9; reference_fixed 33.6-33.6; scanvi 7.6-7.6; scvi_adv 10.9-10.9; sinkhorn 16.9-19.3; sysvi 15.8-15.8

Assumptions: fit cost = steps x measured ms/step (+5 s setup); steps = epochs x ceil(n x train_size / batch); the 8-lane factor measured on immune is applied to every arm and task; scoring runs on CPU cores not used by the fit lanes (its effect on fit throughput is not measured).
