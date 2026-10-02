# Wall time for the Tier 1+2 rerun, stock backbone (regenerate: python scripts/wall_time_report.py --backbone stock)

Measured on the local RTX 3080 (scvi-tools 1.4.2, scvi-tools default backbone (n_latent 10, 1 layer, ZINB, 90% train), batch 128): ms per training step per arm (docs/throughput_rtx3080_stock_backbone.csv); 8 concurrent lanes: 3.77x aggregate on a mixed-arm queue of 64 fits (docs/concurrency_calibration_stock.json); conservative column uses the minimum identical-arm factor 3.17x (barycenter_warm3 3.57x, discriminator 3.56x, none 3.26x, pooled 3.17x); scIB-native scoring 16.2 ms per cell per config, single thread (docs/scoring_time.csv).

| lambda design | uncond seeds | barycenter iterations | fits | GPU lane-h | critic share | scoring CPU-h | local wall (8 lanes) | conservative |
|---|---|---|---|---|---|---|---|---|
| pilot | 5 | 10 | 5,989 | 2,152 | 74% | 779 | 23.8 d | 28.3 d |
| pilot | 5 | 5 | 5,989 | 2,045 | 73% | 779 | 22.6 d | 26.9 d |
| pilot | 5 | 3 | 5,989 | 2,002 | 72% | 779 | 22.1 d | 26.3 d |
| pilot | 3 | 10 | 5,397 | 1,899 | 74% | 677 | 21.0 d | 25.0 d |
| pilot | 3 | 5 | 5,397 | 1,810 | 73% | 677 | 20.0 d | 23.8 d |
| pilot | 3 | 3 | 5,397 | 1,774 | 72% | 677 | 19.6 d | 23.3 d |
| shared | 5 | 10 | 6,829 | 2,618 | 75% | 1,020 | 28.9 d | 34.4 d |
| shared | 5 | 5 | 6,829 | 2,473 | 73% | 1,020 | 27.3 d | 32.5 d |
| shared | 5 | 3 | 6,829 | 2,414 | 73% | 1,020 | 26.7 d | 31.7 d |
| shared | 3 | 10 | 5,853 | 2,199 | 75% | 851 | 24.3 d | 28.9 d |
| shared | 3 | 5 | 5,853 | 2,082 | 73% | 851 | 23.0 d | 27.4 d |
| shared | 3 | 3 | 5,853 | 2,036 | 73% | 851 | 22.5 d | 26.8 d |

ms/step (min-max over tasks): barycenter 50.0-72.9; barycenter_iter5 46.8-46.8; barycenter_warm3 41.2-45.3; discriminator 11.6-11.6; discriminator_sn 16.1-16.1; mmd 12.7-12.7; mmd_ref 11.4-11.4; none 8.3-8.3; pooled 17.8-46.9; pooled_sn 39.6-39.6; reference 47.0-47.0; reference_fixed 39.4-39.4; scanvi 7.8-7.8; scvi_adv 11.4-11.4; sinkhorn 20.6-20.6; sysvi 12.7-12.7

Assumptions: fit cost = steps x measured ms/step (+5 s setup); steps = epochs x ceil(n x train_size / batch); the 8-lane factor measured on immune is applied to every arm and task; scoring runs on CPU cores not used by the fit lanes (its effect on fit throughput is not measured).
