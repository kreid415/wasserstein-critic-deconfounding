# Wall time for the Tier 1+2 rerun, stock backbone (regenerate: python scripts/wall_time_report.py --backbone stock)

Measured on the local RTX 3080 (scvi-tools 1.4.2, scvi-tools default backbone (n_latent 10, 1 layer, ZINB, 90% train), batch 128): ms per training step per arm (docs/throughput_rtx3080_stock_backbone.csv); 8 concurrent lanes: 3.77x aggregate on a mixed-arm queue of 64 fits (docs/concurrency_calibration_stock.json); conservative column uses the minimum identical-arm factor 3.17x (barycenter_warm3 3.57x, discriminator 3.56x, none 3.26x, pooled 3.17x); scIB-native scoring 16.2 ms per cell per config, single thread (docs/scoring_time.csv).

| lambda design | uncond seeds | barycenter iterations | fits | GPU lane-h | critic share | scoring CPU-h | local wall (8 lanes) | conservative |
|---|---|---|---|---|---|---|---|---|
| pilot | 5 | 10 | 6,061 | 2,169 | 74% | 786 | 24.0 d | 28.5 d |
| pilot | 5 | 5 | 6,061 | 2,062 | 73% | 786 | 22.8 d | 27.1 d |
| pilot | 5 | 3 | 6,061 | 2,019 | 72% | 786 | 22.3 d | 26.5 d |
| pilot | 3 | 10 | 5,469 | 1,916 | 74% | 683 | 21.2 d | 25.2 d |
| pilot | 3 | 5 | 5,469 | 1,827 | 73% | 683 | 20.2 d | 24.0 d |
| pilot | 3 | 3 | 5,469 | 1,791 | 72% | 683 | 19.8 d | 23.5 d |
| shared | 5 | 10 | 6,829 | 2,618 | 75% | 1,020 | 28.9 d | 34.4 d |
| shared | 5 | 5 | 6,829 | 2,473 | 73% | 1,020 | 27.3 d | 32.5 d |
| shared | 5 | 3 | 6,829 | 2,414 | 73% | 1,020 | 26.7 d | 31.7 d |
| shared | 3 | 10 | 5,853 | 2,199 | 75% | 851 | 24.3 d | 28.9 d |
| shared | 3 | 5 | 5,853 | 2,082 | 73% | 851 | 23.0 d | 27.4 d |
| shared | 3 | 3 | 5,853 | 2,036 | 73% | 851 | 22.5 d | 26.8 d |

ms/step (min-max over tasks): barycenter 50.0-72.9; barycenter_iter5 46.8-46.8; barycenter_warm3 41.2-45.3; discriminator 11.6-11.6; discriminator_sn 16.1-16.1; mmd 12.7-12.7; mmd_ref 11.4-11.4; none 8.3-8.3; pooled 17.8-46.9; pooled_sn 39.6-39.6; reference 47.0-47.0; reference_fixed 39.4-39.4; scanvi 7.8-7.8; scvi_adv 11.4-11.4; sinkhorn 20.6-20.6; sysvi 12.7-12.7

Assumptions: fit cost = steps x measured ms/step (+5 s setup); steps = epochs x ceil(n x train_size / batch); the 8-lane factor measured on immune is applied to every arm and task; scoring runs on CPU cores not used by the fit lanes (its effect on fit throughput is not measured).
