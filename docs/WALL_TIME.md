# Wall time for the Tier 1+2 rerun, stock backbone (regenerate: python scripts/wall_time_report.py --backbone stock)

Measured on the local RTX 3080 (scvi-tools 1.4.2, scvi-tools default backbone (n_latent 10, 1 layer, ZINB, 90% train), batch 128): ms per training step per arm (docs/throughput_rtx3080_stock_backbone.csv); 8 concurrent lanes: 3.77x aggregate on a mixed-arm queue of 64 fits (docs/concurrency_calibration_stock.json); conservative column uses the minimum identical-arm factor 3.17x (barycenter_warm3 3.57x, discriminator 3.56x, none 3.26x, pooled 3.17x); scIB-native scoring 16.2 ms per cell per config, single thread (docs/scoring_time.csv).

| lambda design | uncond seeds | barycenter iterations | fits | GPU lane-h | critic share | scoring CPU-h | local wall (8 lanes) | conservative |
|---|---|---|---|---|---|---|---|---|
| pilot | 5 | 10 | 6,930 | 2,375 | 74% | 872 | 26.2 d | 31.2 d |
| pilot | 5 | 5 | 6,930 | 2,268 | 73% | 872 | 25.1 d | 29.8 d |
| pilot | 5 | 3 | 6,930 | 2,225 | 72% | 872 | 24.6 d | 29.2 d |
| pilot | 3 | 10 | 6,338 | 2,123 | 74% | 770 | 23.5 d | 27.9 d |
| pilot | 3 | 5 | 6,338 | 2,033 | 73% | 770 | 22.5 d | 26.7 d |
| pilot | 3 | 3 | 6,338 | 1,997 | 72% | 770 | 22.1 d | 26.3 d |
| shared | 5 | 10 | 7,608 | 2,792 | 75% | 1,097 | 30.9 d | 36.7 d |
| shared | 5 | 5 | 7,608 | 2,646 | 73% | 1,097 | 29.2 d | 34.8 d |
| shared | 5 | 3 | 7,608 | 2,588 | 73% | 1,097 | 28.6 d | 34.0 d |
| shared | 3 | 10 | 6,632 | 2,373 | 75% | 928 | 26.2 d | 31.2 d |
| shared | 3 | 5 | 6,632 | 2,256 | 73% | 928 | 24.9 d | 29.7 d |
| shared | 3 | 3 | 6,632 | 2,210 | 73% | 928 | 24.4 d | 29.0 d |

ms/step (min-max over tasks): barycenter 50.0-72.9; barycenter_iter5 46.8-46.8; barycenter_warm3 41.2-45.3; discriminator 11.6-11.6; discriminator_iw 13.1-13.1; discriminator_r1 13.6-13.6; discriminator_ref 15.5-15.5; discriminator_sn 16.1-16.1; mmd 12.7-12.7; mmd_iw 13.1-13.1; mmd_ref 11.4-11.4; none 8.3-8.3; pooled 17.8-46.9; pooled_iw 37.6-37.6; pooled_sn 39.6-39.6; pooled_stratified 41.7-41.7; reference 47.0-47.0; reference_fixed 39.4-39.4; reference_iw 46.0-46.0; reference_stratified 46.0-46.0; scanvi 7.8-7.8; scvi_adv 11.4-11.4; sinkhorn 20.6-20.6; sysvi 12.7-12.7

Assumptions: fit cost = steps x measured ms/step (+5 s setup); steps = epochs x ceil(n x train_size / batch); the 8-lane factor measured on immune is applied to every arm and task; scoring runs on CPU cores not used by the fit lanes (its effect on fit throughput is not measured).

Provisional: the ms/step of 1 arm(s) (mmd_iw) are run2 medians of experiments/bench_missing_arms_step_cost whose repeats failed the timing check PF-16 (spread > 25%; lab notebook NB-20261003-07, -10). In the pilot/5/10 design they carry 72 fits and 11 of 2,375 GPU lane-h (0%); columns provisional_fits and provisional_lane_hours of docs/wall_time_designs.csv give every design.
