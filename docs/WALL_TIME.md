# Wall time for the Tier 1+2 rerun, stock backbone (regenerate: python scripts/wall_time_report.py --backbone stock)

Measured on the local RTX 3080 (scvi-tools 1.4.2, scvi-tools default backbone (n_latent 10, 1 layer, ZINB, 90% train), batch 128): ms per training step per arm (docs/throughput_rtx3080_stock_backbone.csv); 8 concurrent lanes: 3.77x aggregate on a mixed-arm queue of 64 fits (docs/concurrency_calibration_stock.json); conservative column uses the minimum identical-arm factor 3.17x (barycenter_warm3 3.57x, discriminator 3.56x, none 3.26x, pooled 3.17x); scIB-native scoring 16.2 ms per cell per config, single thread (docs/scoring_time.csv).

| lambda design | uncond seeds | barycenter iterations | fits | GPU lane-h | critic share | scoring CPU-h | local wall (8 lanes) | conservative |
|---|---|---|---|---|---|---|---|---|
| pilot | 5 | 10 | 6,718 | 2,375 | 74% | 833 | 26.2 d | 31.2 d |
| pilot | 5 | 5 | 6,718 | 2,268 | 73% | 833 | 25.1 d | 29.8 d |
| pilot | 5 | 3 | 6,718 | 2,225 | 73% | 833 | 24.6 d | 29.2 d |
| pilot | 3 | 10 | 6,126 | 2,123 | 74% | 731 | 23.5 d | 27.9 d |
| pilot | 3 | 5 | 6,126 | 2,033 | 73% | 731 | 22.5 d | 26.7 d |
| pilot | 3 | 3 | 6,126 | 1,997 | 73% | 731 | 22.1 d | 26.3 d |
| shared | 5 | 10 | 7,396 | 2,792 | 75% | 1,058 | 30.9 d | 36.7 d |
| shared | 5 | 5 | 7,396 | 2,646 | 74% | 1,058 | 29.2 d | 34.8 d |
| shared | 5 | 3 | 7,396 | 2,588 | 73% | 1,058 | 28.6 d | 34.0 d |
| shared | 3 | 10 | 6,420 | 2,373 | 75% | 889 | 26.2 d | 31.2 d |
| shared | 3 | 5 | 6,420 | 2,256 | 73% | 889 | 24.9 d | 29.7 d |
| shared | 3 | 3 | 6,420 | 2,210 | 73% | 889 | 24.4 d | 29.0 d |

ms/step (min-max over tasks): barycenter 50.0-72.9; barycenter_iter5 46.8-46.8; barycenter_warm3 41.2-45.3; discriminator 11.6-11.6; discriminator_iw 13.1-13.1; discriminator_r1 14.6-14.6; discriminator_ref 16.5-16.5; discriminator_sn 16.1-16.1; mmd 12.7-12.7; mmd_iw 12.4-12.4; mmd_ref 11.4-11.4; none 8.3-8.3; pooled 17.8-46.9; pooled_iw 36.9-36.9; pooled_sn 39.6-39.6; pooled_stratified 30.5-30.5; reference 47.0-47.0; reference_fixed 39.4-39.4; reference_iw 50.2-50.2; reference_stratified 46.4-46.4; scanvi 7.8-7.8; scvi_adv 11.4-11.4; sinkhorn 20.6-20.6; sysvi 12.7-12.7

Assumptions: fit cost = steps x measured ms/step (+5 s setup); steps = epochs x ceil(n x train_size / batch); the 8-lane factor measured on immune is applied to every arm and task; scoring runs on CPU cores not used by the fit lanes (its effect on fit throughput is not measured).

Provisional: the ms/step of 8 arms (discriminator_r1, discriminator_ref, reference_stratified, pooled_stratified, discriminator_iw, reference_iw, pooled_iw, mmd_iw) are run1 medians of experiments/bench_missing_arms_step_cost, whose timing check PF-16 failed under CPU contention (lab notebook NB-20261002-07, -08). In the pilot/5/10 design they carry 513 fits and 159 of 2,375 GPU lane-h (7%); columns provisional_fits and provisional_lane_hours of docs/wall_time_designs.csv give every design.
