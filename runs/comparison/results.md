| test run | method | step trans err mean [cm] | median [cm] | ATE w/ GT rot [m] | ATE [% path] |
|---|---|---|---|---|---|
| 12_21_2020_ec_hallways_run4 | zero | 23.9 | 24.9 | 11.51 | 11.2 |
| 12_21_2020_ec_hallways_run4 | constant | 13.1 | 12.8 | 4.52 | 4.4 |
| 12_21_2020_ec_hallways_run4 | CNN (stacked pair 3D CNN) | 11.1 | 11.5 | 4.73 | 4.6 |
| 12_21_2020_ec_hallways_run4 | Doppler least squares (no learning) | 8.8 | 8.4 | 2.75 | 2.7 |
| 12_21_2020_ec_hallways_run4 | learned Doppler (proposed) | 8.5 | 8.2 | 2.45 | 2.4 |
| 12_21_2020_arpg_lab_run4 | zero | 13.8 | 14.3 | 8.13 | 8.5 |
| 12_21_2020_arpg_lab_run4 | constant | 16.7 | 16.4 | 13.29 | 14.0 |
| 12_21_2020_arpg_lab_run4 | CNN (stacked pair 3D CNN) | 8.7 | 8.4 | 2.67 | 2.8 |
| 12_21_2020_arpg_lab_run4 | Doppler least squares (no learning) | 7.5 | 6.9 | 2.76 | 2.9 |
| 12_21_2020_arpg_lab_run4 | learned Doppler (proposed) | 7.2 | 6.8 | 2.57 | 2.7 |

| test run | rotation err per step [deg]: CNN | zero | constant | CNN own-rotation ATE [m] |
|---|---|---|---|---|
| 12_21_2020_ec_hallways_run4 | 3.69 | 3.67 | 3.68 | 14.42 |
| 12_21_2020_arpg_lab_run4 | 4.05 | 3.74 | 3.74 | 5.87 |
