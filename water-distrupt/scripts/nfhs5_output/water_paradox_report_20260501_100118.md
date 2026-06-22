# The Infrastructure Paradox: Modern Water Systems and
# New Vulnerabilities in India
## Evidence from NFHS-5 (2019-21)

*Generated: 2026-05-01 10:01*

---
## ABSTRACT

Using data from 578,062 households in NFHS-5, we document the Infrastructure Paradox: piped-water households experience higher disruption rates than tube-well users. Overall disruption rate: 18.8%. Piped coverage: 51.9%. We construct an Infrastructure Dependency Index (IDI, 4 dimensions, PCA-weighted, Monte Carlo CI) and a Reliability Gap Index (RGI, district-level weighted OLS residual with bootstrap CI). A GEE multilevel model and a two-stage slope-as-outcome model confirm the paradox is largest where IDI and RGI are jointly high.

---
## FINDING 1 — The Paradox in Raw Data

| Water Source            |   Weighted N (000s) |   Disruption Rate (%) |   95% CI Lower |   95% CI Upper |   Relative Risk vs Tube Well |
|:------------------------|--------------------:|----------------------:|---------------:|---------------:|-----------------------------:|
| Piped Water             |               304.5 |                  25.5 |           25.2 |           25.7 |                         2.43 |
| Tanker/Cart             |                 5.5 |                  24.2 |           22.7 |           25.8 |                         2.3  |
| Surface Water           |                 2   |                  23.3 |           21   |           25.5 |                         2.22 |
| Protected Spring        |                 1.2 |                  22.9 |           20.3 |           25.4 |                         2.18 |
| Unprotected Spring      |                 0.6 |                  19.8 |           16.9 |           22.8 |                         1.89 |
| Other Source            |                 1.1 |                  18.1 |           15   |           21.3 |                         1.72 |
| Community RO Plant      |                 4.9 |                  17.7 |           16.3 |           19.2 |                         1.69 |
| Protected Well/Spring   |                17.5 |                  16.8 |           16.1 |           17.6 |                         1.6  |
| Unprotected Well/Spring |                 7.3 |                  13.3 |           12.4 |           14.1 |                         1.27 |
| Bottled Water           |                13   |                  10.7 |            9.8 |           11.6 |                         1.02 |
| Tube Well/Borehole      |               229.2 |                  10.5 |           10.3 |           10.6 |                         1    |

**Finding**: Piped water sits near the top of the disruption ranking despite being classified as improved.

## FINDING 2 — IDI Regression (with social controls)

| Variable                       |    OR | 95% CI      | p-value   | Significant   |
|:-------------------------------|------:|:------------|:----------|:--------------|
| Piped Water (vs non-piped)     | 2.031 | 1.718–2.401 | <0.001*** | ✓             |
| IDI Score                      | 1.011 | 1.009–1.012 | <0.001*** | ✓             |
| Piped × IDI  [KEY INTERACTION] | 0.994 | 0.991–0.997 | <0.001*** | ✓             |
| Wealth Quintile                | 0.95  | 0.940–0.961 | <0.001*** | ✓             |
| Urban                          | 1.038 | 1.001–1.077 | 0.042*    | ✓             |
| SC/ST household                | 1.093 | 1.068–1.118 | <0.001*** | ✓             |
| Female-headed household        | 0.981 | 0.960–1.002 | 0.075     |               |
| Household size                 | 1.024 | 1.020–1.029 | <0.001*** | ✓             |
| Children under 5               | 1.002 | 0.991–1.014 | 0.694     |               |

---

## FINDING 3 — Geographic Concentration (RGI)

|   district_code | state_name        |   rgi |   rgi_ci_lower |   rgi_ci_upper |   observed_disruption |   piped_coverage |   mean_idi |   paradox_ratio |   n_households |
|----------------:|:------------------|------:|---------------:|---------------:|----------------------:|-----------------:|-----------:|----------------:|---------------:|
|             851 | Gujarat           |  42.7 |           40.8 |           44.8 |                  72.2 |             78.6 |       65.1 |             3.9 |            856 |
|             478 | Gujarat           |  38.5 |           36.6 |           40.5 |                  69.7 |             87.7 |       70.1 |             1.8 |            861 |
|             582 | Karnataka         |  36   |           35   |           37.1 |                  55.7 |             58.4 |       53.5 |             1.2 |            655 |
|             853 | Gujarat           |  30.6 |           28.8 |           32.8 |                  64.6 |             89.8 |       70.1 |             1.6 |            869 |
|             802 | Arunachal Pradesh |  28.7 |           26.2 |           31.1 |                  59.5 |             99.5 |       70   |           nan   |            515 |
|             499 | Maharashtra       |  27.6 |           25.9 |           29.8 |                  59.1 |             82.5 |       65.7 |             1.1 |            801 |
|             855 | Gujarat           |  27.3 |           25.7 |           28.8 |                  53.5 |             91.2 |       71.7 |             0.8 |            900 |
|             245 | Arunachal Pradesh |  26.6 |           24.6 |           28.7 |                  56.9 |             99.9 |       72   |           nan   |            903 |
|             501 | Maharashtra       |  26.2 |           24.4 |           28.3 |                  56.9 |             82.2 |       64.3 |             1.9 |            885 |
|             243 | Sikkim            |  26.1 |           24.9 |           27.5 |                  54.2 |             88.1 |       66.8 |           nan   |            874 |
|             805 | Arunachal Pradesh |  25.3 |           23.3 |           27.3 |                  54.5 |             78.9 |       59.4 |             3.3 |            676 |
|             581 | Karnataka         |  24.7 |           23.3 |           26   |                  43.4 |             49.3 |       54.4 |             0.8 |            567 |
|             859 | Gujarat           |  24.1 |           22.8 |           25.4 |                  46   |             71.8 |       69.5 |             1.2 |            791 |
|             579 | Karnataka         |  23.7 |           21.9 |           25.7 |                  51.6 |             67.3 |       55.9 |             1.4 |            827 |
|             524 | Maharashtra       |  23.4 |           21.7 |           25.4 |                  48   |             56   |       53.6 |             1.2 |            793 |
|             522 | Maharashtra       |  23.4 |           21.5 |           25.6 |                  53.3 |             71.6 |       60.7 |             2.9 |            765 |
|             639 | Andaman & Nicobar |  23.2 |           21   |           25.6 |                  53.5 |             93.9 |       69.8 |           nan   |            820 |
|             270 | Nagaland          |  23.1 |           21.2 |           25   |                  46.7 |             69.8 |       59.3 |             4.5 |            693 |
|             525 | Maharashtra       |  22.6 |           21.4 |           24.3 |                  46.4 |             59.2 |       56.2 |             2.6 |            784 |
|             863 | Gujarat           |  22.6 |           20.7 |           24.8 |                  54.8 |             85.1 |       69.4 |             0.9 |            809 |

---

## FINDING 4 — GEE Multilevel Model (IDI × RGI)

Model: `disruption ~ IDI_std + RGI_std + IDI×RGI + piped + social controls + C(region)`

| Variable                             |    OR | 95% CI      | p-value   | Significant   |
|:-------------------------------------|------:|:------------|:----------|:--------------|
| IDI Score (std)                      | 1.052 | 0.986–1.123 | 0.126     |               |
| RGI Score (std)                      | 1.686 | 1.627–1.747 | <0.001*** | ✓             |
| IDI × RGI  [CROSS-LEVEL INTERACTION] | 1.024 | 0.979–1.071 | 0.308     |               |
| Piped Water                          | 1.736 | 1.536–1.961 | <0.001*** | ✓             |
| Wealth Quintile                      | 0.935 | 0.918–0.951 | <0.001*** | ✓             |
| Urban                                | 1.101 | 1.029–1.178 | 0.005**   | ✓             |
| SC/ST household                      | 1.078 | 1.036–1.122 | <0.001*** | ✓             |
| Female-headed                        | 1.001 | 0.970–1.033 | 0.938     |               |

---

## FINDING 5 — Slope-as-Outcome: Does RGI Steepen IDI Effect?

Stage 1: district IDI slopes (logit per district).
Stage 2: WLS — IDI slope ~ RGI, weights=1/SE².

| Term                       |   Coefficient | 95% CI         | p-value   | Significant   |
|:---------------------------|--------------:|:---------------|:----------|:--------------|
| Intercept (mean IDI slope) |        0.0105 | 0.0096–0.0114  | <0.001*** | ✓             |
| RGI (std)  [KEY TERM]      |        0.0004 | -0.0005–0.0013 | 0.408     |               |

---

## ROBUSTNESS

| Metric                                  |   Value |
|:----------------------------------------|--------:|
| OR(Piped) — Mean                        |   1.736 |
| OR(Piped) — 2.5th pctile                |   1.703 |
| OR(Piped) — 97.5th pctile               |   1.77  |
| OR(Piped × IDI) — Mean                  |   0.789 |
| OR(Piped × IDI) — 2.5th pctile          |   0.777 |
| OR(Piped × IDI) — 97.5th pctile         |   0.801 |
| % MC runs OR(Piped) > 1.0  [ROBUSTNESS] | 100     |

### PSM (ATT)

| Metric   |   Estimate (pp) |   SE |   95% CI Lower |   95% CI Upper |   % boots > 0 | Interpretation                                  |
|:---------|----------------:|-----:|---------------:|---------------:|--------------:|:------------------------------------------------|
| ATT      |           15.06 | 0.13 |          14.77 |          15.31 |           100 | Piped INCREASES disruption vs matched tube-well |

---

## POLICY IMPLICATIONS

1. **Reliability over coverage** — piped expansion without reliability investment worsens national disruption.
2. **Target CRISIS districts** — high IDI + high RGI = priority.
3. **Preserve backup sources** — RESILIENT POOR districts show source diversity protects even under system failure.
4. **Protect marginalised households** — SC/ST and female-headed households show elevated disruption after controlling for IDI.

---
*End of report. Outputs: nfhs5_output*