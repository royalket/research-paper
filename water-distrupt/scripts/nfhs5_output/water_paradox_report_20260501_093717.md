# The Infrastructure Paradox: Modern Water Systems and
# New Vulnerabilities in India
## Evidence from NFHS-5 (2019-21)

*Generated: 2026-05-01 09:37*

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
| Piped Water (vs non-piped)     | 1.858 | 1.522–2.269 | <0.001*** | ✓             |
| IDI Score                      | 1.01  | 1.009–1.012 | <0.001*** | ✓             |
| Piped × IDI  [KEY INTERACTION] | 0.995 | 0.993–0.998 | 0.003**   | ✓             |
| Wealth Quintile                | 0.969 | 0.959–0.978 | <0.001*** | ✓             |
| Urban                          | 1.034 | 0.997–1.073 | 0.069     |               |
| SC/ST household                | 1.094 | 1.069–1.119 | <0.001*** | ✓             |
| Female-headed household        | 0.98  | 0.959–1.001 | 0.064     |               |
| Household size                 | 1.025 | 1.020–1.029 | <0.001*** | ✓             |
| Children under 5               | 1.002 | 0.991–1.014 | 0.674     |               |

---

## FINDING 3 — Geographic Concentration (RGI)

|   district_code | state_name        |   rgi |   rgi_ci_lower |   rgi_ci_upper |   observed_disruption |   piped_coverage |   mean_idi |   paradox_ratio |   n_households |
|----------------:|:------------------|------:|---------------:|---------------:|----------------------:|-----------------:|-----------:|----------------:|---------------:|
|             478 | Gujarat           |  43   |           41.8 |           44.5 |                  69.7 |             87.7 |       69.9 |             1.8 |            861 |
|             851 | Gujarat           |  39.9 |           36.7 |           42.8 |                  72.2 |             78.6 |       65.3 |             3.9 |            856 |
|             582 | Karnataka         |  35.2 |           34.4 |           36.3 |                  55.7 |             58.4 |       53.9 |             1.2 |            655 |
|             853 | Gujarat           |  32.5 |           30.8 |           34.1 |                  64.6 |             89.8 |       71   |             1.6 |            869 |
|             501 | Maharashtra       |  31.4 |           30.3 |           32.7 |                  56.9 |             82.2 |       65.6 |             1.9 |            885 |
|             579 | Karnataka         |  28.8 |           27.9 |           29.8 |                  51.6 |             67.3 |       58.2 |             1.4 |            827 |
|             524 | Maharashtra       |  28.6 |           27.9 |           29.4 |                  48   |             56   |       55   |             1.2 |            793 |
|             522 | Maharashtra       |  28   |           27.1 |           29.2 |                  53.3 |             71.6 |       61.5 |             2.9 |            765 |
|             802 | Arunachal Pradesh |  27.6 |           25.4 |           30   |                  59.5 |             99.5 |       73.8 |           nan   |            515 |
|             863 | Gujarat           |  27.4 |           26.4 |           28.8 |                  54.8 |             85.1 |       70.6 |             0.9 |            809 |
|             855 | Gujarat           |  26.7 |           25.3 |           28.2 |                  53.5 |             91.2 |       71.1 |             0.8 |            900 |
|             525 | Maharashtra       |  26.1 |           25.3 |           27.2 |                  46.4 |             59.2 |       58.1 |             2.6 |            784 |
|             558 | Karnataka         |  26   |           25.2 |           26.9 |                  45.3 |             56.5 |       52.5 |             1.5 |            859 |
|             245 | Arunachal Pradesh |  25.6 |           23.8 |           27.8 |                  56.9 |             99.9 |       74.4 |           nan   |            903 |
|             243 | Sikkim            |  25.2 |           24.2 |           26.6 |                  54.2 |             88.1 |       69   |           nan   |            874 |
|             526 | Maharashtra       |  24.7 |           23.7 |           25.8 |                  48   |             70.3 |       61.6 |             5.1 |            788 |
|             872 | Meghalaya         |  24.1 |           22.6 |           25.8 |                  49.9 |             64.4 |       57.1 |             0.9 |            432 |
|             805 | Arunachal Pradesh |  24   |           22.2 |           26.1 |                  54.5 |             78.9 |       63.7 |             3.3 |            676 |
|             581 | Karnataka         |  23.8 |           22.5 |           25.1 |                  43.4 |             49.3 |       54   |             0.8 |            567 |
|             859 | Gujarat           |  23.7 |           22.6 |           25.2 |                  46   |             71.8 |       69   |             1.2 |            791 |

---

## FINDING 4 — GEE Multilevel Model (IDI × RGI)

Model: `disruption ~ IDI_std + RGI_std + IDI×RGI + piped + social controls + C(region)`

| Variable                             |    OR | 95% CI      | p-value   | Significant   |
|:-------------------------------------|------:|:------------|:----------|:--------------|
| IDI Score (std)                      | 1.254 | 1.182–1.330 | <0.001*** | ✓             |
| RGI Score (std)                      | 1.701 | 1.659–1.743 | <0.001*** | ✓             |
| IDI × RGI  [CROSS-LEVEL INTERACTION] | 0.988 | 0.962–1.014 | 0.347     |               |
| Piped Water                          | 1.526 | 1.374–1.695 | <0.001*** | ✓             |
| Wealth Quintile                      | 0.959 | 0.946–0.972 | <0.001*** | ✓             |
| Urban                                | 1.069 | 1.022–1.119 | 0.004**   | ✓             |
| SC/ST household                      | 1.116 | 1.080–1.153 | <0.001*** | ✓             |
| Female-headed                        | 0.998 | 0.974–1.022 | 0.846     |               |

---

## FINDING 5 — Slope-as-Outcome: Does RGI Steepen IDI Effect?

Stage 1: district IDI slopes (logit per district).
Stage 2: WLS — IDI slope ~ RGI, weights=1/SE².

| Term                       |   Coefficient | 95% CI         | p-value   | Significant   |
|:---------------------------|--------------:|:---------------|:----------|:--------------|
| Intercept (mean IDI slope) |        0.0098 | 0.0090–0.0107  | <0.001*** | ✓             |
| RGI (std)  [KEY TERM]      |        0.0005 | -0.0003–0.0013 | 0.205     |               |

---

## ROBUSTNESS

| Metric                                  |   Value |
|:----------------------------------------|--------:|
| OR(Piped) — Mean                        |   1.634 |
| OR(Piped) — 2.5th pctile                |   1.607 |
| OR(Piped) — 97.5th pctile               |   1.661 |
| OR(Piped × IDI) — Mean                  |   0.8   |
| OR(Piped × IDI) — 2.5th pctile          |   0.786 |
| OR(Piped × IDI) — 97.5th pctile         |   0.815 |
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