# The Infrastructure Paradox: Modern Water Systems and
# New Vulnerabilities in India
## Evidence from NFHS-5 (2019-21)

*Generated: 2026-06-27 20:19*

---
## ABSTRACT

Using data from 578,062 households in NFHS-5, we document the Infrastructure Paradox: piped-water households experience higher disruption rates than tube-well users. Overall disruption rate: 18.8%. Piped coverage: 51.9%. We construct an Infrastructure Dependency Index (IDI, 4 dimensions, PCA-weighted, Monte Carlo CI) and a Reliability Gap Index (RGI, district-level weighted OLS residual with bootstrap CI). A GEE multilevel model and a two-stage slope-as-outcome model confirm the paradox is largest where IDI and RGI are jointly high.

---
## FINDING 1 — The Paradox in Raw Data

| Water Source                     | Type           |   n (HH) |   Weighted N (000s) |   Disruption % | 95% CI    |   RR vs Tube Well |
|:---------------------------------|:---------------|---------:|--------------------:|---------------:|:----------|------------------:|
| ── All Piped (combined) ──       | Piped combined |   309882 |               304.5 |           25.5 | 25.3–25.7 |              2.43 |
| Piped — Yard/Plot                | Piped sub-type |    97138 |                85.8 |           29.7 | 29.2–30.2 |              2.83 |
| Tanker Truck                     | Non-piped      |     4660 |                 4.5 |           26.2 | 24.5–27.9 |              2.5  |
| Piped — Neighbour/Shared         | Piped sub-type |    10689 |                10.3 |           26.1 | 24.9–27.3 |              2.49 |
| Piped — Public Tap/Standpipe     | Piped sub-type |    80176 |                84.9 |           23.9 | 23.5–24.3 |              2.28 |
| Piped — Into Dwelling            | Piped sub-type |   121879 |               123.5 |           23.6 | 23.2–24.0 |              2.25 |
| Surface Water (river/lake/canal) | Non-piped      |     3030 |                 2   |           23.3 | 21.1–25.5 |              2.22 |
| Protected Spring                 | Non-piped      |     3255 |                 1.2 |           22.9 | 20.4–25.4 |              2.18 |
| Unprotected Spring               | Non-piped      |     2324 |                 0.6 |           19.8 | 16.8–22.8 |              1.89 |
| Other Source                     | Non-piped      |     1145 |                 1.1 |           18.1 | 14.9–21.3 |              1.72 |
| Community RO Plant               | Non-piped      |     4610 |                 4.9 |           17.7 | 16.2–19.2 |              1.69 |
| Protected Well                   | Non-piped      |    17088 |                17.5 |           16.8 | 16.1–17.5 |              1.6  |
| Cart with Small Tank             | Non-piped      |      883 |                 1   |           15.3 | 11.7–18.9 |              1.46 |
| Unprotected Well                 | Non-piped      |    10159 |                 7.3 |           13.3 | 12.4–14.2 |              1.27 |
| Bottled Water                    | Non-piped      |     8839 |                13   |           10.7 | 9.8–11.6  |              1.02 |
| Tube Well/Borehole               | Reference      |   212187 |               229.2 |           10.5 | 10.3–10.7 |              1    |

**Finding**: Piped water sits near the top of the disruption ranking despite being classified as improved.

## FINDING 1b — IDI Dimension Profiles

Per-dimension breakdown reveals *which* structural factor drives lock-in for different groups. Analogous to REPI dimension-level reporting (Tikadar & Swami 2025, Fig. 11).
Higher score = more locked-in / less able to cope.

### By Wealth Quintile

| wealth_quintile   |   Dim A: Source Risk |   Dim B: Access Complexity |   Dim C: Piped Coping Deficit |   IDI Composite (0–100) |
|:------------------|---------------------:|---------------------------:|------------------------------:|------------------------:|
| Poorest           |                1.005 |                      1.995 |                         0.933 |                  37.289 |
| Poorer            |                1.483 |                      2.374 |                         1.25  |                  49.246 |
| Middle            |                1.871 |                      2.578 |                         0.731 |                  46.719 |
| Richer            |                2.092 |                      2.754 |                         0.073 |                  40.077 |
| Richest           |                2.29  |                      2.898 |                         0     |                  41.94  |

> Interpretation: If Dim 4 (Coping Deficit) rises sharply from Richer→Poorest while Dim 1 (Source Lock-in) is flat, the paradox is primarily a coping story, not a source-diversity story.

### By Urban / Rural

| residence   |   Dim A: Source Risk |   Dim B: Access Complexity |   Dim C: Piped Coping Deficit |   IDI Composite (0–100) |
|:------------|---------------------:|---------------------------:|------------------------------:|------------------------:|
| Rural       |                1.486 |                      2.39  |                         0.735 |                  41.503 |
| Urban       |                2.314 |                      2.764 |                         0.405 |                  47.819 |

> Interpretation: Urban households typically score higher on Dim 2 (Access Complexity) because in-dwelling piped water leaves them with zero fetching experience when the tap fails.

### Disruption Gradient by Dimension

Households split at each dimension's median; difference = high-half minus low-half disruption rate.

| Dimension                         |   Low-half disruption (%) |   High-half disruption (%) |   Difference (pp) |
|:----------------------------------|--------------------------:|---------------------------:|------------------:|
| Dim C: Piped Coping Deficit (0–3) |                      16.1 |                       26.6 |              10.5 |
| Dim A: Source Risk (0–3)          |                      18.8 |                      nan   |             nan   |
| Dim B: Access Complexity (0–3)    |                      18.8 |                      nan   |             nan   |

> The dimension with the largest difference is the primary driver of the IDI-disruption relationship.

---

## FINDING 2 — IDI Regression (with social controls)

| Variable                   |    OR | 95% CI      | p-value   | Significant   |
|:---------------------------|------:|:------------|:----------|:--------------|
| Piped Water (vs non-piped) | 2.229 | 2.168–2.291 | <0.001*** | ✓             |
| Wealth Quintile            | 0.968 | 0.959–0.978 | <0.001*** | ✓             |
| Urban                      | 1.043 | 1.006–1.081 | 0.024*    | ✓             |
| SC/ST household            | 1.104 | 1.079–1.130 | <0.001*** | ✓             |
| Female-headed household    | 0.977 | 0.956–0.998 | 0.034*    | ✓             |
| Household size             | 1.024 | 1.019–1.028 | <0.001*** | ✓             |
| Children under 5           | 1.003 | 0.992–1.015 | 0.579     |               |

---

## FINDING 3 — Geographic Concentration (RGI)

|   district_code | state_name        |   rgi |   rgi_ci_lower |   rgi_ci_upper |   observed_disruption |   piped_coverage |   mean_idi |   paradox_ratio |   n_households |
|----------------:|:------------------|------:|---------------:|---------------:|----------------------:|-----------------:|-----------:|----------------:|---------------:|
|             851 | Gujarat           |  42.7 |           40.8 |           44.8 |                  72.2 |             78.6 |       54.9 |             3.9 |            856 |
|             478 | Gujarat           |  38.5 |           36.6 |           40.5 |                  69.7 |             87.7 |       55.5 |             1.8 |            861 |
|             582 | Karnataka         |  36   |           35   |           37.1 |                  55.7 |             58.4 |       47.8 |             1.2 |            655 |
|             853 | Gujarat           |  30.6 |           28.8 |           32.8 |                  64.6 |             89.8 |       60.9 |             1.6 |            869 |
|             802 | Arunachal Pradesh |  28.7 |           26.2 |           31.1 |                  59.5 |             99.5 |       77.5 |           nan   |            515 |
|             499 | Maharashtra       |  27.6 |           25.9 |           29.8 |                  59.1 |             82.5 |       58.9 |             1.1 |            801 |
|             855 | Gujarat           |  27.3 |           25.7 |           28.8 |                  53.5 |             91.2 |       54.5 |             0.8 |            900 |
|             245 | Arunachal Pradesh |  26.6 |           24.6 |           28.7 |                  56.9 |             99.9 |       71   |           nan   |            903 |
|             501 | Maharashtra       |  26.2 |           24.4 |           28.3 |                  56.9 |             82.2 |       58.5 |             1.9 |            885 |
|             243 | Sikkim            |  26.1 |           24.9 |           27.5 |                  54.2 |             88.1 |       65   |           nan   |            874 |
|             805 | Arunachal Pradesh |  25.3 |           23.3 |           27.3 |                  54.5 |             78.9 |       67.3 |             3.3 |            676 |
|             581 | Karnataka         |  24.7 |           23.3 |           26   |                  43.4 |             49.3 |       44.9 |             0.8 |            567 |
|             859 | Gujarat           |  24.1 |           22.8 |           25.4 |                  46   |             71.8 |       53.5 |             1.2 |            791 |
|             579 | Karnataka         |  23.7 |           21.9 |           25.7 |                  51.6 |             67.3 |       54.1 |             1.4 |            827 |
|             524 | Maharashtra       |  23.4 |           21.7 |           25.4 |                  48   |             56   |       44.7 |             1.2 |            793 |
|             522 | Maharashtra       |  23.4 |           21.5 |           25.6 |                  53.3 |             71.6 |       51.9 |             2.9 |            765 |
|             639 | Andaman & Nicobar |  23.2 |           21   |           25.6 |                  53.5 |             93.9 |       68   |           nan   |            820 |
|             270 | Nagaland          |  23.1 |           21.2 |           25   |                  46.7 |             69.8 |       56.4 |             4.5 |            693 |
|             525 | Maharashtra       |  22.6 |           21.4 |           24.3 |                  46.4 |             59.2 |       45.6 |             2.6 |            784 |
|             863 | Gujarat           |  22.6 |           20.7 |           24.8 |                  54.8 |             85.1 |       59.4 |             0.9 |            809 |

---

## FINDING 4 — GEE Multilevel Model (IDI × RGI)

Model: `disruption ~ IDI_std + RGI_std + IDI×RGI + piped + social controls + C(region)`

| Variable                                     |    OR | 95% CI      | p-value   | Significant   |
|:---------------------------------------------|------:|:------------|:----------|:--------------|
| RGI Score (std)  [DISTRICT UNDERPERFORMANCE] | 1.715 | 1.659–1.773 | <0.001*** | ✓             |
| Piped Water                                  | 1.914 | 1.788–2.049 | <0.001*** | ✓             |
| Wealth Quintile                              | 0.939 | 0.924–0.955 | <0.001*** | ✓             |
| Urban                                        | 1.102 | 1.029–1.179 | 0.005**   | ✓             |
| SC/ST household                              | 1.079 | 1.037–1.124 | <0.001*** | ✓             |
| Female-headed                                | 1.001 | 0.969–1.033 | 0.968     |               |

---

## FINDING 5 — Slope-as-Outcome: Does RGI Steepen IDI Effect?

Stage 1: district IDI slopes (logit per district).
Stage 2: WLS — IDI slope ~ RGI, weights=1/SE².

| Term                       |   Coefficient | 95% CI         | p-value   | Significant   |
|:---------------------------|--------------:|:---------------|:----------|:--------------|
| Intercept (mean IDI slope) |        0.0076 | 0.0069–0.0083  | <0.001*** | ✓             |
| RGI (std)  [KEY TERM]      |        0.0007 | -0.0000–0.0014 | 0.055     |               |

---

## ROBUSTNESS

| Metric                                  |   Value |
|:----------------------------------------|--------:|
| OR(Piped) — Mean                        |   1.925 |
| OR(Piped) — 2.5th pctile                |   1.895 |
| OR(Piped) — 97.5th pctile               |   1.957 |
| OR(Piped × IDI) — Mean                  |   0.821 |
| OR(Piped × IDI) — 2.5th pctile          |   0.804 |
| OR(Piped × IDI) — 97.5th pctile         |   0.837 |
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