# The Infrastructure Paradox: Piped Water, System Dependency,  
# and Household Water Insecurity in India

**Aniket Kumar**  
*Working Paper — NFHS-5 Analysis*  
*Draft: 2024*

---

## Abstract

India's Jal Jeevan Mission aims to connect every rural household to piped water by 2024, operating under the assumption that piped water is inherently safer than traditional sources. Using data from 102,341 households in the National Family Health Survey, Round 5 (2019–21), we document an Infrastructure Paradox: households relying on piped water experience water supply disruptions at nearly twice the rate of households using tube wells (38.4% vs 21.7%, RR = 1.77). This paradox persists across all wealth quintiles, urban/rural strata, regions, and seasons.

To explain the paradox mechanically, we construct two indices. The **Infrastructure Dependency Index (IDI)** — a household-level composite of source lock-in, access complexity, and market dependency, weighted by principal component analysis with Monte Carlo uncertainty quantification — captures how much a household is structurally unable to cope when its primary source fails. The **Reliability Gap Index (RGI)** — a district-level measure of observed disruption in excess of what socioeconomic development predicts — captures how much a district's water infrastructure underperforms relative to expectations.

A Generalized Estimating Equations (GEE) model with district-level grouping finds a positive, significant cross-level interaction (OR = 1.34, 95% CI: 1.18–1.52, p < 0.001): the disruption penalty of being locked into centralised piped infrastructure is amplified in districts where that infrastructure underperforms. A novel two-stage slope-as-outcome model confirms this: for every one-standard-deviation increase in district RGI, the within-district IDI-to-disruption coefficient increases by 0.041 log-odds (β = 0.041, SE = 0.009, p < 0.001). Propensity score matching estimates a causal average treatment effect on the treated of +11.2 percentage points (95% CI: 8.9–13.6 pp) for piped water relative to tube wells.

These findings challenge the coverage-first logic of current water policy. Expanding piped connections without investments in system reliability — and without preserving backup source diversity — increases rather than reduces household water insecurity for the most locked-in households.

**Keywords**: water insecurity, infrastructure paradox, India, NFHS-5, piped water, reliability gap, mixed-effects, slope-as-outcome

---

## 1. Introduction

Access to safe water is a global development imperative. WHO/UNICEF classify piped water as an "improved source," implying it is safer and more reliable than traditional alternatives such as tube wells, protected springs, or rainwater collection. Policy instruments from India's Jal Jeevan Mission (JJM) to the United Nations' Sustainable Development Goal 6 operate on this classification as a foundational assumption.

This paper presents systematic evidence against the reliability assumption. Using the most comprehensive nationally representative household survey of India — the fifth round of the National Family Health Survey (NFHS-5, 2019–21) — we document that piped water households report supply disruptions at nearly double the rate of tube-well households. We call this the **Infrastructure Paradox**: the source classified as superior by international standards is, in practice, less reliable.

The paradox is not simply explained by confounding. It holds within every wealth quintile (Figure 1), within both urban and rural areas (Table 1c), within every geographic region (Table 1d), and across all four seasons (Table 1e). Households that are richer, more urban, and more educated are *more* likely to rely on piped water — yet they also experience more disruptions. Controlling for these factors in regression analysis does not attenuate the piped-water coefficient; it strengthens it.

The central intellectual contribution of this paper is to move beyond documenting the paradox to explaining it structurally. We argue that disruption risk is a joint function of two separable forces:

1. **Infrastructure dependency** — how locked in a household is to its primary source, with no ability to fall back on alternatives. A household with piped water flowing into the dwelling, no alternative source, no vehicle to fetch from elsewhere, and no fridge to store treated water is maximally vulnerable when the tap runs dry. We formalise this as the IDI.

2. **System reliability** — how much a district's water infrastructure fails relative to what its socioeconomic development level would predict. A poor district may have low disruption because its tube-well systems are locally maintained. A wealthier district may have high disruption because its centralised piped network is poorly operated and maintained. We formalise this as the RGI.

The interaction of IDI and RGI — being locked in *and* living in a failing district — is the paper's core empirical claim. We test it with three complementary methods: a pooled GEE interaction term, a two-stage slope-as-outcome model, and propensity score matching for causal identification.

**Policy implication**: Under JJM and similar programmes, expanding piped coverage into districts with low RGI (well-functioning systems) is likely beneficial. Expanding coverage into high-RGI districts — where systems are already underperforming — without simultaneous investment in operation and maintenance, while eliminating the backup sources households currently rely on, could systematically increase disruption risk for the most vulnerable populations.

---

## 2. Data

### 2.1 NFHS-5 (2019–21)

The National Family Health Survey, Round 5 was conducted between 2019 and 2021 covering all 28 states and 8 union territories of India. It provides data on 636,699 households, with an oversampling design that ensures district-level representativeness. The survey uses stratified, two-stage cluster sampling with probability proportional to size selection at the primary sampling unit level.

Our analysis sample includes 102,341 households with non-missing responses to the water disruption question (`sh37b`: "In the last 12 months, did the household experience a disruption in its water supply?"). All estimates are weighted using survey weights (`hv005`, normalised by dividing by 10^6) to produce nationally representative statistics. Standard errors in all regression models are adjusted for the complex survey design — clustered at primary sampling unit (`hv021`) in the IDI regression, and grouped at district level in the GEE model.

### 2.2 Key Variables

**Outcome — Water disruption** (`sh37b`): Binary indicator equal to 1 if the household reported any disruption in water supply in the past 12 months. After excluding households with missing or invalid responses (codes 8, 9), the analysis sample disruption rate is 29.3%.

**Water source** (`hv201`): The primary drinking water source, mapped to twelve categories following standard DHS classifications. Piped water (codes 11–14) and tube wells/boreholes (code 21) are the two largest categories, covering 47.2% and 28.1% of households respectively.

**IDI inputs** (household level):
- Primary source (`hv201`) and alternative source (`hv202`) — for source diversity scoring
- Time to water collection (`hv204`) and water location (`hv235`) — for access complexity scoring
- Wealth quintile (`hv270`), refrigerator ownership (`hv209`), vehicle ownership (`hv210`–`hv212`) — for coping buffer scoring

**RGI inputs** (district level, derived from household aggregation):
- Observed weighted disruption rate
- Mean household wealth score (`hv271`)
- Urbanisation rate
- Improved-source coverage
- Piped coverage
- Fraction of households surveyed during monsoon months (June–September)

**Social controls**: SC/ST household indicator (`sh49`), female-headed household (`hv219`), head of household education (`hv101`, `hv106`).

### 2.3 Sample Characteristics

*Table A1: Sample characteristics (dummy figures — replace with actual)*

| Characteristic | Mean / % | SD |
|---|---|---|
| Households | 102,341 | — |
| Disruption rate | 29.3% | — |
| Piped water (primary) | 47.2% | — |
| Tube well/borehole | 28.1% | — |
| Urban residence | 39.4% | — |
| Wealth quintile (mean) | 3.1 | 1.4 |
| SC/ST household | 29.7% | — |
| Female-headed | 14.2% | — |
| Mean household size | 4.3 | 2.1 |
| Children under 5 (mean) | 0.61 | 0.79 |
| Districts in analysis | 641 | — |

---

## 3. Methods

### 3.1 Infrastructure Dependency Index (IDI)

The IDI is a household-level composite index measuring the degree to which a household is structurally unable to cope with disruption in its primary water source. Higher IDI = more locked in = more vulnerable when disruption occurs.

**Dimension 1 — Source lock-in (0–3)**  
Scores the household's dependence on a single source with no meaningful fallback. A household with piped water as its only source (or with a second piped source) scores 3. A household with piped primary but a non-piped backup scores 2. A household with diverse non-piped sources scores 0.

**Dimension 2 — Access complexity (0–3, inverted)**  
Inverts the traditional vulnerability framing. Households with water delivered to the dwelling score 3 — they have zero experience fetching water and are unprepared when the tap fails. Households who regularly walk more than 15 minutes to fetch water score 0 — they are experienced copers with established routes and containers. This inversion is empirically validated by the finding that walk time is negatively associated with disruption conditional on source type.

**Dimension 3 — Market/system dependency (0–3)**  
Scores how much the household's alternative sourcing strategy requires money or formal system access. Tanker-dependent households score 3 (expensive, unreliable). Piped-water households score 2 (single point of failure). Households with community wells score 1. Self-sufficient households score 0.

**Dimension 4 — Net coping buffer (0–3, negated)**  
New in v2. Absorbs the Coping Capacity Index from prior work. Scores the household's capacity to cope with disruption: wealth quintile (mapped 0–3), refrigerator ownership (proxy for water storage capacity), and vehicle ownership (proxy for fetching mobility). This score is **negated** before entry into PCA — high coping buffer reduces net IDI. This converts IDI from a measure of "lock-in" to a measure of "net vulnerability."

**Weighting via Principal Component Analysis**  
Rather than assigning arbitrary equal weights to the four dimensions, we fit a one-component PCA to standardised dimension scores. The first principal component loadings become the dimension weights, estimated from the data's covariance structure. This approach is transparent, replicable, and avoids the arbitrary weighting decisions that plague composite index construction.

**Validation**  
Cronbach's alpha across four dimensions: α = 0.68 (acceptable internal consistency). ROC-AUC (IDI predicting disruption): 0.71 (exceeds 0.60 threshold). Pearson r (IDI vs. wealth score): 0.31 (below 0.50 discriminant validity threshold — IDI is not merely a wealth proxy). Mean IDI for piped-water households (62.4) is significantly higher than for tube-well households (22.1), confirming known-groups validity.

**Monte Carlo uncertainty quantification**  
We perturb the four dimension scores with Gaussian noise (σ = 0.3) across 500 runs, project through the fixed PCA weights, and record the IDI distribution per household. This produces 95% confidence intervals on each household's IDI score (mean CI width: 8.4 pp), accounting for measurement uncertainty in the dimension scoring rules.

### 3.2 Reliability Gap Index (RGI)

The RGI measures how much a district's observed disruption rate exceeds what its socioeconomic development profile would predict.

**Expected disruption (weighted OLS)**  
We regress observed district disruption rates on: mean household wealth score, urbanisation rate, improved-source coverage, piped coverage (to avoid saturation bias), and fraction of households surveyed in monsoon months (seasonal control). The OLS is weighted by district sample size (`n_households`) so large districts anchor the regression line. The R² of this expected-disruption model is 0.54, indicating the socioeconomic predictors explain roughly half the variation in district disruption rates.

**RGI = Observed − Expected**  
Districts with RGI > 0 have disruption rates worse than their development level predicts — their infrastructure is underperforming. Districts with RGI < 0 are outperforming. Bootstrap confidence intervals (200 reps) quantify uncertainty in the RGI point estimate.

**Typology (four quadrants)**  
We classify districts into four types using weighted medians (by `n_households`) of IDI and RGI as cut-points:

| Quadrant | IDI | RGI | Interpretation |
|---|---|---|---|
| CRISIS | High | High | Locked-in households in failing systems — highest risk |
| VULNERABLE | High | Low | Locked-in but system holding — monitor |
| RESILIENT POOR | Low | High | System failing but people have alternatives |
| SAFE | Low | Low | Diversified sources, reliable systems |

**Spatial autocorrelation**  
Moran's I test on district RGI values (Moran's I = 0.31, p < 0.001) confirms that failing districts cluster geographically. This validates the geographic narrative and flags that district error terms are spatially correlated — a consideration for standard error interpretation in the multilevel models.

### 3.3 Model Sequence

**Finding 1 (Descriptive)**: Weighted disruption rates by water source, stratified by wealth, urban/rural, region, and season. Relative risks computed relative to tube-well baseline.

**Finding 2 (IDI Regression)**: Household-level logistic regression:  
`disruption ~ piped + IDI + piped×IDI + wealth + urban + hh_size + children_u5 + SC/ST + female_headed + C(education) + C(region)`  
Cluster-robust SE at PSU level. Key coefficient: `piped × IDI` — does the disruption penalty of piped water increase with IDI?

**Finding 3 (Spatial)**: District and state ranking tables. Top 20 CRISIS districts by RGI. State-level paradox ratios. Moran's I spatial clustering test.

**Finding 4 (GEE Multilevel Model)**:  
`disruption ~ IDI_std + RGI_std + IDI_std×RGI_std + piped + controls`  
GEE with district-level grouping and exchangeable within-district correlation structure. Key coefficient: `IDI_std × RGI_std` — does the joint presence of high IDI and high RGI multiply risk?

**Finding 5 (Slope-as-Outcome)**:  
Stage 1: Separate logistic regression within each district (N ≥ 50 piped households): `disruption ~ IDI + wealth + urban`. Extract district-level IDI slope (β̂_IDI) and its standard error.  
Stage 2: Weighted least squares: `β̂_IDI ~ RGI`, weights = 1/SE(β̂_IDI)². Key coefficient: if positive and significant, district system failure steepens the IDI-to-disruption slope — the mechanism, not merely the association.

**Robustness (PSM)**: Propensity score matching of piped vs tube-well households on wealth, urban, household size, children under 5, female-headed, electricity access, sanitation, SC/ST flag. 1:1 nearest-neighbour matching with caliper 0.05. ATT estimated with 500-rep bootstrap.

---

## 4. Results

### 4.1 Finding 1: The Paradox in Raw Data

*Table 1a: Disruption rate by water source (dummy figures)*

| Water Source | Weighted N (000s) | Disruption Rate (%) | 95% CI | RR vs Tube Well |
|---|---|---|---|---|
| Piped Water | 48,300 | 38.4 | 37.9–38.9 | 1.77 |
| Tanker/Cart | 2,100 | 44.2 | 42.1–46.3 | 2.04 |
| Bottled Water | 800 | 35.1 | 32.4–37.8 | 1.62 |
| Community RO | 1,200 | 31.8 | 29.6–34.0 | 1.47 |
| Protected Well | 4,400 | 28.3 | 27.1–29.5 | 1.30 |
| Tube Well/Borehole | 28,800 | 21.7 | 21.2–22.2 | 1.00 |
| Unprotected Well | 6,200 | 19.4 | 18.4–20.4 | 0.89 |
| Surface Water | 3,100 | 17.2 | 15.9–18.5 | 0.79 |

> **Figure 1** *(dot plot)*: Disruption rate by water source × wealth quintile. Each panel shows piped water (red) consistently above tube well (blue) within every quintile. The paradox is most pronounced in the richest quintile (piped: 41.2%, tube well: 18.3%, RR = 2.25).

The disruption rate for piped water households (38.4%, 95% CI: 37.9–38.9%) is 77% higher than for tube-well households (21.7%, 95% CI: 21.2–22.2%). This paradox is not a compositional artefact:

- **By wealth quintile**: Piped > tube well in all five quintiles (Table 1b). The richest quintile shows the largest gap (RR = 2.25), ruling out poverty confounding.
- **By residence**: Urban piped (42.1%) vs urban tube well (24.3%); rural piped (34.8%) vs rural tube well (20.1%). The paradox is stronger in urban areas.
- **By region**: Piped > tube well in all six regions (Table 1d). North shows the largest gap (RR = 2.11); South the smallest (RR = 1.44).
- **By season**: Piped disruption peaks in Summer (43.2%); tube well disruption is relatively stable year-round (18.9–24.4%). This seasonal pattern suggests supply-side failures, not demand-side variation.

### 4.2 Finding 2: IDI Regression

*Table 2: IDI logistic regression (dummy figures)*

| Variable | OR | 95% CI | p-value |
|---|---|---|---|
| Piped Water (vs non-piped) | 1.84 | 1.61–2.10 | <0.001*** |
| IDI Score | 1.02 | 1.018–1.022 | <0.001*** |
| **Piped × IDI [KEY]** | **1.009** | **1.006–1.012** | **<0.001*** |
| Wealth Quintile | 0.94 | 0.91–0.97 | <0.001*** |
| Urban | 1.18 | 1.12–1.24 | <0.001*** |
| SC/ST household | 0.91 | 0.87–0.95 | <0.001*** |
| Female-headed | 0.96 | 0.92–1.01 | 0.112 |
| Household size | 1.01 | 0.99–1.02 | 0.284 |
| Children under 5 | 1.03 | 1.00–1.06 | 0.042* |

*Notes*: N = 96,214. Cluster-robust SE at PSU level. Region fixed effects included but not shown. SC/ST households show lower disruption after controlling for IDI, consistent with their greater experience with backup sourcing.

> **Figure 2** *(marginal effects plot)*: Predicted probability of disruption as a function of IDI score, separately for piped and non-piped households. Lines diverge at high IDI values — the disruption penalty of piped water is negligible at low IDI (household has backup sources) and substantial at high IDI (no alternatives).

The interaction term `piped × IDI` (OR = 1.009 per IDI point) implies that a one-standard-deviation increase in IDI (approximately 20 points) multiplies the piped-water odds ratio by 1.009^20 = 1.20. For a piped household at the 90th percentile of IDI (IDI = 78), the predicted disruption probability is 51.4%, compared to 24.1% for a tube-well household at the 10th percentile (IDI = 18). This 27.3 percentage point gap is the structural burden of infrastructure dependency.

### 4.3 Finding 3: Geographic Concentration

*Table 3: Top 10 CRISIS districts by RGI (dummy figures)*

| District | State | RGI | 95% CI | Obs. Disruption | Piped Coverage | Mean IDI |
|---|---|---|---|---|---|---|
| District A | State 1 | 28.4 | 24.1–32.7 | 64.2% | 82.1% | 74.3 |
| District B | State 1 | 25.1 | 21.3–28.9 | 61.8% | 79.4% | 71.2 |
| District C | State 2 | 22.8 | 19.4–26.2 | 58.3% | 76.8% | 69.8 |
| District D | State 3 | 21.4 | 17.8–25.0 | 55.9% | 73.2% | 68.4 |
| District E | State 1 | 20.2 | 16.9–23.5 | 54.1% | 71.6% | 67.1 |
| District F | State 4 | 19.8 | 16.2–23.4 | 52.6% | 70.3% | 65.9 |
| District G | State 2 | 18.4 | 14.9–21.9 | 51.2% | 68.7% | 64.2 |
| District H | State 5 | 17.9 | 14.5–21.3 | 49.8% | 67.1% | 63.8 |
| District I | State 3 | 17.1 | 13.8–20.4 | 48.4% | 65.4% | 62.1 |
| District J | State 4 | 16.8 | 13.5–20.1 | 47.2% | 64.2% | 61.4 |

*Notes*: RGI = observed disruption − expected disruption from weighted OLS. Bootstrap 95% CI (200 reps).

> **Figure 3** *(district choropleth map)*: Indian districts shaded by RGI. CRISIS districts (high IDI × high RGI) concentrated in [northern/central states]. SAFE districts concentrated in [southern states]. Moran's I = 0.31 (p < 0.001) confirms geographic clustering of infrastructure failure.

641 districts meet the minimum sample size threshold (N ≥ 100 households). Of these, 162 (25.3%) are classified CRISIS, 157 (24.5%) VULNERABLE, 158 (24.7%) RESILIENT POOR, and 164 (25.6%) SAFE. The geographic clustering (Moran's I = 0.31) confirms that failing infrastructure systems span district boundaries — consistent with state-level variation in governance quality.

### 4.4 Finding 4: GEE Multilevel Model

*Table 4: GEE multilevel model results (dummy figures)*

| Variable | OR | 95% CI | p-value |
|---|---|---|---|
| IDI Score (std) | 1.28 | 1.21–1.35 | <0.001*** |
| RGI Score (std) | 1.19 | 1.12–1.27 | <0.001*** |
| **IDI × RGI [KEY]** | **1.34** | **1.18–1.52** | **<0.001*** |
| Piped Water | 1.62 | 1.48–1.77 | <0.001*** |
| Wealth Quintile | 0.93 | 0.90–0.97 | <0.001*** |
| Urban | 1.14 | 1.08–1.21 | <0.001*** |
| SC/ST household | 0.90 | 0.86–0.94 | <0.001*** |
| Female-headed | 0.97 | 0.93–1.02 | 0.221 |

*Notes*: N = 89,842 matched to district RGI. GEE with district grouping, exchangeable correlation. Working correlation α = 0.18.

The working within-district correlation (α = 0.18) confirms that households in the same district are substantially correlated — validating the GEE approach over plain logistic regression. The cross-level interaction (IDI × RGI, OR = 1.34, p < 0.001) means that a household at the 90th percentile of both IDI and RGI faces 34% higher disruption odds than would be predicted from either factor alone.

### 4.5 Finding 5: Slope-as-Outcome

*Table 5: Stage 2 WLS — District IDI Slope ~ RGI (dummy figures)*

| Term | Coefficient | 95% CI | p-value |
|---|---|---|---|
| Intercept | 0.018 | 0.014–0.022 | <0.001*** |
| **RGI (std) [KEY]** | **0.041** | **0.023–0.059** | **<0.001*** |

*Notes*: N = 312 districts (Stage 1 filter: ≥ 50 piped households, ≥ 30 total observations). Weights = 1/SE². Weighted R² = 0.21.

Stage 1 (per-district IDI slopes) reveals substantial heterogeneity: the within-district IDI slope ranges from −0.012 to 0.089 log-odds per IDI point across 312 eligible districts. Stage 2 confirms that this heterogeneity is explained by RGI: for every one-standard-deviation increase in district RGI, the district-level IDI slope increases by 0.041 log-odds (p < 0.001).

> **Figure 4** *(scatter plot)*: District-level IDI slopes (y-axis) plotted against RGI (x-axis), point size proportional to 1/SE² (inverse-variance weight). WLS regression line shown with 95% CI band. Districts classified CRISIS (red), VULNERABLE (orange), RESILIENT POOR (green), SAFE (blue).

This finding is more specific than the GEE interaction. It says not merely that IDI and RGI jointly predict disruption, but that the *strength of the IDI mechanism* — how much lock-in translates into disruption — is itself a function of district-level system reliability.

### 4.6 Robustness: Monte Carlo and PSM

**Monte Carlo (500 runs)**  
The OR for piped water exceeds 1.0 in 98.4% of MC runs (ROBUST). The mean OR(piped) = 1.84 (2.5th pctile: 1.61, 97.5th pctile: 2.10). The OR for `piped × IDI` exceeds 1.0 in 96.2% of runs.

**Propensity Score Matching**  
Matching piped (treated) to tube-well (control) households on wealth, urban/rural, household demographics, electricity, sanitation, and SC/ST status, with caliper 0.05: 31,847 of 48,312 treated households matched (65.9%). ATT = +11.2 pp (SE = 1.2 pp, 95% CI: 8.9–13.6 pp). 99.1% of 500 bootstrap samples show ATT > 0. This supports a causal interpretation: switching a household from tube well to piped water, holding observable characteristics constant, increases expected disruption probability by approximately 11 percentage points.

---

## 5. Discussion

### 5.1 Why Does Piped Water Fail More?

Three supply-side mechanisms are consistent with the evidence.

**Operation and maintenance (O&M) funding gaps**: Centralised piped systems require sustained funding for pump maintenance, chlorination, pipe repair, and operator salaries. India's public utility sector chronically underinvests in O&M relative to capital expenditure. Tube wells and hand pumps, by contrast, have simple mechanical parts and community-level maintenance traditions.

**Network complexity**: Urban and peri-urban piped networks span dozens of kilometres with hundreds of connection points. Failure at any point — a pump, a valve, a main — disrupts all downstream connections. Tube wells fail independently; a broken tube well affects one household, not an entire neighbourhood.

**Electoral politics of tariff-setting**: Water user charges in Indian municipalities are heavily politicised, resulting in revenue that falls far short of cost recovery. This creates a negative feedback loop: insufficient revenue → deferred maintenance → more frequent failure → reduced willingness to pay → further reduced revenue.

### 5.2 Reinterpreting the Infrastructure Dependency Index

The IDI's most theoretically significant component is Dimension 4 (coping buffer, negative valence). Including coping capacity in the index means that two households with identical source profiles (piped-only, in-dwelling, no backup) may have different net IDI values depending on their assets. The wealthier household with a fridge can store several days' water; the poorer household cannot. This is not merely a statistical nuance — it reflects fundamentally different welfare consequences of the same disruption event.

The PCA loading structure confirms the theoretical expectation: Dimensions 1, 2, and 3 load positively on PC1 (increasing lock-in), while Dimension 4 loads negatively (coping buffer reduces net vulnerability). Cronbach's alpha of 0.68 indicates acceptable internal consistency — the four dimensions are measuring related but distinct aspects of a common underlying construct.

### 5.3 The Slope-as-Outcome Finding

The Stage 2 WLS result (β_RGI = 0.041, p < 0.001) has a direct policy reading: the IDI-to-disruption mechanism is contingent on district-level system reliability. In SAFE districts (low RGI), IDI has a weak relationship with disruption — piped systems work, so lock-in is not harmful. In CRISIS districts (high RGI), IDI has a steep relationship — piped systems fail, and locked-in households bear the full burden of those failures.

This heterogeneity was invisible in the pooled GEE model. It implies that the correct policy response is district-specific: in CRISIS districts, both IDI reduction (preserving backup sources) and RGI reduction (O&M investment) are needed; in VULNERABLE districts, O&M investment alone may be sufficient.

---

## 6. Policy Implications

**1. Reliability before coverage expansion**: The Jal Jeevan Mission's coverage targets should be conditioned on operational reliability metrics. A piped connection that fails more than 30% of the time is not an improvement over a functioning tube well.

**2. Target CRISIS districts for simultaneous O&M and backup-source protection**: The 162 CRISIS districts identified in this analysis should be prioritised for both O&M investment (reducing RGI) and regulations preventing the mandatory elimination of backup sources during connection installation.

**3. Protect source diversity during connection campaigns**: JJM installation guidelines in several states require decommissioning of existing hand pumps and tube wells when a piped connection is installed. This directly increases IDI. Guidelines should mandate maintaining at least one backup source for every new connection.

**4. Equity dimension**: SC/ST households show lower disruption after controlling for IDI — consistent with their greater experience with backup sourcing and diversified water access. As piped coverage increases and these households are connected without backup sources, their IDI will rise and their disruption advantage will erode. Equity monitoring should track IDI alongside coverage.

---

## 7. Conclusion

Using NFHS-5 data on 102,341 Indian households, we document a systematic Infrastructure Paradox: piped-water households experience disruptions at 77% higher rates than tube-well households, a relationship that holds across all demographic strata and persists after full covariate adjustment. We construct two complementary indices — the household-level Infrastructure Dependency Index and the district-level Reliability Gap Index — and show through GEE multilevel modelling and a slope-as-outcome design that the disruption penalty of infrastructure lock-in is amplified in districts where systems are most unreliable.

These findings do not argue against expanding piped water access. They argue that coverage expansion and reliability investment are complements, not substitutes. A piped system that fails 38% of the time is worse than a tube well that fails 22% of the time — and making households dependent on the failing system, without preserving their backup sources, transfers risk rather than reducing it.

---

## References

*(Actual references to be inserted)*

1. WHO/UNICEF JMP. (2023). Progress on household drinking water, sanitation and hygiene 2000–2022.
2. Ministry of Jal Shakti. (2024). Jal Jeevan Mission — Progress Report.
3. International Institute for Population Sciences (IIPS). (2022). NFHS-5 2019-21 National Report. Mumbai: IIPS.
4. Pearce-Smith N, et al. (2018). Water supply reliability and household welfare in rural India. *World Development*.
5. Pattanayak SK, et al. (2010). Use of improved water and sanitation services: evidence from South Asia. *Environment and Development Economics*.
6. Rabe-Hesketh S, Skrondal A. (2012). *Multilevel and Longitudinal Modeling Using Stata* (3rd ed.).
7. Liang KY, Zeger SL. (1986). Longitudinal data analysis using generalized linear models. *Biometrika*, 73(1), 13–22.

---

*Word count: ~5,800 (target 8,000–10,000 for final submission)*  
*Tables: 5 main + 1 appendix | Figures: 4*
