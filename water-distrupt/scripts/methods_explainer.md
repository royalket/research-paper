# Methods Explainer: Every Calculation, Why We Do It, and What It Means

*This document explains every statistical and computational step in the NFHS-5 Water Paradox analysis. It is written to be understood by a reader with basic statistics knowledge — not assuming familiarity with the specific methods used.*

---

## Part 1: The Data Foundation

### 1.1 What is NFHS-5 and why use it?

The National Family Health Survey (NFHS) is India's equivalent of the Demographic and Health Survey (DHS). Round 5 was conducted 2019–21, covering all states and union territories. It is:
- **Nationally representative**: stratified two-stage cluster sampling ensures every district has enough observations to produce reliable estimates
- **Weighted**: survey weights (`hv005`) correct for unequal probability of selection. We divide by 10^6 because DHS stores weights as integers multiplied by 10^6.
- **Detailed on water**: it asks not just what source households use, but whether it was disrupted, where it is located, how long it takes to reach, and who fetches it — exactly the variables we need for IDI

**Why survey weights matter**: Without weights, our estimates would over-represent states where more households were sampled. A disruption rate calculated without weights would not represent the true national disruption experience.

### 1.2 The outcome variable: water_disrupted (sh37b)

**What it is**: A binary variable (1 = yes, 0 = no) answering "In the last 12 months, did your household experience a disruption in its water supply?"

**Why it is binary and what that means**: Disruption is a 0/1 event per household. This means we use logistic regression (not linear regression) for all individual-level models, because the outcome is a probability bounded between 0 and 1.

**What we lose**: Duration, frequency, and severity of disruption are collapsed into a single bit. A household disrupted for one day is coded the same as one disrupted for three months. This is a limitation of the data, not the analysis. We note it in the paper but cannot address it without additional data.

**Missing value handling**: Responses coded 8 (don't know) or 9 (missing) are dropped from the analysis sample. We report how many households are dropped and check that the drop rate does not differ systematically by wealth or water source (which would introduce selection bias).

---

## Part 2: The Infrastructure Dependency Index (IDI)

### 2.1 Why build a composite index at all?

We could simply put piped_flag, time_to_water, and wealth into the regression separately. The problem is:

1. **Conceptual**: The paradox mechanism is about a *type of household* — one that is locked in. No single variable captures this. Piped water alone does not tell you whether the household has a backup. Time-to-water alone does not tell you whether the household is piped. We need a composite.

2. **Statistical**: A composite reduces the multiple-testing problem. Instead of running 12 separate tests on each dimension and worrying about Type I error inflation, we test one well-constructed index.

3. **Policy**: An index is actionable. A policymaker can target "households with IDI > 70" in a way they cannot target "households that are piped AND have water in the dwelling AND have no alternative AND are in the bottom three wealth quintiles."

### 2.2 Dimension 1: Source Lock-in

**What it measures**: Whether the household has any meaningful alternative if its primary source fails.

**The scoring logic**:
- Score 3: Piped water as primary source, AND either no alternative source OR another piped source as alternative. This household has zero fallback outside the same failing system.
- Score 2: Piped water primary, but a non-piped alternative exists. The household can escape the piped system if it fails, but normally doesn't.
- Score 1: Non-piped primary source but only the same type of alternative (e.g., two different tube wells). Some diversity, but limited — both could fail for the same reason (e.g., groundwater depletion).
- Score 0: Non-piped primary with a genuinely different type of alternative. Maximally diversified.

**Why this matters**: This is the core of the paradox mechanism. A household that is 100% dependent on a single centralised system cannot cope when that system fails. The diversity-resilience relationship is well-established in ecology; we apply it to water supply systems.

**Variable used**: `hv201` (primary source) and `hv202` (alternative/other source).

### 2.3 Dimension 2: Access Complexity (Inverted)

**What it measures**: The household's *experience* with fetching water — and therefore its preparedness for disruption.

**The inversion**: Traditional vulnerability indices treat long walk times as a sign of vulnerability (the household has to work hard to get water). We invert this. A household that already walks 30 minutes to fetch water every day has:
- Established routes to alternative sources
- Containers appropriate for carrying water
- Physical capacity for fetching
- Knowledge of which sources are open when

When disruption hits, this household barely changes behaviour. The household with in-dwelling piped water has none of these — it is completely unprepared.

**The scoring logic**:
- Score 3: Water in dwelling (zero fetching experience, maximally unprepared)
- Score 2: Water in yard or plot (some fetching experience, mostly unprepared)
- Score 1: Must go elsewhere, < 15 minutes (some experience)
- Score 0: Must go elsewhere, ≥ 15 minutes (experienced coper)

**Variables used**: `hv235` (water location: 1=in dwelling, 2=in yard, 3=elsewhere) and `hv204` (time to water in minutes; 996 = on premises, recoded to 0).

**Empirical validation of the inversion**: In the data, conditional on water source type, households with longer walk times have *lower* disruption rates. This confirms the inversion is empirically grounded, not merely theoretical.

### 2.4 Dimension 3: Market/System Dependency

**What it measures**: How much the household's coping strategy — when its primary source fails — requires money or formal system access.

**The scoring logic**:
- Score 3: Primary source is tanker or bottled water. These are expensive, unreliable, and themselves depend on supply chains. When the piped system fails, these households must buy water from vendors who may also raise prices.
- Score 2: Piped water. The system is managed by a utility that may fail for reasons entirely outside the household's control (pump failure, electricity outage, pipe burst, billing dispute).
- Score 1: Community RO plant, protected well, or protected spring. Managed sources, but community-level — less subject to centralised failure.
- Score 0: Own well, rainwater collection, surface water. These may be lower quality but they are self-sufficient — the household controls access.

**Why market dependency amplifies lock-in**: A household whose only coping option is to buy water from a tanker vendor is at the mercy of both the failed piped system AND the vendor's willingness to supply. In genuine water crises, tanker prices spike (documented in Chennai 2019). This is a second-order vulnerability that Dimension 1 alone does not capture.

### 2.5 Dimension 4: Net Coping Buffer (Negative Valence)

**What it measures**: The household's *capacity* to absorb a disruption without severe welfare consequences.

**Why add this**: Dimensions 1–3 all measure structural features of the water system — what type of source, where it is, how dependent you are. They do not measure whether the household can *cope* if disruption occurs. Two households with identical water system profiles may have very different outcomes: one is wealthy with a large fridge stocked with filtered water and a car to fetch from an alternative; the other is poor with no storage and no transport.

**The scoring logic** (0–3, then negated):
- Wealth quintile: mapped 1→0, 2→0, 3→1, 4→2, 5→3. The bottom two quintiles add nothing to the buffer; the richest adds 3.
- Refrigerator (`hv209`): adds 1 if owned. A fridge allows storage of several days' worth of water.
- Vehicle (`hv210`/`hv211`/`hv212`): adds 1 if motorcycle or car is owned. A vehicle allows the household to fetch water from more distant alternatives that are not accessible on foot.
- Clipped to [0, 3] before negation.

**The negation**: The score is multiplied by −1 before entering PCA. So a household with a strong coping buffer (score = 3) enters PCA as −3, pulling the IDI *down*. A household with no coping buffer (score = 0) enters as 0, providing no offset.

**What we expect from PCA**: If coping genuinely offsets lock-in, the PCA loading on Dim 4 (already negated) should be positive — consistent with Dims 1, 2, 3. If it loads negatively, it means in the data, higher coping is associated with *lower* lock-in (less piped, more traditional sources) — which would still be theoretically coherent but would mean Dim 4 is pulling in a different direction. We report this loading explicitly and discuss its interpretation.

### 2.6 Principal Component Analysis (PCA) for Weighting

**The problem with equal weights**: If we simply add Dims 1 + 2 + 3 + 4 with equal weights, we are assuming each dimension contributes equally to the underlying concept of "infrastructure dependency." There is no reason to believe this. PCA lets the data determine the weights.

**What PCA does**: PCA finds the linear combination of the four standardised dimensions that has the maximum variance across households. This first principal component is the direction in the four-dimensional space along which households are most spread out — which is also the direction that captures the most of what varies between households in terms of water vulnerability.

**Standardisation**: Before PCA, each dimension is standardised to mean 0, standard deviation 1. This is necessary because the dimensions have different natural ranges (all 0–3, but with different distributions). Without standardisation, whichever dimension has the most variance would dominate the PCA.

**The loadings as weights**: The first PC loadings (e.g., Dim1: 0.41, Dim2: 0.38, Dim3: 0.44, Dim4: 0.33 — hypothetical) are the coefficients by which we multiply the standardised dimension scores to get each household's IDI. These are the data-driven weights.

**Variance explained**: If PC1 explains 65% of the variance in the four dimensions, that means 65% of what varies across households in their water system configuration is captured by a single underlying dimension — "infrastructure dependency." We report this.

**Cronbach's alpha**: A supplementary measure of internal consistency. Alpha > 0.7 is conventionally considered "good." With four dimensions, alpha = 0.68 is acceptable. Importantly, if alpha were very low (say, 0.3), it would suggest the four dimensions are not measuring a common underlying construct and should not be combined — we would need to keep them separate.

### 2.7 Monte Carlo Uncertainty Quantification

**The problem**: Our dimension scores are not measured perfectly. The mapping of `hv235` (water location) to a score of 0, 1, 2, or 3 involves judgment calls. The noise parameter σ = 0.3 represents our uncertainty about those scoring thresholds. If the boundary between score 2 and score 3 were shifted slightly, how would the IDI change?

**The procedure** (500 runs):
1. Add Gaussian noise to each dimension score: `perturbed = clip(dim_score + N(0, 0.3), 0, 3)`
2. Project perturbed scores through the *fixed* PCA weights (the weights don't change — only the inputs vary)
3. Record the resulting IDI score for each household
4. Also fit a logistic regression in each run to record the OR for piped water

**Why NOT refit PCA in each run**: If we refitted PCA each run, we would be changing the weights AND the inputs simultaneously, confounding the two sources of uncertainty. We want to isolate input uncertainty from weighting uncertainty. The PCA is fit once on the observed data.

**What we get**:
- Per household: mean IDI across 500 runs, 2.5th and 97.5th percentiles → 95% confidence interval on that household's IDI
- OR(piped) distribution: if piped water OR > 1 in 98% of runs, the finding is robust to dimension scoring uncertainty

**Speed improvements (v2)**:
- Pre-built positional array: instead of a Python dictionary lookup per row per run (100,000 × 500 = 50 million operations), we pre-compute which position in the full household array each regression-sample row corresponds to. This is done once before the loop. Inside the loop, it's a single NumPy array index.
- No cluster-robust SE in MC loop: cluster-robust standard errors require computing a "meat" matrix (a sum over PSU groups). This doubles computation per logit fit. In the MC loop, we only need OR point estimates, not their standard errors. Using plain Newton solver cuts each fit by ~60%.
- In-place update: instead of copying the entire regression DataFrame 500 times (500 × 100,000 rows = 50 million row-copies), we pre-allocate a float array and update it in place each run.

---

## Part 3: The Reliability Gap Index (RGI)

### 3.1 Why a district-level index?

Water supply systems in India are managed at the district or state level. A household's disruption experience is partly a function of its own characteristics (captured by IDI) and partly a function of the system it is plugged into. That system quality varies by district — some districts maintain their infrastructure well, others don't.

If we include district effects only as fixed effects in the regression, we cannot identify what *district characteristics* explain variation in disruption. The RGI gives us a single district-level measure that we can then interact with IDI.

### 3.2 The Expected-Disruption Model

**Goal**: Predict, for each district, what disruption rate we would expect given its socioeconomic development level — if its infrastructure performed normally.

**Features used** in the OLS:
1. `mean_wealth_score`: wealthier districts have more resources to cope with disruption (demand-side)
2. `pct_urban`: urban areas have more complex networks but also more resources
3. `improved_coverage`: districts with more improved-source access should have lower disruption
4. `piped_coverage` *(new in v2)*: controls for saturation bias. A district with 90% piped coverage mechanically has more households exposed to piped-system failure. Without this control, high-coverage districts look like "infrastructure underperformers" when they are really just more exposed.
5. `pct_monsoon` *(new in v2)*: the fraction of households in the district surveyed during monsoon months (June–September). Monsoon disruptions (flooding, contamination) are different from dry-season supply shortfalls. Districts surveyed more during monsoon will have higher observed disruption for a mechanical reason unrelated to infrastructure quality. This controls for that.

**Weighted OLS**: The regression is weighted by `n_households` in each district. This means districts with 3,000 households have 30× more influence on the regression line than districts with 100 households. The large districts have more reliable observed disruption rates, so this is methodologically sound.

**What R² tells us**: If R² = 0.54, that means 54% of variation in district disruption rates is explained by socioeconomic development alone. The remaining 46% is what the RGI captures — the "extra" disruption not explained by poverty, urbanisation, or coverage.

### 3.3 RGI = Observed − Expected

**Interpretation**:
- RGI = +20 pp: This district's disruption rate is 20 percentage points higher than its development level predicts. Its infrastructure is severely underperforming.
- RGI = −10 pp: This district's disruption rate is 10 pp lower than predicted. Its infrastructure is outperforming — possibly because of better O&M, stronger community management, or both.
- RGI ≈ 0: Performing as predicted.

**Why this is better than observed disruption alone**: A district with 50% disruption in a poor rural area might just have a lot of poor households. A district with 50% disruption in a wealthy urban area has a severe infrastructure problem. The RGI separates these.

### 3.4 Bootstrap Confidence Intervals on RGI

**The problem**: The RGI point estimate (RGI = observed − expected) has uncertainty because the expected-disruption OLS is fit on a sample of districts, not the population of districts. If we had a different sample of districts, the OLS line would be slightly different, and all RGI values would shift.

**The procedure** (200 bootstrap reps):
1. Draw a bootstrap sample of districts (with replacement, same N)
2. Refit the OLS on the bootstrap sample
3. Compute predicted (expected) disruption for all districts using the bootstrap OLS
4. Compute bootstrap RGI = observed − bootstrap expected
5. After 200 reps, take the 2.5th and 97.5th percentiles of RGI for each district

**Result**: Each district gets a 95% CI on its RGI. A district with RGI = 15, CI [12, 18] is clearly a "Severe Failure" district. A district with RGI = 6, CI [−2, 14] overlaps with "As Expected" — we should classify it cautiously.

### 3.5 Weighted Median for Typology Cut-points

**Why weighted median**: When we split districts into high-IDI and low-IDI halves using the median, we want the cut-point to represent the typical household, not the typical district. A simple median treats a district with 100 households the same as one with 5,000.

**How it works**: Sort districts by IDI. Accumulate n_households weights. The weighted median is the IDI value at which the cumulative household count crosses 50% of total households. This cut-point now divides the population of households, not the population of districts.

**Practical effect**: Small, noisy districts with extreme IDI values have less influence on where the quadrant boundaries are drawn.

### 3.6 Moran's I Spatial Autocorrelation

**What Moran's I tests**: Whether districts with similar RGI values are geographically clustered. Moran's I = 1 means perfectly clustered (every district surrounded by districts with similar RGI). Moran's I = 0 means random spatial pattern. Moran's I = −1 means perfectly dispersed (every district surrounded by opposites).

**Why we test this**:
1. *Validation*: Real infrastructure systems (river systems, state water boards, electrical grids) span district boundaries. If RGI is measuring real infrastructure quality, it should cluster geographically. A significant positive Moran's I validates that RGI is capturing genuine spatial variation in system quality, not statistical noise.
2. *Model assumption warning*: Our GEE and slope-as-outcome models treat district-level RGI values as independent observations. If districts are strongly spatially autocorrelated, standard errors may be under-estimated. We report the Moran's I and note this limitation.

**Using libpysal**: We use k=5 nearest-neighbour spatial weights (based on district code ordering as a proxy — ideally replaced with geographic coordinates). The randomisation-based p-value is computed from permutation tests (999 random reassignments of RGI values).

---

## Part 4: The Regression Models

### 4.1 IDI Regression (Finding 2)

**Why logistic regression**: The outcome is binary (0/1 disruption). Logistic regression models the log-odds of disruption as a linear function of predictors. The exponentiated coefficient is an Odds Ratio (OR): OR > 1 means higher odds of disruption.

**The key interaction term (piped_flag × idi_mean)**:
This is the central test. Without this term, we would only be asking: "Is piped water associated with higher disruption?" (yes, we know that). With this term, we ask: "Does the disruption penalty of piped water increase as IDI increases?" If the interaction OR > 1, then: among piped households, higher IDI means even higher disruption — the paradox is amplified by lock-in.

**Social controls (new in v2)**:
- `sc_st_flag`: SC/ST households may face water access discrimination or may rely more on community sources. Including this controls for caste-based water access patterns.
- `female_headed`: Female-headed households often have different time allocation (women fetching water) and different bargaining power with utilities.
- `C(head_education)`: Education affects knowledge of water quality risks, ability to navigate utility bureaucracy, and income-earning potential.

These absorb the old WVI "social component." By including them as controls rather than baking them into the index, we avoid a key conceptual error: caste is not a feature of the *water system's* failure. It is a characteristic of the household that modifies the household's experience of failure. It belongs in the regression as a control, not in the index.

**Cluster-robust standard errors at PSU level**: Households within the same primary sampling unit (village or urban block) are surveyed together and share community-level characteristics. Ignoring this clustering would under-estimate standard errors, producing artificially significant p-values. Clustering at PSU level corrects this.

### 4.2 GEE Multilevel Model (Finding 4)

**The problem with flat logistic regression**: Households within the same district share unmeasured characteristics — the quality of the local water utility, the reliability of the electricity supply that powers pumps, local political investment in infrastructure maintenance. When we ignore this, we violate the independence assumption and under-estimate standard errors.

**Why not a random-effects GLMM**: A full random-effects logistic regression (e.g., using lme4 in R) requires distributional assumptions about the random effects (typically normally distributed random intercepts). With 641 districts, these assumptions are difficult to verify and estimation is computationally demanding. Additionally, the parameter estimates from random-effects models are subject-specific (they describe the typical individual's probability) rather than population-averaged (they describe the average probability in the population), which is what we want for policy claims.

**Why GEE**: GEE is a semi-parametric approach that accounts for within-group correlation without specifying a full random-effects distribution. It estimates population-averaged parameters and uses an empirical "sandwich" estimator for standard errors that is robust to mis-specification of the within-district correlation structure. For policy claims ("the average Indian household's probability of disruption increases by X for each unit increase in IDI"), population-averaged estimates are the right estimand.

**Exchangeable correlation structure**: We assume all pairs of households within the same district are equally correlated. The working correlation coefficient α is estimated from the data. If α = 0.18, that means two randomly selected households in the same district have a correlation of 0.18 in their disruption outcomes — substantial enough to matter for inference.

**The cross-level interaction (idi_std × rgi_std)**: Both IDI and RGI are standardised (mean 0, SD 1) before the interaction is computed. This makes the interaction coefficient interpretable: it represents the change in log-odds of disruption for a one-SD increase in IDI, for every one-SD increase in RGI. Standardising also reduces multicollinearity between main effects and the interaction term.

### 4.3 Slope-as-Outcome Model (Finding 5)

**The limitation of the GEE interaction**: The interaction term `idi_std × rgi_std` in the GEE model tells us that IDI and RGI jointly predict disruption. But it pools all districts together. It cannot directly answer: "In districts with high RGI, is the IDI effect genuinely steeper?"

**Stage 1 — District-level IDI slopes**: For each district with at least 50 piped-water households and 30 total observations, we fit:

`logit(disruption) = α_d + β_d × IDI + γ_d × wealth + δ_d × urban`

The key output is β_d — the IDI coefficient within district d. Some districts will have β_d = 0.08 (IDI strongly predicts disruption locally) and others will have β_d = 0.01 (IDI barely predicts disruption locally). This heterogeneity is the phenomenon we want to explain.

We also record SE(β_d) — the standard error of the slope estimate in each district. Districts with fewer households will have larger SE (less precise slope estimates).

**Why we need a minimum of 50 piped households**: A logistic regression needs variation in the predictor (IDI) and in the outcome (disruption). If a district has only 20 piped households, the IDI slope estimate will be highly imprecise — essentially noise. Setting a minimum of 50 piped households (and 30 total observations) ensures that Stage 1 slopes are based on sufficient data.

**Stage 2 — Weighted least squares**: We then regress the Stage 1 slopes on district-level RGI:

`β̂_d = a + b × RGI_d + ε_d`

This is a cross-level regression: the unit of analysis is the district (not the household). The outcome is the district's IDI slope from Stage 1.

**Inverse-variance weighting**: The regression is weighted by 1/SE(β_d)². Districts with precise slope estimates (small SE) get high weight; districts with imprecise slopes (large SE) get low weight. This is statistically optimal — it minimises the variance of the Stage 2 coefficient estimates. It is equivalent to what a random-effects meta-analysis would do.

**Interpretation of b (the Stage 2 coefficient)**:
- Positive and significant: Districts with higher reliability gaps (failing infrastructure) show steeper IDI-to-disruption relationships. In those districts, being locked in to the piped system has a large per-unit effect on disruption probability.
- Not significant: The district-level IDI effect does not vary systematically with RGI. The GEE interaction might still be significant due to household-level co-variation, but the specific mechanism (RGI steepening the IDI effect) would not be confirmed.

**Why this is stronger than the GEE interaction**: The GEE interaction picks up any joint effect of IDI and RGI. The slope-as-outcome explicitly tests whether the IDI mechanism (how strongly lock-in predicts disruption within a district) varies as a function of RGI. This is the classic contextual moderation design in multilevel analysis.

### 4.4 Propensity Score Matching (Robustness)

**The problem with regression-based causal claims**: Even with many controls, a logistic regression associating piped water with disruption cannot rule out unmeasured confounders. Perhaps piped-water households live in areas with worse electricity supply (which powers pumps) — if we do not control for electricity reliability, the piped coefficient is confounded.

**What PSM does**: PSM creates a matched sample where piped-water (treatment) and tube-well (control) households look identical on all observable characteristics. Within this matched sample, the comparison between piped and tube-well disruption rates is closer to a causal estimate.

**Covariates used for matching**:
- `wealth_q_num`, `urban`, `hh_size`, `children_u5` — demographics
- `female_headed`, `sc_st_flag` — social
- `has_electricity`, `improved_sanitation` — infrastructure access
These are the observable factors that predict both water source choice AND disruption. By balancing on them, we remove their confounding.

**Propensity score**: A logistic regression predicting `treatment = piped` from the covariates. The predicted probability of being piped is the propensity score. Matching on the propensity score is equivalent to matching on all covariates simultaneously (under the balancing property of propensity scores).

**1:1 nearest-neighbour matching with caliper 0.05**: Each piped household is matched to the tube-well household with the closest propensity score, within 0.05 of a standard deviation (the caliper). The caliper prevents poor matches — if no tube-well household is within 0.05 of a piped household's propensity score, the piped household is excluded. This reduces sample size but improves match quality.

**Average Treatment Effect on the Treated (ATT)**: The ATT asks: "For piped-water households (the treated group), what is the average difference in disruption probability compared to what they would have experienced with tube wells?" This is the relevant policy parameter — it tells us the effect of piped water for the households that actually have piped water.

ATT = mean(disruption | piped, matched) − mean(disruption | tube well, matched)

**Bootstrap SE**: We resample from the matched pairs (with replacement) 500 times and recompute the ATT each time. The standard deviation of the bootstrap distribution is our standard error. The 95% CI covers 2.5th to 97.5th percentile of bootstrap ATTs.

**Interpretation**: If ATT = +11.2 pp (SE = 1.2 pp), we can say: "Switching a household from tube well to piped water, holding observable characteristics constant, increases its expected disruption probability by approximately 11 percentage points." The phrase "holding observable characteristics constant" is important — PSM only controls for what we can measure.

---

## Part 5: Aggregation and Seasonal Adjustment

### 5.1 Why add pct_monsoon to the RGI model

NFHS-5 was conducted over two years (2019–21), not all at once. Some districts were surveyed predominantly during the monsoon season (June–September), when water disruptions from flooding and contamination are more common. Other districts were surveyed during the dry season, when supply shortfalls dominate.

If we do not control for when the district was surveyed, a district surveyed in July will appear to have higher disruption than an identical district surveyed in February — purely because of survey timing, not infrastructure quality. This would inflate the RGI for monsoon-surveyed districts and deflate it for dry-season-surveyed districts.

`pct_monsoon` = weighted fraction of households in the district surveyed in months 6, 7, 8, or 9. Adding this to the OLS removes the survey-timing artifact from the RGI residual.

### 5.2 Why add C(season) to the IDI Monte Carlo regression

Same logic applied at the household level. The Monte Carlo regression inside `run_monte_carlo` estimates OR(piped) in each of 500 runs. If we do not control for season, the OR distribution includes variation from survey timing rather than just from the IDI scoring uncertainty. Adding `C(season)` as a categorical control removes this source of noise.

This also absorbs the old standalone "seasonal analysis table" from script-d.py — we no longer need a separate seasonal disruption table because seasonal variation is controlled for in both major models.

---

## Part 6: Validation and Diagnostics

### 6.1 IDI Validation

**Predictive validity (Pearson r and AUC)**: We correlate IDI with observed disruption. Pearson r > 0 and p < 0.05 means IDI is significantly associated with the outcome we designed it to predict. AUC > 0.60 means IDI can discriminate disrupted from non-disrupted households better than chance. AUC = 0.50 is chance; AUC = 1.0 is perfect discrimination.

**Discriminant validity (r with wealth score)**: We check that IDI is not simply a wealth proxy. If |r(IDI, wealth)| > 0.80, the index adds no information beyond wealth — we would be better off just using the wealth score. We require |r| < 0.50.

**Known-groups validity**: We compare mean IDI for piped vs tube-well households. Since we built IDI partly around the piped/non-piped distinction, piped households should have higher IDI. If mean IDI(piped) < mean IDI(tube well), something is wrong with the scoring logic.

**Cronbach's alpha**: As described earlier, we want α > 0.60 to justify combining the dimensions.

### 6.2 RGI Diagnostics

**OLS R²**: Higher is better — it means the socioeconomic predictors explain more of the disruption variation, leaving a cleaner residual for RGI. R² around 0.5 is reasonable; very low R² (e.g., 0.1) would mean RGI is mostly noise.

**Moran's I**: Significant positive Moran's I validates that RGI clusters geographically (expected if it captures real infrastructure quality variation). Non-significant Moran's I would suggest RGI is geographically random — potentially just noise.

**Bootstrap CI width**: Narrow CIs (e.g., ±3 pp) indicate that RGI estimates are stable across samples. Wide CIs (e.g., ±15 pp) in a district with small sample size flag that the RGI should be interpreted cautiously for that district.

---

## Summary Table: Every Output and What It Answers

| Output | File | Answers |
|---|---|---|
| Table 1a–1e | tables/ | Does the paradox hold across strata? |
| Table 2 (IDI logit) | results/ | What is the OR for piped×IDI interaction? |
| Table 3 (crisis districts) | tables/ | Which districts are CRISIS? |
| Table 4 (state ranking) | tables/ | Which states are failing most? |
| Table 5 (GEE) | results/ | Does IDI×RGI interaction hold multilevel? |
| Table 6a (stage1 slopes) | results/ | How does IDI predict disruption in each district? |
| Table 6b (stage2 WLS) | results/ | Does RGI explain heterogeneity in IDI slopes? |
| Table 7 (PSM ATT) | results/ | What is the causal effect of piped water? |
| idi_monte_carlo_summary.csv | results/ | Is the piped OR robust to scoring uncertainty? |
| idi_validation.csv | results/ | Does IDI pass validation checks? |
| district_rgi_summary.csv | tables/ | GIS-ready district RGI for mapping |
| rgi_moran_test.csv | results/ | Does RGI cluster geographically? |
| water_paradox_report.md | / | Full research paper (auto-generated) |
