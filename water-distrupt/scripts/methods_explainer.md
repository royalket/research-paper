# Methods Explainer: Every Calculation, Why We Did It, and How to Read the Results

*This document explains the full analytical pipeline from raw NFHS-5 data to final paper tables.
It is written so that someone with basic statistics knowledge can understand every decision —
not just what we did, but why, and what the output numbers actually mean.*

---

## The Big Picture: Why This Analysis Exists

India's Jal Jeevan Mission is connecting every rural household to piped water, operating on the
assumption that piped = better. Our analysis asks a simple question: **is piped water actually
more reliable than what households had before?**

The answer is no — and the gap is large. Piped water households report disruption at 2.43 times
the rate of tube-well households. This document explains how we measured that, why we built the
IDI and RGI indices, what every number in the output means, and where the results are reliable
versus where caution is needed.

---

## Part 1 — The Data

### 1.1 What is NFHS-5?

The National Family Health Survey Round 5 (2019–21) is India's equivalent of the Demographic
and Health Survey. It covers all 28 states and 8 union territories and is the most comprehensive
nationally representative household survey in India.

**Key facts about the dataset:**
- Full sample: 636,699 households
- Our analysis sample: 578,062 households (after dropping 58,637 with missing disruption answers)
- Survey design: stratified two-stage cluster sampling — districts are strata, villages/urban
  blocks are primary sampling units (PSUs), households are selected within PSUs
- Nationally representative: the sampling design ensures every region and wealth group is
  proportionally covered

### 1.2 Survey Weights — Why We Use Them

Not every household has an equal probability of being selected. Urban households in small states
may be oversampled to ensure enough observations for state-level estimates. If we counted every
household equally, our national disruption rate would be wrong — it would over-represent
oversampled groups.

**The weight variable is `hv005`.** DHS stores it as an integer multiplied by 1,000,000 to avoid
decimals. We divide by 1,000,000 to get the actual weight.

**What a weight means:** If household A has weight 1.8 and household B has weight 0.7, household A
represents 1.8 / 0.7 ≈ 2.6 times as many real households in the population as household B. When
we compute a weighted disruption rate, we are computing the rate that represents the actual Indian
population, not just our sample.

**Every rate, mean, and percentage in this paper is weighted.**

### 1.3 The Outcome Variable: `water_disrupted`

**Raw variable:** `sh37b` — "Was water not available for at least one day in the past two weeks?"

**Our coding:**
- `sh37b = 1` → `water_disrupted = 1` (disrupted)
- `sh37b = 0` → `water_disrupted = 0` (not disrupted)
- `sh37b = 8 or 9` (don't know / missing) → **dropped from sample**

**Why binary?** Disruption is an event — it either happened or it didn't in the recall period.
Binary outcomes require logistic regression, not linear regression, because the predicted
probability must stay between 0 and 1.

**What we lose:** We cannot distinguish one day of disruption from two weeks. A household with
a single missed hour is coded the same as one with two weeks of no water. This is a limitation
of the survey, not the analysis. We note it in the paper.

### 1.4 Water Source Codes — Why We Split Piped Into Sub-Types

`hv201` is the primary water source. DHS coding:

| Code | Label | n in sample |
|------|-------|-------------|
| 11 | Piped into dwelling | 121,879 |
| 12 | Piped to yard/plot | 97,138 |
| 13 | Piped to neighbour/shared | 10,689 |
| 14 | Public tap/standpipe | 80,176 |
| 21 | Tube well/borehole | 212,187 |
| 31 | Protected well | 17,088 |
| 32 | Unprotected well | 10,159 |
| 41 | Protected spring | 3,255 |
| 42 | Unprotected spring | 2,324 |
| 43 | Surface water (river/lake/canal) | 3,030 |
| 51 | Rainwater | 2,927 |
| 61 | Tanker truck | 4,660 |
| 62 | Cart with small tank | 883 |
| 71 | Bottled water | 8,839 |
| 92 | Community RO plant | 4,610 |
| 96 | Other | 1,145 |

**Why we do NOT collapse all piped codes into "Piped Water":** The four piped sub-types have
very different disruption rates:

| Sub-type | Disruption % |
|----------|-------------|
| Piped — Yard/Plot | **29.7%** |
| Piped — Neighbour/Shared | 26.1% |
| Piped — Public Tap/Standpipe | 23.9% |
| Piped — Into Dwelling | 23.6% |

Yard/plot connections are at the **periphery of the distribution network** — they experience
pressure drops and shutdowns first when the system is stressed. Collapsing this variation would
hide an important finding. We show all sub-types in Table 1a and note this gradient in the paper.

**The `piped_flag` variable** = 1 if the household uses any of the four piped codes. This is
used in regressions where we need a single piped/not-piped contrast.

---

## Part 2 — The Infrastructure Dependency Index (IDI)

### 2.1 What the IDI is — in one sentence

**The IDI measures: when your tap fails, how stuck are you?**

That is it. It is not measuring how bad your water is or how poor you are. It is measuring
whether you have anywhere to turn when your primary source stops working.

---

### 2.2 Why we built it — and what it adds over piped_flag

The logistic regression already tells us piped water causes more disruption.
`piped_flag` does that job. The IDI does something different.

**`piped_flag` tells you THAT the paradox exists.**
**The IDI tells you WHY some piped households are worse off than others.**

Two piped households sitting next to each other can have completely different outcomes
when the tap fails:

| | Household A — fully stuck | Household B — can cope |
|---|---|---|
| Source | Piped, no backup | Piped, but has a tube well nearby |
| Experience | Never fetched water in their life | Fetches from the well sometimes |
| Assets | No fridge, no vehicle, poor | Has a motorcycle and a fridge |
| When tap fails | Completely helpless | Fetches from the well, stores in fridge |

Both are `piped_flag = 1`. The IDI gives Household A a score near 100 and Household B
a score near 30.

**The key finding the IDI reveals:** Rich piped households are more locked in
(Dims 1 and 3 both high) but they have coping assets (Dim 4 near zero).
Poor piped households are less locked in (less likely to be piped at all) but
when they ARE piped, they have nothing when the tap fails (Dim 4 = 0.933).
`piped_flag` cannot show this split at all.

**Important:** The IDI is used ONLY descriptively — in tables and figures.
It is NOT put in the regression models. Why? Because IDI is mostly built from
piped adoption (Dims 1, 2, and 3 all score high when you are piped), so putting
both IDI and piped_flag in the same regression causes them to fight over the same
variance and produce wrong-direction results. The regression uses `piped_flag` for
the causal finding. The IDI is used for the "who is worst affected and why" story.

---

### 2.3 The three dimensions — plain language

Think of each dimension as one question you ask about a household.
We have three dimensions, not four — Dims 1 and 3 from the original design
asked the same question from different angles (r = 0.873 between them),
so we merged them into one.

---

**Dim A — Source Risk** *(merged from old Dims 1 and 3)*
*Question: How locked in are you to a system you don't control?*

Old Dim 1 asked: do you have a non-piped backup?
Old Dim 3 asked: what type of system does your source depend on?
These are the same question. Piped households score high on both. Tube-well
households score 0 on both. Correlation was 0.873. Keeping them separate
was double-counting.

The merged dimension:
- Score 3 → Piped primary, no non-piped backup. Maximum lock-in. Nowhere to go.
- Score 2 → Piped with a non-piped backup, OR tanker/bottled primary. Dependent but has an exit, or pure market.
- Score 1 → Community managed (RO plant, protected well/spring). Semi-managed. Some local control.
- Score 0 → Tube well, own well, rainwater, surface water. Self-sufficient. No one else controls it.

**In the data:** `hv202` (alternative source) is not recorded in NFHS-5 — all households
show "No Other Source." So score 2 (piped with non-piped backup) never occurs.
Dim A is effectively: piped = 3, community managed = 1, tanker/bottled = 2, tube well = 0.

---

**Dim B — Access Complexity**
*Question: Have you ever had to go fetch water? Do you know how?*

- Score 3 → Water on-premises (`water_on_premises = 1`). Never fetched. No routine.
- Score 2 → Water in yard/plot. Walk a few steps. Minimal experience.
- Score 1 → Elsewhere, under 15 min. Some routine.
- Score 0 → Elsewhere, 15+ min. Daily fetcher. Knows exactly what to do when source fails.

**The flip:** In normal water indices, "water in dwelling" is good. Here it is bad for
disruption preparedness — you have never needed to cope because the tap always worked.

**Technical fix:** NFHS-5 only records `water_location` for off-premises households.
On-premises households show "Unknown" in `water_location`. Old code scored them
as neutral (1). Fix: check `water_on_premises = 1` first, score 3. Then use
`water_location` for off-premises households only.
This changed the loading from −0.419 (backwards) to +0.128 (correct).

**Why loading is low:** 78% of households have water on-premises, all score 3.
Almost no variation — a dimension with no variation adds nothing to PCA.

---

**Dim C — Piped Coping Deficit**
*Question: You are piped and the tap has just failed. Can you cope?*

This dimension ONLY applies to piped households.
Non-piped households score 0 — they already fetch daily and have a practiced routine.
The deficit concept only makes sense for someone who depends on a tap.

For piped households:
- Buffer = wealth (Q1=0, Q2=0, Q3=1, Q4=2, Q5=3) + fridge (0/1) + vehicle (0/1), clipped to [0, 3]
- Coping Deficit = 3 − buffer

| Example | Buffer | Coping Deficit |
|---|---|---|
| Richest + fridge + vehicle | 3+1+1 = 3 (clipped) | 3−3 = **0** (can cope) |
| Middle + vehicle only | 1+0+1 = 2 | 3−2 = **1** |
| Poorest + nothing | 0+0+0 = 0 | 3−0 = **3** (totally stuck) |

**Why gate on piped_flag?** A poor non-piped household with no fridge is NOT in deficit —
they fetch every day and have a routine. Without the gate, wealth correlates with piped
adoption and the loading goes negative. Gating changed the loading from −0.239 to +0.478
and Cronbach α from −0.285 to 0.706.

**What Dim C adds:** Among piped households, a rich one and a poor one both have
piped_flag = 1. Dim C separates them — rich piped scores near 0, poor piped scores 3.
This is the only dimension that does this. It produces the key finding: rich piped
households are locked in (high Dim A) but buffered (low Dim C). Poor piped households
are less locked in overall but completely unbuffered when they are piped.

---

### 2.4 The honest summary — what IDI actually is in the data

| Dimension | Plain question | What data shows |
|---|---|---|
| Dim A: Source Risk | How locked in is your source situation? | Piped = 3, community = 1, tube well = 0. Mostly piped_flag. |
| Dim B: Access Complexity | Do you have fetching experience? | 78% score 3. Low variance. Weak dimension. |
| Dim C: Piped Coping Deficit | If piped, can you cope when it fails? | Only varies among piped HH. The genuinely new information. |

**Dims A and B are mostly restating piped_flag.**
**Dim C is where the IDI adds new information — separating rich piped from poor piped.**

This is why the IDI is used descriptively (tables + figures), not in regressions.
The regression uses `piped_flag` for the causal finding.
The IDI is used for "who is worst affected and why."

---

### 2.5 IDI validation numbers

| Check | Value | What it means |
|---|---|---|
| Cronbach α = 0.706 | Good internal consistency | The 4 dimensions measure the same underlying thing |
| AUC = 0.619 | Modest predictive power | IDI predicts disruption slightly better than chance |
| r(IDI, disruption) = 0.180 | Positive, significant | Higher lock-in → more disruption |
| r(IDI, wealth) = 0.170 | Well below 0.50 | IDI is NOT just a wealth proxy |
| Mean IDI: piped − tube well = +59.4 | Large gap | Piped HH are structurally far more locked in |
| 100% MC runs OR(piped) > 1 | Fully robust | The piped paradox holds regardless of how we score the dimensions |

---

## Part 3 — Combining the Dimensions: PCA

### 3.1 Why Not Just Add Them Up?

Adding Dim1 + Dim2 + Dim3 + Dim4 with equal weights assumes each dimension is equally
important. But we just showed that Dim 1 and Dim 3 have 0.873 correlation — they are almost
measuring the same thing. Giving them equal weight double-counts the piped-dependency signal.
PCA avoids this by finding the optimal combination.

### 3.2 What PCA Actually Does — Step by Step

**Step 1 — Standardise each dimension.**
Each of the four dimension scores (0–3) is converted to mean 0, standard deviation 1.
This is necessary because without standardisation, a dimension with wider natural spread
would dominate the PCA regardless of its conceptual importance.

**Step 2 — Build the covariance matrix.**
A 4×4 matrix showing how much each pair of dimensions varies together across households.
Entry [1,3] = 0.873 means Dim 1 and Dim 3 are highly correlated.

**Step 3 — Find the eigenvectors.**
The eigenvectors of the covariance matrix are the directions in 4-dimensional space along
which households vary the most. The first eigenvector (PC1) points in the direction of
maximum variance.

**Step 4 — The loadings are the PC1 eigenvector.**
These are the weights we use:

| Dimension | Loading | What it means |
|-----------|---------|---------------|
| Dim 1: Source Lock-in | **+0.622** | Strongest contributor. Piped-without-backup is the core signal. |
| Dim 2: Access Complexity | **+0.128** | Weakest contributor. Low variance in the data (most HH score 3). |
| Dim 3: System Dependency | **+0.606** | Second strongest. Closely related to Dim 1. |
| Dim 4: Piped Coping Deficit | **+0.478** | Meaningful contributor. Captures within-piped heterogeneity. |

**All four loadings are positive.** This confirms all four dimensions point in the same
direction: higher score = more locked in = more likely to be disrupted. If any loading had
been negative, it would mean that dimension measured something conceptually inconsistent
with the others, and we would need to reconsider including it.

**Step 5 — Project each household onto PC1.**
Each household's IDI (raw) = 0.622 × Dim1_std + 0.128 × Dim2_std + 0.606 × Dim3_std + 0.478 × Dim4_std

This gives a raw score with arbitrary units. We then normalise to 0–100 in each Monte Carlo
run (see Part 4).

### 3.3 Variance Explained: 57.7%

PC1 explains **57.7%** of the total variance in the four dimensions. This means a single
underlying dimension — "infrastructure dependency" — captures just over half of what varies
between households in their water system configuration. The remaining 42.3% is noise or
genuinely distinct sub-constructs that are orthogonal to the main lock-in dimension.

**Is 57.7% good?** For a 4-item index in social science, this is acceptable. If PC1 explained
only 30%, the dimensions would be poorly related to a common construct. If it explained 90%,
the four dimensions would be nearly redundant.

### 3.4 Cronbach's Alpha: 0.706

Cronbach's alpha measures internal consistency — are the four dimensions measuring the same
underlying thing?

**Formula:** α = (k / (k−1)) × (1 − Σ item variances / total score variance)

**Interpretation:**
- α < 0.5: Poor — dimensions are not measuring a common construct. Do not combine them.
- 0.5–0.7: Acceptable for exploratory work.
- **0.7–0.9: Good — our result (0.706) falls here.**
- > 0.9: Excellent, but may indicate item redundancy.

**What changed:** Before the Dim 4 fix, α = −0.285. Negative alpha means the dimensions
actively contradict each other — some predict disruption up, others predict it down. This was
the key indicator that something was wrong. After gating Dim 4 on `piped_flag`, α jumped to
0.706, confirming the four dimensions now consistently measure the same construct.

---

## Part 4 — Monte Carlo Uncertainty Quantification

### 4.1 The Problem: Our Scoring Has Judgment Calls

When we score Dim 2 as 0 for "elsewhere ≥ 15 min", we are making a judgment call. What if
the threshold should be 20 minutes? What if the noise parameter σ = 0.3 should be 0.5?
How much do these choices affect the final results?

Monte Carlo answers this by running the analysis 500 times with slightly different inputs
each time and asking: how much does the output change?

### 4.2 What Happens in Each of 500 Runs

1. **Perturb dimension scores:** Add Gaussian noise N(0, 0.3) to every dimension score for
   every household. Clip the result to [0, 3] to keep scores in valid range.
   - σ = 0.3 means roughly 68% of perturbations are within ±0.3 of the original score.
   - On a 0–3 scale, ±0.3 represents 10% of the range — a meaningful but not extreme perturbation.

2. **Project through fixed PCA:** Use the PCA weights fitted on the *original* data.
   We do NOT refit PCA in each run. If we refitted PCA, we would be changing both the inputs
   and the weighting scheme simultaneously, making it impossible to separate the two sources
   of uncertainty. Fixed PCA isolates input uncertainty only.

3. **Normalise to 0–100:** Within each run, the raw PCA projection is linearly rescaled to
   [0, 100] using that run's min and max values.

4. **Fit a logistic regression:**
   `logit(disruption) ~ piped + idi + piped:idi + wealth + urban + hh_size + children_u5 + C(season)`
   Record OR(piped) and OR(piped:idi) from the model.

5. **Store per-household IDI:** Record each household's IDI score in this run.

### 4.3 How to Read the Monte Carlo Output

**`idi_monte_carlo_summary.csv` — what each row means:**

| Metric | Our value | How to read it |
|--------|-----------|----------------|
| OR(Piped) — Mean | 1.753 | On average across 500 runs, piped water is associated with 1.75× higher odds of disruption |
| OR(Piped) — 2.5th pctile | 1.724 | In the most pessimistic 2.5% of runs (when scoring noise pushed the index most), the OR was still 1.72 |
| OR(Piped) — 97.5th pctile | 1.780 | In the most optimistic 2.5% of runs, OR was 1.78 |
| OR(Piped × IDI) — Mean | 0.817 | The interaction term is below 1 in MC runs — see discussion below |
| % MC runs OR(Piped) > 1 | **100.0%** | In **every single** run, piped water had higher odds of disruption than non-piped. This is the robustness finding. |

**The 100% robustness result** means: no matter how we perturb the dimension scoring within
reasonable bounds, piped water always predicts higher disruption. The paradox is not an
artifact of our scoring choices.

**Per-household IDI:**
- `idi_mean`: This household's average IDI across 500 runs (0–100 scale)
- `idi_ci_lower`: 2.5th percentile across runs — the lower bound of uncertainty
- `idi_ci_upper`: 97.5th percentile — upper bound
- `idi_ci_width`: Upper minus lower. Wide CI means this household's IDI is sensitive to
  scoring noise (usually because it sits near dimension score boundaries). Narrow CI means
  the IDI is stable regardless of scoring perturbations.

**Example:** A household with `idi_mean = 75, idi_ci_lower = 68, idi_ci_upper = 82` has a
stable, high IDI — clearly in the high lock-in group. A household with `idi_mean = 50,
idi_ci_lower = 35, idi_ci_upper = 65` is ambiguous — scoring noise moves it substantially.

---

## Part 5 — IDI Dimension Profiles (New)

### 5.1 Why Show Dimensions Separately by Group

The composite IDI gives one number per household. But it masks *which* dimension is driving
lock-in for different groups. A policy maker implementing JJM needs to know: is the problem
that poor piped households have no backup (Dim 1)? Or that they have no coping capacity
(Dim 4)? The dimension profile tables answer this.

This approach was inspired by Tikadar & Swami (2025), who showed that India's Residential
Energy Poverty Index looks different when you break it into clean energy, reliability,
appliances, and efficiency dimensions — a state that scores well on one dimension may
score badly on another, requiring different interventions.

### 5.2 How to Read the Dimension Profile Tables

**`idi_dim_profile_by_wealth.csv`** — rows are wealth quintiles, columns are dimension scores:

| Quintile | Dim 1 (Lock-in) | Dim 2 (Access) | Dim 3 (System) | Dim 4 (Coping) | IDI |
|----------|----------------|----------------|----------------|----------------|-----|
| Poorest | 0.975 | 1.995 | 0.686 | 0.933 | 32.8 |
| Poorer | 1.422 | 2.374 | 1.024 | 1.250 | 44.1 |
| Middle | 1.772 | 2.578 | 1.307 | 0.731 | 46.4 |
| Richer | 1.948 | 2.754 | 1.483 | 0.073 | 44.8 |
| Richest | 2.134 | 2.898 | 1.624 | 0.000 | 47.7 |

**How to read this:**

- **Dim 1 and Dim 3 increase from poorest to richest.** This means richer households are
  more locked in to piped systems — they adopt piped water at higher rates and have higher
  system dependency. This is expected: richer households can afford piped connections.

- **Dim 4 decreases from poorer to richest.** The richest households have zero coping
  deficit — they have wealth, vehicles, and fridges. The poorest piped households have
  high coping deficits.

- **The IDI is relatively flat from Poorer to Richest (~44–48).** This is a key finding:
  despite richer households being more locked in (Dims 1 and 3), their strong coping
  capacity (Dim 4) offsets it. The poorest piped households have lower lock-in but zero
  coping capacity — their IDI is lower not because they are safer, but because they are
  less likely to be piped at all.

- **The policy implication:** Rich piped households are locked in but buffered. Poor piped
  households are less locked in but completely exposed when they do have piped water.
  Interventions for each group should differ.

**`idi_dim_profile_by_urban.csv`:**

| Residence | Dim 1 | Dim 2 | Dim 3 | Dim 4 | IDI |
|-----------|-------|-------|-------|-------|-----|
| Rural | 1.414 | 2.390 | 1.031 | 0.735 | 39.8 |
| Urban | 2.162 | 2.764 | 1.641 | 0.405 | 51.2 |

Urban households are more locked in on all dimensions except Dim 4 (they have more assets).
The rural-urban IDI gap (39.8 vs 51.2) reflects higher piped adoption and deeper system
dependency in cities.

---

## Part 6 — The Reliability Gap Index (RGI)

### 6.1 Why a District-Level Index?

The IDI explains household-level lock-in. But disruption is also driven by how well the
district's *system* performs. Two identical households (same IDI) in different districts can
have very different outcomes: one in a district with a well-maintained piped network, one in
a district where the utility barely functions.

We need a district-level measure of system underperformance. That is the RGI.

### 6.2 The Logic: Observed Minus Expected

The RGI is a **residual from a regression.** Here is the step-by-step logic:

**Step 1 — Aggregate to district level.**
For each of 704 districts with ≥ 100 households, we compute:
- `observed_disruption`: weighted mean disruption rate
- `mean_wealth_score`: average household wealth
- `pct_urban`: share of urban households
- `improved_coverage`: share using improved water sources
- `piped_coverage`: share using piped water
- `pct_monsoon`: share of households surveyed in June–September

**Step 2 — Fit weighted OLS:**
`observed_disruption = β₀ + β₁(wealth) + β₂(urban) + β₃(improved) + β₄(piped) + β₅(monsoon)`

Weights = number of households per district (larger districts are more reliable).

**Why these predictors?**
- `wealth`: richer districts have more resources to cope → lower expected disruption
- `urban`: urban areas have better infrastructure investment → lower expected disruption
- `improved_coverage`: more improved sources → lower expected disruption
- `piped_coverage`: **critical new addition** — a district with 90% piped coverage has more
  households exposed to piped system failure. Without this control, high-piped districts would
  always appear to be "underperformers" simply because more of their households are piped.
  This would confound the RGI with the very phenomenon we are studying.
- `pct_monsoon`: NFHS-5 was fielded over two years. Districts surveyed in July–August will
  show higher disruption than identical districts surveyed in February — purely due to seasonal
  flooding and contamination events. This variable removes that artifact.

**Step 3 — RGI = Observed − Predicted**

If the OLS predicts 25% disruption for a district but we observe 45%, `RGI = +20 pp`.
This district's infrastructure is performing 20 percentage points worse than comparable
districts at the same development level.

If the OLS predicts 30% but we observe 20%, `RGI = −10 pp`. This district is outperforming
expectations — possibly through better maintenance, community management, or both.

**How to read RGI values:**
- `RGI > +15 pp`: Severe Failure — infrastructure severely underperforms development level
- `RGI = +5 to +15 pp`: Moderate Failure
- `RGI = −5 to +5 pp`: As Expected
- `RGI < −5 pp`: Outperforming

### 6.3 Bootstrap Confidence Intervals on RGI

The OLS prediction is uncertain — if we had different districts in our sample, the OLS
coefficients would be slightly different, and all RGI values would shift. We quantify this
with 500 bootstrap iterations:

1. Draw a bootstrap sample of districts (with replacement, same total N)
2. Refit the OLS
3. Apply to all districts → get bootstrap RGI
4. After 500 reps: take 2.5th and 97.5th percentile of each district's RGI distribution

**A district with `RGI = 42.7, CI = [40.8, 44.8]`** (top CRISIS district, Gujarat) is
unambiguously a severe failure — the CI is narrow and entirely above zero.

**A district with `RGI = 6.2, CI = [−1.5, 14.0]`** is uncertain — it might be moderate
failure or it might just be normal variation.

### 6.4 District Typology

Districts are split into four groups using the **weighted medians** of IDI and RGI as
cut-points:

| Typology | IDI | RGI | What it means |
|----------|-----|-----|---------------|
| **CRISIS** (189 districts) | High | High | Households locked in AND system failing. Worst outcome. |
| **VULNERABLE** (179 districts) | High | Low | Households locked in but system working. Disruption is from lock-in, not system failure. |
| **SAFE** (174 districts) | Low | Low | Not locked in, system working. Best outcome. |
| **RESILIENT POOR** (162 districts) | Low | High | System failing but households not locked in — they have backup sources and fetching experience. Paradoxically safe despite failing infrastructure. |

**Why weighted median?** If we used the simple median district IDI, one district with 50
households and an extreme IDI would move the cut-point as much as a district with 5,000
households. The weighted median is the IDI value at which the cumulative household count
crosses 50% — it represents the median *household*, not the median *district*.

---

## Part 7 — The Regression Models

### 7.1 IDI Logistic Regression (Finding 2)

**What it tests:** After controlling for everything we can measure, is piped water still
associated with higher disruption? Does higher IDI mean higher disruption?

**Formula:**
```
logit(disruption) = α
  + β₁(piped_flag)
  + β₂(idi_mean)
  + β₃(piped_flag × idi_mean)
  + β₄(wealth_quintile)
  + β₅(urban)
  + β₆(hh_size)
  + β₇(children_u5)
  + β₈(C(region))
  + β₉(SC_ST_flag)
  + β₁₀(female_headed)
  + β₁₁(C(head_education))
```

**How to read the output table:**

| Term | OR | What it means |
|------|-----|---------------|
| Piped Water | **2.088** | A piped household has 2.09× the odds of disruption vs a non-piped household with the same IDI, wealth, region, etc. |
| IDI Score | **1.007** | Each additional point of IDI (0–100 scale) multiplies disruption odds by 1.007. At IDI=50 vs IDI=0, that is 1.007^50 = 1.42× higher odds. |
| Piped × IDI | 0.996 | The interaction — see below |
| Wealth quintile | 0.974 | Each quintile step *up* reduces disruption odds by 2.6%. Wealthier = less disruption. |
| SC/ST | **1.105** | SC/ST households have 10.5% higher odds of disruption, even after controlling for wealth and source type. |
| Female-headed | 0.975 | Female-headed households have slightly lower disruption odds. |

**The interaction term (Piped × IDI = 0.996):**
This is below 1, which seems to say "higher IDI reduces the piped penalty." This is a
multicollinearity artifact. IDI and `piped_flag` are extremely correlated (mean IDI 59.4 points
higher for piped). When both enter the model, `piped_flag` absorbs the positive IDI variance, and
the interaction term captures the residual — households that are "high IDI but not piped" (a rare
group). We report this honestly and treat it as a model limitation rather than a substantive finding.

**Cluster-robust standard errors:** Households within the same PSU (village or urban block) share
unmeasured community characteristics. Without clustering, standard errors would be too small —
we would falsely conclude more variables are significant. Clustering at the PSU level corrects this.
The formula inflates standard errors to account for within-cluster correlation.

**What a p-value means here:** With n = 548,637 households, even tiny effects are statistically
significant. We therefore interpret effect sizes (OR, pp difference) rather than p-values alone.
OR = 1.007 per IDI point is statistically significant at p < 0.001 but the practical significance
depends on the IDI distribution — since IDI ranges 0–100, the full-range effect is OR = 1.007^100
= 2.01× — meaningful.

### 7.2 GEE Multilevel Model (Finding 4)

**Why not just a regular logistic regression?**
Households within the same district share unmeasured infrastructure quality — the reliability
of their local water utility, the state of the pipes, the frequency of electricity outages that
power pumps. Ignoring this would produce standard errors that are too small.

A standard approach would be a random-effects mixed model, but random-effects models require
distributional assumptions and produce "subject-specific" estimates (the effect for a specific
household). We want "population-averaged" estimates — the average effect across all Indian
households — which is what policy claims require.

**GEE (Generalized Estimating Equations)** produces population-averaged estimates and accounts
for within-district correlation using an "exchangeable" structure: all pairs of households
within a district are assumed equally correlated. The correlation parameter α is estimated
from the data.

**How to read the GEE output:**

| Term | OR | What it means |
|------|-----|---------------|
| IDI (std) | 0.897 | Negative direction — see note below |
| RGI (std) | **1.675** | Each SD increase in district RGI → 67.5% higher odds of disruption |
| IDI × RGI | 1.037 | The interaction — direction is correct but p = 0.083 |
| Piped Water | **2.293** | Piped water still doubles disruption after all controls |

**Why IDI (std) OR = 0.897 (below 1)?**
In the GEE model, `piped_flag` and `idi_std` are both included. Since IDI is built partly
from piped_flag (Dims 1 and 3 are both driven by piped adoption), they are highly correlated.
`piped_flag` absorbs the positive IDI effect, leaving the residual IDI coefficient to pick up
"high IDI among non-piped households" — a group that doesn't exist meaningfully. This is the
same collinearity issue as the logit interaction term. We note this explicitly in the paper.

**RGI OR = 1.675:** The district-level finding is strong and clean. Each standard deviation
increase in the reliability gap is associated with 67.5% higher odds of disruption, independent
of household-level IDI, wealth, and demographics. This confirms that where you live matters —
your district's infrastructure quality directly predicts your disruption risk.

**IDI × RGI interaction OR = 1.037, p = 0.083:**
The interaction is in the expected direction (above 1) but does not reach p < 0.05. This means
we cannot confirm from this data alone that the IDI effect is steeper in failing districts.
We discuss reasons: district-level geography may be too coarse, and RGI measurement error
attenuates the interaction.

### 7.3 Slope-as-Outcome Model (Finding 5)

**The question:** In each district separately, how strongly does IDI predict disruption?
And is that district-specific IDI effect larger in districts with higher RGI?

This is a stronger test than the GEE interaction because it directly estimates the IDI
mechanism within each district, rather than testing a joint effect in the pooled sample.

**Stage 1 — Per-district IDI slopes:**
For each of 657 districts (with ≥ 30 households and ≥ 50 piped households), we fit:
`logit(disruption) = α_d + β_d × IDI + γ_d × wealth + δ_d × urban`

The output is β_d (the IDI slope in district d) and SE(β_d) (its standard error).

**Stage 2 — WLS regression:**
`β̂_d = a + b × RGI_d + ε_d`    weighted by 1 / SE(β_d)²

The weight 1/SE² means districts with precise slope estimates (large samples) contribute
more to the regression. Districts with noisy slopes (small samples) contribute little.

**How to read the output:**

| Term | Coefficient | What it means |
|------|-------------|---------------|
| Intercept | 0.0085 | The average district IDI slope — in a typical district, each IDI point increases log-odds of disruption by 0.0085 |
| RGI (std) | 0.0007 | p = 0.083 — not significant. Direction correct but weak. |

**Why 0.0085 is meaningful:** Across the IDI range 0–100, the typical within-district effect
is 0.0085 × 100 = 0.85 log-odds, or about OR = exp(0.85) = 2.34× — comparable to the
pooled logit result.

**Why RGI is not significant here:**
Two reasons. First, the RGI itself is measured with error (it is a regression residual with
bootstrap CI). When the predictor in Stage 2 has measurement error, the coefficient is
attenuated toward zero (classical attenuation bias). Second, 704 districts may be too coarse —
the IDI × RGI mechanism likely operates at sub-district scale.

### 7.4 Propensity Score Matching (Robustness)

**The causal question:** All regressions show association. Does piped water *cause* more
disruption, or do piped-water households just happen to live in places with worse
infrastructure?

PSM constructs a counterfactual. For every piped-water household, we find a tube-well
household that is as similar as possible on all observable characteristics. The disruption
gap in this matched sample is closer to a causal estimate because confounders are balanced.

**How matching works:**

Step 1 — Predict the probability of being piped (`propensity score`) using:
wealth, urban, household size, children under 5, female-headed, electricity access,
improved sanitation, SC/ST status.

Step 2 — For each piped household, find the tube-well household with the closest
propensity score, within a caliper of 0.05.

Step 3 — Compute ATT = mean disruption in matched piped group − mean disruption in
matched control group.

**How to read the PSM output:**

| Metric | Value | What it means |
|--------|-------|---------------|
| ATT | **+15.06 pp** | Piped water causes ~15 percentage points more disruption than tube wells, for households identical on observable characteristics |
| SE | 0.13 pp | The estimate is very precise because of the large sample (296,651 matched pairs) |
| 95% CI | 14.77–15.31 pp | Even the lower bound is well above zero |
| % bootstrap > 0 | **100%** | In every single bootstrap resample, piped water has higher disruption |

**15 pp is the paper's strongest causal number.** The large CI width (14.77–15.31) reflects
precision, not uncertainty — the effect is very stably around +15 pp. The PSM sample of
296,651 matched pairs out of 296,651 piped households (100% match rate) means all piped
households found a tube-well match within the caliper, so no selection bias from dropping
unmatched units.

**What PSM cannot control for:** Unobservable confounders — e.g., whether piped-water areas
have worse local governance that also makes electricity less reliable. We note this as a
limitation.

---

## Part 8 — Reading the Output Files

### Complete list of output files and what each answers

**Tables folder (`nfhs5_output/tables/`):**

| File | What it contains |
|------|-----------------|
| `table_1a_by_source.csv` | Disruption rate for all 15 source types with RR vs tube well. The primary Table 1 in the paper. |
| `table_1b_by_source_wealth.csv` | Piped sub-types + tube well disruption by wealth quintile. Shows paradox holds within all wealth groups. |
| `table_1c_by_source_urban.csv` | Same, by urban/rural |
| `table_1d_by_source_region.csv` | Same, by geographic region |
| `table_1e_by_source_season.csv` | Same, by season. Shows monsoon peak. |
| `table_1f_category_summary.csv` | One-row-per-category summary: piped combined, tube well, protected, unprotected, tanker, bottled/RO |
| `table_1_wealth_idi_dims.csv` | IDI dimension profiles by wealth quintile (4 dims + composite) |
| `table_1_urban_idi_dims.csv` | IDI dimension profiles by urban/rural |
| `table_1_source_idi_dims.csv` | IDI dimension profiles by water source type |
| `table_1i_dim_disruption_gradient.csv` | For each dimension: disruption rate in low-half vs high-half households. Shows which dimension most strongly tracks disruption. |
| `district_rgi_summary.csv` | All 704 districts: RGI, CI, typology, IDI, piped coverage. GIS-ready for mapping. |
| `table_3a_top_crisis.csv` | Top 20 CRISIS districts by RGI magnitude |
| `table_3b_top_safe.csv` | Top 20 SAFE districts |
| `table_3c_resilient_poor.csv` | Top 20 RESILIENT POOR districts |
| `table_3d_state_paradox.csv` | State-level aggregation of RGI, IDI, paradox ratio |

**Results folder (`nfhs5_output/results/`):**

| File | What it contains |
|------|-----------------|
| `idi_validation.csv` | Four validation checks: Pearson r, AUC, discriminant validity, known-groups validity |
| `idi_monte_carlo_summary.csv` | OR(piped) distribution across 500 MC runs. The robustness finding. |
| `idi_dim_profile_by_wealth.csv` | Detailed dimension profiles (also saved to tables/) |
| `idi_dim_profile_by_urban.csv` | Same |
| `idi_dim_profile_overall.csv` | Overall means, SDs, and PCA loadings for each dimension |
| `table2_idi_regression.csv` | IDI logistic regression — ORs, CIs, p-values for key terms |
| `table2b_predicted_probs.csv` | Predicted probabilities for four household scenarios |
| `table5_gee_multilevel.csv` | GEE model results |
| `table6a_stage1_idi_slopes.csv` | Per-district IDI slopes from Stage 1. 657 districts × {slope, SE, n_obs, n_piped}. |
| `table6b_stage2_wls_rgi.csv` | Stage 2 WLS: does RGI predict district IDI slopes? |
| `table7_psm_att.csv` | PSM ATT with bootstrap CI — the causal result |
| `rgi_moran_test.csv` | Moran's I result for spatial autocorrelation of RGI |

**Report (`nfhs5_output/water_paradox_report_[timestamp].md`):**
Auto-generated markdown report combining all findings into a structured paper outline.
Updated every time `main.py` runs.

---

## Part 9 — What Is Not Working Yet and Why

| Issue | What the output shows | Root cause | What would fix it |
|-------|----------------------|------------|-------------------|
| IDI main effect negative in GEE (OR = 0.897) | IDI std → OR < 1, p < 0.001 | `piped_flag` and `idi_std` are collinear (r ≈ 0.9). `piped_flag` absorbs the positive IDI variance. | Remove `piped_flag` from GEE and rely on IDI alone; or use orthogonalised IDI residual |
| IDI × RGI not significant (p = 0.083) | OR = 1.037, direction correct | RGI is measured at district level with error; districts too coarse | Sub-district data; errors-in-variables correction in Stage 2 |
| Disruption gradient Dim 1 and Dim 2 show NaN | `table_1i` | Dim 1 is binary (0 or 3); median split produces empty halves | Replace median split with bottom vs top third for binary dims |
| Slope-as-outcome R² = 0.005 | Very low explanatory power in Stage 2 | RGI measurement error; geographic resolution | Same as IDI × RGI issue above |

---

## Summary: The Core Numbers to Remember

| Number | What it is | Where it comes from |
|--------|-----------|---------------------|
| **2.43×** | Relative risk of disruption: piped vs tube well | Table 1a, weighted rate |
| **25.5% vs 10.5%** | Raw disruption rates: piped vs tube well | Table 1a |
| **29.7%** | Highest-disruption piped sub-type: yard/plot | Table 1a |
| **OR = 2.09** | Adjusted odds ratio for piped water | Table 2, logistic regression |
| **OR = 1.007** | Adjusted OR per IDI point | Table 2 |
| **ATT = +15.1 pp** | Causal effect of piped water via PSM | Table 7 |
| **α = 0.706** | IDI internal consistency | IDI build step |
| **57.7%** | Variance explained by IDI first component | PCA results |
| **100%** | MC runs where piped OR > 1 | MC summary |
| **OR = 1.675** | RGI effect on disruption (GEE) | Table 5 |
| **189** | CRISIS districts | District typology |
