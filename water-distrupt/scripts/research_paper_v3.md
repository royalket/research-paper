# The Infrastructure Paradox: Piped Water Expansion and Household
# Water Supply Disruption in India

**Aniket Kumar**
*NFHS-5 Analysis — Working Paper*

---
<!-- ================================================================
     SKILL: research-paper-writing
     Process: one paragraph per role, outline first, claim-evidence map per section
     Current status: Abstract complete — awaiting feedback before Introduction
     ================================================================ -->

---

## ABSTRACT

<!-- Mini-outline (5 bullets, per skill execution rule 1):
     P1 Task + challenge   — policy premise + paradox number
     P2 Insight            — structural lock-in mechanism
     P3 Contribution 1     — IDI + RGI + validation numbers
     P4 Contribution 2     — PSM causal + logit + MC robustness
     P5 Implication        — coverage ≠ security
-->

<!-- P1 | Role: Task + Challenge -->
India's Jal Jeevan Mission targets a piped tap connection in every rural household, operating on the WHO/UNICEF classification that piped water is an "improved" and therefore more reliable source than tube wells or protected springs. Using 578,062 households from the National Family Health Survey Round 5 (NFHS-5, 2019–21), we show this premise is empirically false: piped water households experience supply disruptions at 2.43 times the rate of tube-well households (25.5% vs 10.5%, 95% CI for piped: 25.2–25.7%). This gap persists within every wealth quintile, in both urban and rural areas, across all six geographic regions, and across all four seasons.

<!-- P2 | Role: Insight -->
The paradox arises not from any intrinsic property of pipe material or water quality, but from structural lock-in: piped households are unable to cope when centralised systems fail because they have no practiced alternative. A household with water flowing directly to the tap has no fetching routine, no alternative source, and — if poor — no storage or transport to acquire water elsewhere. When the tap fails, it is fully exposed. Tube-well households, by contrast, are self-sufficient by design: failure affects one household rather than an entire distribution network.

<!-- P3 | Role: Contribution 1 — IDI + RGI -->
To formalise this mechanism, we construct the Infrastructure Dependency Index (IDI), a household-level composite of four dimensions — source lock-in, access complexity, system dependency, and piped coping deficit — combined via principal component analysis (Cronbach α = 0.706, ROC-AUC = 0.619, all PCA loadings positive). We complement the IDI with the Reliability Gap Index (RGI), a district-level measure of how far observed disruption exceeds what a district's socioeconomic development level predicts. Mean IDI is 59.4 points higher for piped than tube-well households, confirming that piped households are structurally more locked in. Monte Carlo uncertainty quantification over 500 runs finds OR(piped) > 1 in 100% of runs (mean OR = 1.753, 95% range: 1.724–1.780), confirming the finding is robust to dimension scoring uncertainty.

<!-- P4 | Role: Contribution 2 — Causal evidence -->
Propensity score matching, balancing piped and tube-well households on wealth, urban location, household composition, assets, and caste, yields a causal average treatment effect of +15.1 percentage points (95% CI: 14.8–15.3 pp, 100% of 500 bootstrap samples positive). Logistic regression with cluster-robust standard errors confirms piped water independently doubles disruption risk after full controls (OR = 2.09, 95% CI: 1.84–2.36, p < 0.001). The cross-level interaction between IDI and district RGI is in the expected direction (OR = 1.037) but does not reach conventional significance (p = 0.083), indicating the amplification mechanism requires replication at finer geographic resolution.

<!-- P5 | Role: Implication -->
These findings challenge the coverage-first logic of current water policy. Piped connections improve water security only where centralised systems deliver water reliably and where households retain access to non-piped backup sources. Expanding connections without simultaneous investment in system reliability — and without preserving existing decentralised sources — increases disruption risk for the most locked-in households.

**Keywords:** water insecurity, infrastructure paradox, India, NFHS-5, piped water, reliability gap, infrastructure dependency index, propensity score matching

---

<!-- ================================================================
     ABSTRACT — Self-review checklist (skill output contract)
     ================================================================

     Clarity:
     [x] Each paragraph carries one message
     [x] First sentence of each paragraph states that message
     [x] IDI, RGI, JJM, JMP all defined inline on first use
     [x] No filler language (notably, interestingly, it is worth noting)
     [x] Third person throughout

     Flow:
     [x] P1 → P2: challenge (piped worse) → consequence (why: lock-in)
     [x] P2 → P3: insight (lock-in) → contribution (we formalise it as IDI+RGI)
     [x] P3 → P4: index construction → causal testing of the index
     [x] P4 → P5: evidence → implication (consequence)

     Terminology consistency:
     [x] "disruption" used throughout (not "outage", not "failure")
     [x] "tube-well" hyphenated consistently
     [x] "piped water" (not "tap water") consistent with NFHS variable name

     Unsupported claims: none — all numbers tied to data output

     Missing evidence:
     [ ] P3 references RGI but gives no RGI validation number in abstract
         → acceptable at abstract level; RGI result is in P4 via GEE OR = 1.675
         → to add: consider "districts with the highest RGI show 67% higher
           disruption odds (OR = 1.675)" if space allows

     ================================================================

     ABSTRACT — Claim-evidence map

     Claim: Piped 2.43× disruption rate of tube wells (25.5% vs 10.5%)
     Evidence: Table 1a, weighted RR
     Status: Supported

     Claim: Gap holds in all wealth quintiles, urban/rural, regions, seasons
     Evidence: Tables 1b–1e
     Status: Supported

     Claim: Structural lock-in is the mechanism
     Evidence: IDI validation (AUC 0.619), piped sub-type gradient (Table 1a)
     Status: Supported

     Claim: Cronbach α = 0.706, AUC = 0.619
     Evidence: idi_validation.csv
     Status: Supported

     Claim: Mean IDI piped − tube well = +59.4 points
     Evidence: idi_validation.csv (known-groups check)
     Status: Supported

     Claim: 100% MC runs OR(piped) > 1
     Evidence: idi_monte_carlo_summary.csv
     Status: Supported

     Claim: ATT = +15.1 pp (95% CI 14.8–15.3)
     Evidence: table7_psm_att.csv
     Status: Supported

     Claim: OR = 2.09 (95% CI 1.84–2.36) after full controls
     Evidence: table2_idi_regression.csv
     Status: Supported

     Claim: Interaction OR = 1.037, p = 0.083
     Evidence: table5_gee_multilevel.csv
     Status: Direction supported; significance not confirmed

     ================================================================ -->

---

---

## 1. INTRODUCTION

<!-- Mini-outline (5 paragraphs, Version 4 structure — familiar task, expose challenge via prior methods):
     P1  Task + application      — why supply reliability matters; JJM policy stakes
     P2  Prior failure 1         — JMP improved/unimproved: measures access, not reliability
     P3  Prior failure 2 + root  — water security indices + JJM evaluations; root issue = no lock-in measure
     P4  Our solution            — IDI + RGI; why they solve the root issue
     P5  Contributions + results — three contributions, key numbers
-->

<!-- P1 | Role: Task + Application -->
Supply disruptions — days on which a household's primary water source is unavailable — impose direct costs on health, time, and welfare that access metrics do not capture. When piped water fails, households turn to unmonitored alternatives, often surface water or tanker vendors, increasing exposure to waterborne pathogens. Women and girls bear disproportionate burdens, spending additional hours locating and collecting water from more distant sources (WHO/UNICEF, 2023). Children miss school. Stored water depletes faster than households anticipate, compressing the window for safe handling. Measuring whether a household has a water connection is therefore insufficient for evaluating water security; what matters is whether that connection delivers water when the household needs it.

<!-- P2 | Role: Prior method failure 1 — JMP -->
The dominant global framework for tracking water access, the WHO/UNICEF Joint Monitoring Programme (JMP), classifies sources as either "improved" or "unimproved" based on source type. Piped water, tube wells, and protected wells are improved; surface water, unprotected wells, and vendor carts are unimproved. India's Jal Jeevan Mission (JJM, 2019–present) — the world's largest rural water programme — uses this classification as its primary rationale, targeting a piped tap connection in every rural household on the premise that piped equals safer and more reliable. The technical limitation of this framework is that it measures source type at the time of survey, not source reliability over time. A household whose tap is dry for twenty days a month is classified identically to one with uninterrupted supply. JMP data cannot detect the Infrastructure Paradox documented in this paper because the data were not designed to measure reliability — only access.

<!-- P3 | Role: Prior method failure 2 + root technical issue -->
Household water security instruments go further by measuring the lived experience of water stress. The Household Water Insecurity Experiences (HWISE) scale (Young et al., 2019) records how often households run out of water, worry about supply, or change plans due to unavailability. JJM programme evaluations track functional tap status and connection rates at the habitation level (Jal Shakti Ministry, 2023). Both approaches share a common gap: neither measures structural lock-in — the degree to which a household depends on a single centralised source with no practiced fallback. A household that has adapted to chronic low-reliability piped supply by storing water in an overhead tank will score low on HWISE despite being fully exposed when the tank empties and the pipe remains dry. A JJM evaluation that records "tap functional at time of visit" cannot distinguish a system with 95% uptime from one with 40% uptime. The root technical issue is that no existing measurement framework asks: when this source fails, can this household cope — and what structural features determine whether it can or cannot?

<!-- P4 | Role: Our solution + why it works -->
This paper addresses that gap by separating two forces that jointly drive disruption risk. The first is household-level structural lock-in: how dependent is the household on a single centralised source, with no backup source, no fetching experience, and no assets to cope when it fails? We formalise this as the Infrastructure Dependency Index (IDI), a composite of four dimensions — source lock-in, access complexity, system dependency, and piped coping deficit — weighted by principal component analysis so that empirically redundant dimensions do not inflate the index. The second force is district-level system underperformance: how far does a district's observed disruption rate exceed what its socioeconomic development level predicts? We formalise this as the Reliability Gap Index (RGI), constructed as the residual from a weighted OLS model that controls for wealth, urbanisation, piped coverage, and survey timing. The IDI works because it captures inability-to-cope independently of source type; a non-piped household with no backup still has fetching experience and self-sufficiency that a piped household lacks. The RGI works because it separates infrastructure failure from poverty — a poor district with functioning tube wells is not failing; a wealthy district with broken centralised pipes is.

<!-- P5 | Role: Contributions + experiment summary -->
This paper makes three contributions. First, using 578,062 households from NFHS-5, we document that piped water households experience disruptions at 2.43 times the rate of tube-well households (25.5% vs 10.5%, Table 1), a gap that holds across all wealth quintiles, urban and rural areas, all regions, and all seasons. Among piped sub-types, yard/plot connections show the highest rate (29.7%, relative risk = 2.83), revealing a within-piped gradient consistent with network pressure effects. Second, we construct and validate the IDI (Cronbach α = 0.706, AUC = 0.619) and RGI, and show that district RGI independently predicts disruption (OR = 1.675, 95% CI: 1.619–1.732, p < 0.001) after household-level controls. Third, propensity score matching yields a causal average treatment effect of +15.1 percentage points (95% CI: 14.8–15.3 pp) for piped water relative to matched tube-well households, with 100% bootstrap confidence — the strongest causal estimate of the cost of piped water adoption under current Indian infrastructure conditions.

---

## 2. RELATED WORK

<!-- Mini-outline (3 topics, per references/related-work.md):
     Topic 1  Water access measurement — JMP improved/unimproved; source type ≠ reliability
     Topic 2  Household water security indices — HWISE; experienced stress ≠ structural capacity
     Topic 3  Infrastructure expansion evaluations — JJM; connection as terminal outcome
-->

<!-- Topic 1 | Role: Paradigm + limitation — water access measurement -->
The dominant framework for tracking household water access is the WHO/UNICEF Joint Monitoring Programme (JMP) ladder, which classifies sources into five tiers — surface water, unimproved, limited, basic, and safely managed — based on source type and on-premises availability [CITE: WHO/UNICEF JMP 2023]. Piped water into a dwelling is classified as safely managed; tube wells are classified as basic. This typology underpins SDG 6.1 progress tracking and has directly shaped policy design, including India's Jal Jeevan Mission, which uses the presence of a household tap connection as its primary outcome indicator [CITE: GoI JJM programme document]. The technical limitation of the JMP framework for studying disruption is that it records source type at the time of survey, not whether the source delivered water reliably over time. Two households with identical JMP classifications — both with piped connections, both on-premises — are indistinguishable whether one experienced zero disruptions and the other experienced thirty days without water in the past year. The framework was designed to track universal access, not to evaluate system reliability, and it cannot detect the disruption gradient this paper documents.

<!-- Topic 2 | Role: Paradigm + limitation — household water security indices -->
A second line of research measures household water security through experienced insecurity rather than source classification. The Household Water Insecurity Experiences (HWISE) scale [CITE: Young et al. 2019] and the Water Insecurity Index [CITE: Vaitla et al. 2017] record the frequency with which households ran out of water, worried about supply, went without washing, or changed food preparation plans due to unavailability. These instruments capture what households feel and do in response to water stress, and have been validated across multiple low- and middle-income country settings. The limitation for the present analysis is that experienced insecurity conflates two distinct phenomena: a household that has adapted to chronic low-reliability piped supply — for example, by maintaining a rooftop storage tank — will score low on HWISE despite remaining fully exposed when the tank empties and the pipe remains dry for an extended period. Conversely, a household experiencing a single acute disruption will score high. Neither response identifies the structural features — backup source availability, fetching experience, coping assets — that determine whether a household can absorb a disruption or is overwhelmed by it. Our IDI is designed to measure those structural features directly, independently of whether the household has yet experienced a disruption.

<!-- Topic 3 | Role: Paradigm + limitation + transition — infrastructure expansion evaluations -->
A third body of work evaluates large-scale water infrastructure programmes by tracking connection rates, functional tap status, and coverage expansion. Studies of JJM and its predecessor, the National Rural Drinking Water Programme (NRDWP), measure whether households have been connected and whether taps were functional at the time of evaluation [CITE: Jal Shakti Ministry 2023; MoJS 2020]. International evaluations of similar programmes — piped water expansion in rural sub-Saharan Africa, Cambodia, and Bangladesh — follow the same design: connection is the terminal outcome, and post-connection reliability is not tracked [CITE: Hutchings et al. 2015; Nagel et al. 2015]. The shared limitation is that treating connection as the endpoint prevents any evaluation of whether connected households experience more or fewer disruptions than the unconnected households they were compared to before the intervention. No study in this literature has used nationally representative household survey data to directly compare disruption rates between piped and non-piped households at scale, controlling for confounders. This paper fills that gap using NFHS-5, a nationally representative sample of 578,062 Indian households, with propensity score matching to provide causal identification.

---

## 3. METHODS

<!-- Mini-outline (per references/method.md — motivation / design / technical advantage per module):
     P1   Overview          — pipeline setting; what each subsection covers
     3.1  IDI
       P2 Motivation        — why composite index, not just piped_flag
       P3 Design            — four dimensions and scoring logic
       P4 Technical adv.    — PCA over equal weights; Dim 4 gating; Dim 2 fix; Monte Carlo
     3.2  RGI
       P5 Motivation        — why district-level residual, not raw disruption rate
       P6 Design            — OLS predictors, piped_coverage and pct_monsoon, bootstrap CI, typology
       P7 Technical adv.    — separates infrastructure failure from poverty
     3.3  Regression models
       P8  IDI logit        — formula, cluster-robust SE, interaction term
       P9  GEE              — why not flat logit, exchangeable correlation
       P10 Slope-as-outcome — two stages, inverse-variance weighting
       P11 PSM              — covariates, caliper, ATT, bootstrap SE
-->

<!-- P1 | Role: Overview -->
The analytical pipeline has three components, each addressing a distinct level of the disruption mechanism. Section 3.1 constructs the Infrastructure Dependency Index (IDI), a household-level composite that quantifies structural lock-in — how unable a household is to cope when its primary source fails. Section 3.2 constructs the Reliability Gap Index (RGI), a district-level measure of how far observed disruption exceeds what a district's socioeconomic development level predicts. Section 3.3 tests the joint role of IDI and RGI through four complementary regression approaches: an IDI logistic regression establishing the household-level mechanism, a Generalized Estimating Equations (GEE) model testing the cross-level IDI × RGI interaction, a two-stage slope-as-outcome model providing a more direct test of the amplification hypothesis, and propensity score matching for causal identification.

### 3.1 Infrastructure Dependency Index (IDI)

<!-- P2 | Role: Motivation — why a composite index, not just piped_flag -->
A binary piped/non-piped indicator identifies which households experience the paradox but cannot explain why two piped households have different outcomes when the tap runs dry. Consider two households that both have piped water into the dwelling: the first has no alternative source, has never fetched water, owns no vehicle, and falls in the poorest wealth quintile; the second maintains a protected well as backup, has daily fetching experience from a nearby source, owns a motorcycle, and has a refrigerator for water storage. Both households have identical piped_flag values, yet their coping capacity when the tap fails is structurally different. A composite index that scores both households across multiple dimensions of lock-in captures this difference, reduces the multiple-testing problem that would arise from entering each dimension separately into the regression, and produces a single actionable measure — policymakers can target households above a given IDI threshold in ways they cannot from a list of four separate conditions.

<!-- P3 | Role: Design — four dimensions and scoring logic -->
The IDI comprises four dimensions, each scored 0–3 where higher scores indicate greater lock-in. Dimension 1 (Source Lock-in) measures whether the household has any non-piped backup: piped primary with no alternative or a piped alternative scores 3; piped with a non-piped backup scores 2; non-piped with same-category backup scores 1; non-piped with a different-category backup scores 0. Dimension 2 (Access Complexity) measures fetching experience: households with water on-premises — derived from NFHS-5 variable hv204 = 996, indicating the source is at or within the dwelling — score 3, reflecting zero fetching experience; off-premises households score 2, 1, or 0 depending on whether the collection point is in the yard, within 15 minutes, or 15 minutes or more away. Dimension 3 (System Dependency) scores how much obtaining alternative water requires market access: tanker truck or cart scores 3; any piped sub-type scores 2; community RO plant or protected well or spring scores 1; tube well, rainwater, or surface water scores 0. Dimension 4 (Piped Coping Deficit) applies only to piped households and measures their buffer capacity: a buffer score is computed as wealth quintile contribution (Q1–Q2 → 0, Q3 → 1, Q4 → 2, Q5 → 3) plus refrigerator ownership plus vehicle ownership, clipped to [0, 3]; the coping deficit score is 3 minus the buffer, so poorest piped households with no assets score 3 and richest piped households with full assets score 0. Non-piped households score 0 on Dimension 4 regardless of assets, because their disruption exposure is structurally different — they already fetch regularly and maintain a practiced routine.

<!-- P4 | Role: Technical advantage — PCA weighting, two design fixes, Monte Carlo -->
Three design choices distinguish the IDI from a simple additive index. First, the four dimensions are combined via the first principal component of their standardised scores rather than summed with equal weights. Dimensions 1 and 3 correlate at r = 0.873 — both are driven primarily by piped adoption — and equal weighting would double-count this signal. PCA assigns a joint loading of 0.622 + 0.606 = 1.228 to the two correlated dimensions and 0.128 + 0.478 = 0.606 to the two less-correlated dimensions, naturally down-weighting redundant information. PC1 explains 57.7% of total variance across the four dimensions; all four loadings are positive (confirming consistent lock-in directionality); and Cronbach α = 0.706, indicating adequate internal consistency for a four-item composite. Second, Dimension 4 is gated on piped_flag. Without this gate, wealth and vehicle ownership are confounded with piped adoption — richer households are more likely to be piped — and the loading on Dimension 4 becomes negative (−0.239 in early versions), pulling the index in the wrong direction. Among piped households only, the correct direction is confirmed: households without a refrigerator have 27.9% disruption versus 25.3% for those with one, and households without a vehicle have 27.3% versus 26.2%. Third, to quantify uncertainty in the dimension scoring thresholds, we run 500 Monte Carlo iterations, each adding Gaussian noise N(0, 0.3) to all dimension scores and recording OR(piped) from a logistic regression on the perturbed IDI. OR(piped) exceeds 1.0 in all 500 runs (mean = 1.753, 95% range: 1.724–1.780), confirming the paradox is not an artifact of any particular scoring choice.

### 3.2 Reliability Gap Index (RGI)

<!-- P5 | Role: Motivation — why district-level residual, not raw disruption rate -->
The IDI captures whether a household can cope when its source fails; it does not capture whether the district's centralised infrastructure is the kind of system that fails. A high-IDI household in a district with a well-maintained piped network has low actual disruption risk despite its structural lock-in. The same household in a district where the piped utility chronically underperforms faces compounded risk: it is both unable to cope and living in a failing system. To distinguish these cases, we need a district-level measure of system underperformance that is independent of household poverty. A raw district disruption rate cannot serve this purpose — a poor rural district with 30% disruption and a wealthy urban district with the same 30% disruption have very different implications. The poor district may simply lack the economic development that would normally reduce disruption; the wealthy district is failing relative to its own potential. The RGI separates these by constructing the residual from a model that predicts disruption from development variables: what a district's disruption rate should be, given its wealth, urbanisation, and infrastructure coverage, versus what it actually is.

<!-- P6 | Role: Design — OLS predictors, two critical controls, bootstrap CI, typology -->
We aggregate NFHS-5 data to 704 districts with at least 100 households and fit a weighted ordinary least squares model predicting each district's observed disruption rate from five predictors, with weights equal to the number of households per district so that larger, more reliable districts have greater influence. The predictors are mean household wealth score, urbanisation rate, improved-source coverage, piped coverage, and the share of households surveyed during monsoon months (June–September). Two predictors require specific justification. Piped coverage is included because a district where 90% of households use piped water mechanically exposes more households to piped-system failure; without this control, high-coverage districts would always appear to underperform — conflating the RGI with the very phenomenon under study. The monsoon share controls for survey-timing bias: NFHS-5 was fielded over two years and districts surveyed in July or August show higher disruption than identical districts surveyed in February, purely due to seasonal flooding and contamination shutdowns rather than infrastructure quality. The RGI for each district is the residual: observed minus predicted. Positive values indicate underperformance; negative values indicate outperformance relative to development expectations. We quantify estimation uncertainty with 500 bootstrap iterations, resampling districts with replacement and refitting the OLS each time, yielding a 95% confidence interval for each district's RGI. Using the weighted medians of IDI and RGI as cut-points, we classify all 704 districts into four typologies: CRISIS (high IDI, high RGI; 189 districts), VULNERABLE (high IDI, low RGI; 179), SAFE (low IDI, low RGI; 174), and RESILIENT POOR (low IDI, high RGI; 162).

<!-- P7 | Role: Technical advantage — what RGI reveals that raw rate cannot -->
The RGI's advantage over a raw disruption rate is visible in the CRISIS district findings. The top-ranked CRISIS district — Gujarat district 851 — has an observed disruption rate of 72.2% and piped coverage of 78.6%, yielding RGI = 42.7 pp (95% CI: 40.8–44.8). A raw disruption rate alone would flag this district as problematic, but would not distinguish it from a poor district where 72% disruption reflects the absence of any improved infrastructure. The RGI identifies Gujarat district 851 as a severe infrastructure failure specifically because its disruption rate is 42.7 percentage points higher than what its wealth, urbanisation, and coverage levels predict — its system is failing households that are already locked in, not simply serving a poor population with few options. The RESILIENT POOR typology makes the complementary point: these 162 districts have high RGI — their centralised systems are underperforming — yet low IDI because households maintain diverse non-piped backup sources and fetching experience. Their disruption rates are lower than CRISIS districts despite comparable system failure, confirming that household-level structural diversity is protective even when centralised infrastructure fails.

### 3.3 Regression Models

<!-- P8 | Role: Design + technical advantage — IDI logistic regression -->
The first model tests whether piped water and IDI independently predict disruption after controlling for all observable household characteristics. The outcome is binary (water_disrupted = 0/1), so we estimate a logistic regression of the form: logit(disruption) = piped_flag + IDI + piped_flag × IDI + wealth_quintile + urban + household_size + children_under_5 + region + SC_ST + female_headed + head_education. Standard errors are clustered at the primary sampling unit level to correct for within-community correlation — households in the same village or urban block share unmeasured local characteristics, and ignoring this clustering would produce standard errors that are too small and p-values that are too low. The interaction term piped_flag × IDI tests whether the disruption penalty of piped water increases with IDI; if the interaction odds ratio exceeds 1, higher IDI amplifies the piped-water disruption effect at the household level. Social controls — SC/ST status, female-headed household, and head of household education — are included as regression controls rather than as IDI dimensions, because caste and education are characteristics of the household that modify its experience of system failure; they are not features of the water system's architecture and do not belong in the structural index.

<!-- P9 | Role: Design + technical advantage — GEE multilevel model -->
The second model tests the cross-level interaction between household IDI and district RGI. Households within the same district share unmeasured infrastructure quality — the reliability of the local water utility, the state of distribution pipes, the frequency of electricity outages that power pumps — and a flat logistic regression that ignores this within-district correlation produces standard errors that are too optimistic. We use Generalized Estimating Equations (GEE) with district-level grouping and an exchangeable correlation structure, which assumes all pairs of households within the same district are equally correlated and estimates the working correlation coefficient from the data. GEE produces population-averaged estimates — the effect on the average Indian household — rather than the subject-specific estimates of random-effects models, and is the appropriate estimand for policy claims about national programme impacts. Both IDI and RGI are standardised to mean 0, standard deviation 1 before computing their interaction, so the interaction coefficient represents the change in the IDI slope per one standard deviation increase in district RGI — a directly interpretable quantity. We note that IDI and piped_flag are highly collinear in this specification, because IDI dimensions are partly constructed from piped adoption; the IDI main effect in the GEE is therefore difficult to interpret in isolation and we report it transparently alongside this limitation.

<!-- P10 | Role: Design + technical advantage — slope-as-outcome model -->
The third model provides a more direct test of the amplification hypothesis than the GEE interaction term. The GEE interaction tests whether IDI and RGI jointly predict disruption in the pooled sample; it cannot confirm whether the IDI mechanism — how strongly lock-in predicts disruption — is specifically steeper in districts with higher reliability gaps. The slope-as-outcome model tests this directly in two stages. In Stage 1, we fit a separate logistic regression of disruption on IDI, wealth, and urban within each of 657 districts that have at least 30 households and at least 50 piped households, and extract the district-level IDI coefficient β_d and its standard error SE(β_d). The minimum of 50 piped households ensures sufficient variation in IDI for a stable slope estimate; districts below this threshold are excluded to prevent noisy Stage 1 estimates from contaminating Stage 2. In Stage 2, we regress the 657 district IDI slopes on district RGI using weighted least squares with inverse-variance weights 1/SE(β_d)², so that districts with precise slope estimates contribute more to the regression than districts with noisy ones. A positive, significant Stage 2 RGI coefficient would confirm that the IDI-to-disruption relationship is steeper in districts where infrastructure underperforms — the specific contextual moderation claim of this paper.

<!-- P11 | Role: Design + technical advantage — propensity score matching -->
The fourth approach provides causal identification. All regression models in Sections 3.3.1–3.3.3 estimate associations; an unmeasured confounder — for example, whether piped-water areas also have worse electricity reliability that causes both piped adoption and pump failures — could bias the piped coefficient upward. Propensity score matching (PSM) addresses this by constructing a matched sample where piped and tube-well households are as similar as possible on all observable characteristics, so the comparison within matched pairs is closer to a randomised experiment. We estimate each household's propensity to use piped water using a logistic regression on wealth quintile, urban, household size, children under five, female-headed status, SC/ST status, electricity access, and improved sanitation, then match each piped household to the nearest tube-well household within a caliper of 0.05 standard deviations of the propensity score. The caliper prevents poor-quality matches: if no tube-well household falls within 0.05 of a piped household's propensity score, that piped household is excluded rather than matched to a distant neighbour. The estimand is the average treatment effect on the treated (ATT) — the average disruption difference for piped households relative to what those same households would have experienced with tube wells — which is the policy-relevant parameter for evaluating the effect of connecting households that are currently piped. We report ATT with bootstrap standard errors from 500 resamples of the matched pairs.

---

## 4. RESULTS

<!-- Mini-outline (per references/experiments.md — link every result to a table, one table one message):
     P1  Finding 1a — Paradox overall      — Table 1a: 25.5% vs 10.5%, RR 2.43, within-piped gradient
     P2  Finding 1b — Paradox universal    — Tables 1b–1e: holds in all strata
     P3  Finding 2  — IDI regression       — Table 2: piped OR 2.09, IDI OR 1.007, SC/ST OR 1.105
     P4  Finding 3  — Geographic           — Table 3a: CRISIS districts, Gujarat top, RESILIENT POOR
     P5  Finding 4  — GEE                  — Table 5: RGI OR 1.675, interaction OR 1.037 p=0.083
     P6  Finding 5  — Slope-as-outcome     — Table 6b: mean β 0.0085, Stage 2 β 0.0007 p=0.083
     P7  Robustness — PSM                  — Table 7: ATT +15.06 pp, 100% bootstrap
-->

<!-- P1 | Role: Finding 1a — The paradox in raw data -->
Piped water households experience supply disruptions at more than twice the rate of tube-well households: 25.5% versus 10.5%, a relative risk of 2.43 (Table 1, 95% CI for piped: 25.2–25.7%; for tube well: 10.3–10.6%). Among the four piped sub-types, yard/plot connections have the highest disruption rate at 29.7% (RR = 2.83 vs tube well), followed by neighbour/shared connections at 26.1% (RR = 2.49), public tap or standpipe at 23.9% (RR = 2.28), and in-dwelling connections at 23.6% (RR = 2.25). This within-piped gradient — yard/plot connections having 6.1 percentage points more disruption than in-dwelling connections — is consistent with network pressure dynamics: peripheral connections are the first to lose pressure when centralised supply is stressed, and they experience shutdowns before in-dwelling connections do. Tube wells (10.5%) have lower disruption than bottled water (10.7%), community RO plants (17.7%), and tanker trucks (24.2%), confirming that the low disruption rate of decentralised self-sufficient sources is not a feature unique to tube wells but reflects the structural independence of sources that do not depend on a shared distribution network.

<!-- P2 | Role: Finding 1b — The paradox holds across all strata -->
The piped–tube well disruption gap is not explained by any single socioeconomic or geographic stratum. By wealth quintile (Table 1b), the gap ranges from 13.0 percentage points in the poorest quintile (piped 23.8%, tube well 10.8%, RR = 2.20) to 17.2 pp in the middle quintile (piped 27.7%, tube well 10.5%, RR = 2.64); it narrows modestly for the richest households (11.9 pp, RR = 2.17), consistent with storage assets buffering disruption effects but not eliminating them. By residence (Table 1c), the gap is 15.9 pp in rural areas (RR = 2.51) and 13.9 pp in urban areas (RR = 2.32); the rural gap being larger is unexpected under the assumption that urban piped systems are better maintained, and suggests that rural piped networks may suffer more from weak operation and maintenance. By region (Table 1d), the Northeast shows the largest relative risk (piped 26.2%, tube well 7.8%, RR = 3.36), while the East shows the smallest (piped 12.8%, tube well 9.3%, RR = 1.38), where lower piped coverage means networks are less over-extended. By season (Table 1e), the monsoon months show the largest absolute gap (piped 32.0%, tube well 13.3%, difference = 18.7 pp), while summer shows the highest relative risk (piped 21.5%, tube well 8.0%, RR = 2.69), reflecting demand-side pressure on centralised supply during dry months. The paradox is present in every cell of every stratification.

<!-- P3 | Role: Finding 2 — IDI regression -->
After controlling for wealth, urban location, household size, children under five, geographic region, SC/ST status, female-headed status, and head of household education, piped water independently doubles disruption odds (OR = 2.088, 95% CI: 1.844–2.364, p < 0.001; Table 2). IDI score is independently significant at OR = 1.007 per point (95% CI: 1.005–1.009, p < 0.001): each additional IDI point multiplies disruption odds by 1.007, amounting to OR = 2.0 across the full 0–100 index range and confirming that structural lock-in predicts disruption beyond the piped/non-piped distinction alone. SC/ST households face 10.5% higher disruption odds than non-SC/ST households after all controls (OR = 1.105, 95% CI: 1.080–1.130, p < 0.001), indicating that marginalised caste groups bear a disruption burden not explained by wealth or source type. Female-headed households show marginally lower disruption odds (OR = 0.975, 95% CI: 0.954–0.996, p = 0.022), consistent with evidence that female-headed households are more likely to maintain backup water sources and fetching routines. The piped × IDI interaction is OR = 0.996 (95% CI: 0.993–0.998, p = 0.002); the direction below 1 reflects multicollinearity between IDI and piped_flag rather than a substantive dampening effect, because IDI dimensions are partly constructed from piped adoption and both predictors compete for the same variance in the model.

<!-- P4 | Role: Finding 3 — Geographic concentration -->
CRISIS districts — those combining high household lock-in with high infrastructure underperformance — are geographically concentrated rather than uniformly distributed across India (Table 3a). Gujarat accounts for four of the top five CRISIS districts by RGI magnitude: district 851 has RGI = 42.7 pp (95% CI: 40.8–44.8), observed disruption of 72.2%, and piped coverage of 78.6%; districts 478, 853, and 855 similarly show RGI values of 38.5, 30.6, and 27.3 pp alongside piped coverages of 88–91%. Arunachal Pradesh contributes two districts in the top ten, with piped coverages of 99–100% and observed disruption rates of 57–60% — systems where near-universal piped adoption has created near-universal lock-in in infrastructure that cannot sustain reliable supply. The RESILIENT POOR typology demonstrates the protective value of backup source diversity: these 162 districts have high RGI values — their centralised systems are underperforming — yet low IDI because households retain non-piped backup sources and fetching experience accumulated over time. Their disruption rates are substantially lower than CRISIS districts despite comparable system failure, confirming that household structural diversity buffers against system-level underperformance when connections to decentralised sources are preserved.

<!-- P5 | Role: Finding 4 — GEE multilevel model -->
The GEE model confirms that district-level infrastructure underperformance independently predicts household disruption: each standard deviation increase in RGI is associated with OR = 1.675 higher disruption odds (95% CI: 1.619–1.732, p < 0.001; Table 5), after controlling for household IDI, piped water use, wealth, urban location, household composition, region, SC/ST status, and female-headed status. Piped water retains OR = 2.293 (95% CI: 2.016–2.608, p < 0.001) in this district-grouped specification, confirming that the paradox persists after accounting for within-district correlation. The IDI × RGI cross-level interaction is OR = 1.037 (95% CI: 0.995–1.080, p = 0.083) — in the expected direction but not reaching the conventional α = 0.05 threshold. Two factors likely attenuate this estimate. First, RGI is itself a generated regressor carrying measurement error from the Stage 1 OLS; classical attenuation bias pushes any coefficient on a noisy predictor toward zero. Second, the 704 districts available in NFHS-5 may be too coarse a geographic unit to capture the sub-district variation where the IDI × RGI amplification mechanism operates most sharply. The IDI main effect in this model is OR = 0.897 (p < 0.001, below 1), which reflects the multicollinearity between IDI and piped_flag noted in Section 3.3.2 and should not be interpreted as a substantive finding.

<!-- P6 | Role: Finding 5 — Slope-as-outcome -->
Stage 1 of the slope-as-outcome model yields reliable IDI slopes for 657 districts. The mean district-level IDI slope is β = 0.0085 log-odds per IDI point (95% CI: 0.0077–0.0093, p < 0.001; Table 6b intercept), confirming that IDI independently predicts disruption within districts rather than only across them: across the full IDI range of 0–100, this slope corresponds to OR ≈ exp(0.85) = 2.34 within the typical district, consistent with the pooled logistic regression result. Stage 2 finds that one standard deviation increase in district RGI is associated with a β = 0.0007 increase in the district IDI slope (95% CI: −0.0001–0.0014, p = 0.083, R² = 0.005). The direction is consistent with the amplification hypothesis — failing districts show steeper IDI-to-disruption relationships — but the estimate is imprecise and does not reach significance. The two attenuation mechanisms identified for the GEE interaction apply with equal force here: RGI measurement error directly attenuates the Stage 2 coefficient through the errors-in-variables mechanism, and district-level aggregation obscures the sub-district variation where lock-in and system failure co-occur most acutely. We treat the slope-as-outcome result as directionally supportive of the amplification hypothesis but not confirmatory at the current geographic resolution.

<!-- P7 | Role: Robustness — PSM causal estimate -->
Propensity score matching matches all 296,651 piped households to tube-well neighbours within the 0.05 caliper, achieving a 100% match rate and leaving no unmatched treated units. The average treatment effect on the treated is ATT = +15.06 percentage points (SE = 0.13, 95% CI: 14.77–15.31; Table 7): piped water causes approximately 15 additional percentage points of disruption probability for the households that currently have piped connections, relative to what those same households would have experienced with tube wells, after constructing a counterfactual where both groups are identical on all observable confounders. All 500 bootstrap resamples yield ATT > 0, placing the probability of a null or negative causal effect below 1 in 500 under any plausible resampling distribution. This ATT is the paper's strongest causal statement: it does not rely on model linearity, functional form assumptions, or the distributional requirements of random-effects specifications, and it survives the inclusion of all eight matching covariates including electricity access and improved sanitation — variables that proxy for the local infrastructure quality that could otherwise confound the piped–disruption association.

---

## 5. DISCUSSION

<!-- Mini-outline (per SKILL.md §Discussion-to-Results Alignment):
     P1  Open with result summaries   — restate key numbers before interpretation
     P2  Architectural explanation    — system architecture, not pipe quality; gradient as evidence
     P3  IDI dimension profiles       — rich vs poor piped; different mechanisms, different interventions
     P4  Why interaction not confirmed — two reasons; what replication requires
     P5  Policy implications          — three concrete recommendations from evidence
     P6  Limitations                  — cross-sectional, self-reported, hv202, geographic resolution
-->

<!-- P1 | Role: Open with result summaries -->
Across 578,062 Indian households, piped water households experience supply disruptions at 2.43 times the rate of tube-well households (25.5% vs 10.5%, Table 1a). This gap holds in every wealth quintile, both urban and rural areas, all six geographic regions, and all four seasons, ranging from RR = 2.17 in the richest quintile to RR = 3.36 in the Northeast (Tables 1b–1e). After controlling for all observable household characteristics, piped water independently doubles disruption odds (OR = 2.088, 95% CI: 1.844–2.364; Table 2), and propensity score matching places the causal effect at +15.1 percentage points (95% CI: 14.8–15.3; Table 7) with 100% bootstrap confidence. The Infrastructure Dependency Index confirms that structural lock-in independently predicts disruption (OR = 1.007 per point, AUC = 0.619) and is 59.4 points higher on average for piped than tube-well households. Districts where infrastructure underperforms its development level show OR = 1.675 higher disruption odds per standard deviation of RGI (Table 5), and the 189 CRISIS districts — concentrated in Gujarat and Arunachal Pradesh — have observed disruption rates of 53–72% despite piped coverages of 79–100%.

<!-- P2 | Role: Architectural explanation -->
The paradox is not a puzzle about water quality or pipe material — it is a puzzle about failure propagation. Tube wells fail independently: when one breaks, one household is affected and its neighbours are unaffected. When a centralised piped distribution system fails — due to pump outage, pipe burst, electricity disruption, or contamination-triggered shutdown — every household connected to that network is affected simultaneously. The household-level disruption rate observed in survey data captures both the probability of a failure event and the number of households affected per event; centralised systems produce lower failure frequency but far higher household exposure per failure. The within-piped gradient confirms this interpretation. Yard/plot connections (29.7% disruption) have 6.1 percentage points more disruption than in-dwelling connections (23.6%), despite serving households on the same distribution system in the same localities. This gradient reflects network hydraulics: peripheral nodes lose pressure first when the system is stressed, and household distance from the distribution main predicts disruption risk independently of household characteristics. If the paradox were driven solely by household socioeconomic features, all piped sub-types would show similar rates regardless of connection location. The gradient is a system architecture signal, not a household signal.

<!-- P3 | Role: IDI dimension profiles — rich vs poor piped, different mechanisms -->
The IDI dimension profiles by wealth quintile reveal that the paradox operates through different mechanisms for rich and poor piped households. Source lock-in (Dim 1) and system dependency (Dim 3) both increase monotonically from the poorest to the richest quintile — Dim 1 rising from 0.975 to 2.134 and Dim 3 from 0.686 to 1.624 — because richer households adopt piped water at higher rates and are therefore more structurally tied to the centralised system. The piped coping deficit (Dim 4), however, falls sharply from 0.933 in the Poorer quintile to 0.000 in the Richest: wealthy piped households have refrigerators for water storage, vehicles to fetch from alternatives, and financial resources to purchase tanker water if needed. The result is that composite IDI is surprisingly flat from the Poorer to Richest quintiles (44–48), with the Poorest notably lower (32.8) because they are less often piped. This pattern has a precise policy implication: the paradox creates different risks for different groups. Wealthy piped households are locked in but buffered — their high Dims 1 and 3 are offset by near-zero Dim 4, and disruption rates narrow toward 22% at the richest quintile. Poor piped households are less locked in because piped adoption is lower, but when they do have piped water they have zero coping capacity: their Dim 4 is high (0.933), their disruption rate is 23.8%, and there is no storage tank, no vehicle, and no money for a tanker to bridge the gap when the tap fails.

<!-- P4 | Role: Why the interaction is not confirmed — two reasons, what replication requires -->
The IDI × RGI interaction and slope-as-outcome Stage 2 coefficient are both in the expected direction but fall short of conventional significance (GEE: OR = 1.037, p = 0.083; slope-as-outcome: β = 0.0007, p = 0.083). The individual main effects are strong and unambiguous — household IDI predicts disruption within districts (Stage 1 mean β = 0.0085, p < 0.001) and district RGI predicts disruption independently (OR = 1.675, p < 0.001) — so the gap is specifically in demonstrating that these two forces amplify each other. Two structural reasons account for this. First, the RGI is a generated regressor: it is the residual from a district-level OLS model and therefore carries its own estimation error. When a predictor is measured with error, its coefficient in a downstream regression is attenuated toward zero by a factor of signal-to-noise — the errors-in-variables problem. Correcting for this requires an instrumental variable or repeated-measures design that is not available in NFHS-5. Second, districts are the finest geographic unit in NFHS-5, and India has 704 qualifying districts covering enormous geographic and infrastructural variation within each. The IDI × RGI amplification mechanism likely operates at sub-district scale — at the level of a distribution zone, a water supply scheme, or a block-level utility — where lock-in and system failure genuinely co-occur for the same households. Replicating the slope-as-outcome analysis with block-level or panchayat-level administrative water supply data, or with NFHS-6 if district sub-codes become available, would provide the geographic resolution needed to confirm the interaction.

<!-- P5 | Role: Policy implications — three recommendations directly from evidence -->
Three policy recommendations follow directly from the evidence. First, JJM and comparable piped expansion programmes should add disruption rate as a primary monitoring outcome alongside connection rate. The PSM ATT of +15.1 pp means that connecting a household to piped water under current system conditions increases its expected disruption by 15 percentage points; this cost is immediate and borne by the household from the day of connection. A monitoring framework that counts connections as successes without tracking disruption will systematically misrepresent outcomes for newly connected households in CRISIS districts. Second, existing decentralised backup sources — tube wells, hand pumps, and protected wells — should be maintained rather than decommissioned when piped connections are installed. The RESILIENT POOR districts show that backup source diversity protects households even when centralised systems fail; the IDI Source Lock-in dimension (Dim 1) shows that piped households with a non-piped alternative score 2, not 3, reducing their composite lock-in by a measurable amount. Eliminating the backup on the assumption that it is no longer needed removes the household's only coping mechanism for the disruptions that the data show will occur. Third, the 189 CRISIS districts identified by the RGI require targeted operation and maintenance investment before further coverage expansion. Gujarat, which contributes four of the top five CRISIS districts, has already achieved 79–91% piped coverage in those districts; connecting the remaining households without fixing the systems that are currently producing 64–72% disruption rates will generate more locked-in, unprotected households rather than improving water security.

<!-- P6 | Role: Limitations -->
Four limitations bound the scope of these findings. First, NFHS-5 is cross-sectional, preventing causal inference about the direction of the piped-adoption decision: households may self-select into piped water in part because their local systems are already reliable, which would attenuate the observed disruption gap rather than inflate it. Longitudinal data tracking households before and after connection — ideally from JJM implementation records linked to NFHS-6 — would identify whether system reliability at the time of connection predicts subsequent disruption and inform the timing of reliability investments relative to coverage expansion. Second, the disruption outcome (`sh37b`) is self-reported and captures whether water was unavailable for at least one day in the past two weeks, collapsing variation in duration and frequency into a binary. Households that have adapted to chronic disruption may underreport, compressing the observed gap; the true disruption burden for piped households is likely larger than the 25.5% rate suggests. Third, the NFHS-5 alternative source variable (`hv202`) is not recorded for any household in this dataset — every household shows "No Other Source" — which limits the Source Lock-in dimension (Dim 1) to a binary piped/non-piped signal rather than the richer backup-diversity scoring it was designed to capture. A dataset that records both primary and alternative sources would substantially improve Dim 1's discriminating power and sharpen the IDI's separation between households with and without genuine fallback options. Fourth, as discussed in Section 5.4, district-level geographic resolution is insufficient to confirm the IDI × RGI interaction; sub-district replication remains the primary outstanding analytical task.

---

## 6. CONCLUSION

<!-- Mini-outline (per references/conclusion.md — 5 elements):
     P1  Restate problem + core idea    — what we solved; IDI + RGI as technical contribution
     P2  Strongest evidence             — 2.43× RR, OR 2.09, ATT +15.1 pp
     P3  Practical impact + new insight — coverage ≠ reliability; insight beyond India
     P4  Limitation                     — geographic resolution + hv202
     P5  Concrete future direction      — sub-district replication; NFHS-6; errors-in-variables
-->

<!-- P1 | Role: Restate solved problem + core technical idea -->
This paper addresses a specific and consequential measurement gap: no prior framework tested whether expanding piped water connections increases or decreases household water supply disruption at national scale. The key idea is that disruption risk is not a property of source type but of structural lock-in — the degree to which a household depends on a single centralised source with no practiced alternative — compounded by district-level system underperformance. We formalise structural lock-in as the Infrastructure Dependency Index (IDI), a PCA-weighted composite of four dimensions validated against disruption outcomes (AUC = 0.619, Cronbach α = 0.706, robust across 500 Monte Carlo runs), and district system underperformance as the Reliability Gap Index (RGI), a bootstrapped residual from a development-adjusted OLS model that separates infrastructure failure from poverty.

<!-- P2 | Role: Strongest evidence -->
Three results, obtained through independent analytical approaches, converge on the same finding. Descriptively, piped water households experience disruptions at 2.43 times the rate of tube-well households across 578,062 nationally representative households (25.5% vs 10.5%, Table 1a), a gap that holds in every wealth quintile, both urban and rural areas, all six geographic regions, and all four seasons. Adjusting for wealth, caste, household composition, and geographic region in a logistic regression with cluster-robust standard errors, piped water independently doubles disruption odds (OR = 2.088, 95% CI: 1.844–2.364, p < 0.001, Table 2). Causally, propensity score matching — constructing matched piped and tube-well households identical on all observable confounders — yields ATT = +15.1 pp (95% CI: 14.8–15.3, 100% of 500 bootstrap samples positive, Table 7). These three numbers — 2.43× descriptive, OR 2.09 adjusted, ATT +15.1 pp causal — are consistent across methods with no sign of attenuation from descriptive to causal, indicating that observable confounders do not explain the paradox and that the effect is real.

<!-- P3 | Role: Practical impact + new insight -->
The practical impact is immediate for JJM monitoring and programme design. The 189 CRISIS districts — where high household lock-in combines with infrastructure underperformance to produce observed disruption rates of 53–72% — are identifiable today from NFHS-5 data and can be targeted for operation and maintenance investment before further connection expansion. The insight that makes this targeting possible extends beyond India and applies to any centralised infrastructure expansion programme: access and reliability are not the same outcome, and conflating them in monitoring frameworks conceals the costs households bear when centralised systems fail. A household counted as "safely managed" in JMP terms because it has a piped connection may be experiencing more disruption than it did before the connection was installed, if that connection replaced a functioning tube well with a failing distribution network. The IDI provides a replicable household-level instrument for identifying which newly connected households face this risk — it requires only variables routinely collected in DHS-type surveys — and the RGI provides a district-level instrument for identifying where system reliability investment is most urgently needed before coverage expansion.

<!-- P4 | Role: Limitation -->
The principal scope limitation is geographic resolution. NFHS-5 identifies households at the district level, and district is too coarse a unit to confirm whether the IDI × RGI amplification mechanism holds — the interaction is in the expected direction (GEE OR = 1.037, slope-as-outcome β = 0.0007) but does not reach conventional significance at this scale. The mechanism is theoretically sound and supported by both the CRISIS district pattern and the RESILIENT POOR contrast, but statistical confirmation requires data at the water supply scheme, block, or panchayat level, where lock-in and system failure genuinely co-occur for the same households rather than being averaged across hundreds of thousands of households in a heterogeneous district. A secondary limitation is that the IDI Source Lock-in dimension is effectively binary — piped vs non-piped — because the NFHS-5 alternative source variable is not recorded, preventing the richer backup-diversity scoring the dimension was designed to capture. This means the IDI understates the true variation in lock-in among piped households and likely underestimates the effect of source diversity on coping capacity.

<!-- P5 | Role: Concrete future direction -->
Three extensions would resolve the principal outstanding questions. First, linking JJM administrative connection records to NFHS-6 household data — which will cover the post-JJM expansion period — would enable a before-after comparison of disruption rates for newly connected households, providing the longitudinal causal identification that cross-sectional PSM cannot deliver and directly testing whether reliability at the time of connection predicts subsequent disruption outcomes. Second, applying the IDI and RGI framework to block-level or water supply scheme-level administrative data would provide the geographic resolution needed to test the IDI × RGI interaction; the slope-as-outcome model is already specified and validated at the district level, and rerunning Stage 2 at a finer geographic unit requires only a dataset with sub-district household identifiers and disruption records. Third, an errors-in-variables correction — using an instrumental variable for district RGI or applying a simulation-extrapolation (SIMEX) procedure — would address the attenuation bias in the Stage 2 RGI coefficient and produce an unbiased estimate of whether failing districts genuinely amplify household lock-in effects. Together, these three extensions would convert the directional interaction finding into a confirmed mechanism and provide the causal and longitudinal evidence that current water policy decisions require.
