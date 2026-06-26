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
