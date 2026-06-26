# Complete Paper Publishing Workflow: Step-by-Step Prompts

**Use this file to guide your paper from draft to publication-ready state.**
**Work through ONE section at a time. Copy the prompt, paste it into chat, and provide the required inputs.**

---

## QUICK START: Ask AI for the Info (Easiest Way)

If you don't want to fill in the form yourself, just copy this simple prompt and send it:

```
Use the research-paper-writing skill.

I want to assess my paper for publication. Please ask me the questions needed to do a rapid quality check.
```

The AI will ask you:
- Paper title
- Target journal
- Word limit
- Paper type
- Current state
- Research question
- Key findings
- Available data/tables/figures
- Statistical tests used

Then it will run the full assessment automatically. **This is the easiest way to get started.**

---

## PHASE 1: PAPER ASSESSMENT (Start here)

### Step 1: Assess Your Current Paper

Copy and use this prompt:

```
Use the research-paper-writing skill.

I'm preparing my paper for publication. Here's my current status:

PAPER TITLE: [Your title]
VENUE/JOURNAL: [e.g., "Cleaner Water", "Water Research", "Science of The Total Environment"]
TARGET WORD LIMIT: [e.g., 8000, 12000 words]
PAPER TYPE: [Empirical/Review/Methods]
CURRENT STATE: [Draft/Revised draft/Ready for final polish]

RESEARCH QUESTION: [Your main question in 1-2 sentences]
KEY FINDING: [Your main result/conclusion]

AVAILABLE EVIDENCE:
- Datasets: [e.g., "NFHS-5, 636,699 households"]
- Tables: [How many? What do they show?]
- Figures: [How many? What do they show?]
- Statistical tests: [What did you use? Chi-square, logistic regression, etc.]

Do a rapid quality check on my paper for:
1. Clarity: Are claims evidence-backed? Any unsupported statements?
2. Structure: Are sections in the right order (Intro→Lit Review→Methods→Results→Discussion→Conclusion)?
3. Jargon: Are there buzzwords I should cut or plain up?
4. Tone: Does it read as formal and objective?
5. Missing pieces: What's notably absent?

Return:
1. A 5-bullet assessment of strengths
2. A 5-bullet assessment of weaknesses to fix
3. A recommended revision priority order (which section to fix first?)
```

---

## PHASE 2: SECTION-BY-SECTION REVISION

**DO EACH SECTION ONE AT A TIME.** After completing a section, move to the next.

---

### Step 2: ABSTRACT

When you're ready to work on the Abstract, copy and use this prompt:

```
Use the research-paper-writing skill.

I'm revising my ABSTRACT. Apply Academic Writing Clarity Rules and Empirical Results Reporting standards.

CURRENT ABSTRACT:
[PASTE YOUR ABSTRACT HERE]

PAPER CONTEXT:
- Main finding: [1 sentence summary of your key result]
- Sample size: [e.g., "636,699 households"]
- Main predictors tested: [List 3-5 main variables you examined]
- Primary conclusion: [What should readers take away?]

TARGET JOURNAL: [Journal name]
WORD LIMIT: [e.g., 250 words]

REVISION GOALS:
1. Make the opening sentence state the problem precisely (not "Access to safe water is important")
2. Include specific numbers from your Results section (prevalence %, effect sizes, AOR with CI)
3. Ensure every claim can be traced to a finding (no unsupported statements)
4. Remove vague phrases like "significantly," "notably," "it is found that"
5. End with concrete implications (not "further research needed")

Return:
1. Revised abstract (word count)
2. A 1-sentence summary of what changed
3. Claim-evidence map showing each major claim backed by your results
```

---

### Step 3: INTRODUCTION

When ready for the Introduction, copy and use:

```
Use the research-paper-writing skill.

I'm revising my INTRODUCTION. Apply Academic Writing Clarity Rules.

CURRENT INTRODUCTION:
[PASTE YOUR INTRODUCTION HERE]

PAPER CONTEXT:
- Problem statement: [Your water/sanitation challenge in India]
- Why it matters: [Health burden? Policy gap? Scientific gap?]
- What's missing in the literature: [What gap does your study fill?]
- Your research question: [What exactly are you asking?]

REVISION GOALS:
1. Opening paragraph: State the global problem, then narrow to India, then your specific gap
2. Middle paragraphs: Build evidence for why this problem matters (cite the 70% surface water unsafe, arsenic contamination stats, etc.)
3. Literature gap: Show what's known (HWT is effective) and what's NOT known (patterns and predictors in India at national scale)
4. Bridge to your study: "This gap is critical because [why your study matters]"
5. Final paragraph: State your research objectives clearly

Remove:
- Phrases like "it is known that," "previous research shows," "scholars argue" (replace with specific citations)
- Nested clauses that make sentences hard to follow
- Hype words ("groundbreaking," "novel," "first-ever") unless your results truly justify them

Return:
1. Revised introduction (section-by-section)
2. A reverse outline: thesis statement + topic sentence for each paragraph + one evidence point per paragraph
3. Claim-evidence map: For each major claim in the Intro, map to a citation or your Results section
```

---

### Step 4: LITERATURE REVIEW / RELATED WORK

When ready for Literature Review:

```
Use the research-paper-writing skill.

I'm revising my LITERATURE REVIEW section. Apply Academic Writing Clarity Rules and Discussion-to-Results Alignment.

CURRENT LITERATURE REVIEW:
[PASTE YOUR LIT REVIEW HERE]

KEY THEMES IN YOUR REVIEW:
1. Water scarcity and quality in India: [Key points you covered]
2. HWT practices globally and in India: [What's known?]
3. Predictors of HWT adoption: [Socio-economic, demographic, infrastructural factors]
4. Your gap: [What's NOT studied at national scale in India?]

REVISION GOALS:
1. Each subsection should have a clear topic sentence (not just a heading)
2. Within subsections: Present evidence → Note the gap → Say why it matters for your study
3. Avoid listing studies; instead, synthesize patterns ("Multiple studies show X; however, they differ in Y")
4. End of Review: State clearly what YOU are studying and why (transition to your methods)
5. Remove vague attributions: Replace "Studies show" with "Smith et al. (2023) and Jones (2024) found..."

Example:
- Before: "HWT is important in developing countries."
- After: "Freeman et al. (2012) found that HWT reduced diarrheal disease by 30% in rural India. However, no national-scale study has examined which households adopt HWT and which predictors drive adoption."

Return:
1. Revised literature review (subsection by subsection)
2. A paragraph-by-paragraph reverse outline showing: Topic Sentence | Evidence point | Why it matters to your study
3. Identification of any unsupported claims that need citation
```

---

### Step 5: METHODS

When ready for Methods:

```
Use the research-paper-writing skill.

I'm revising my METHODS section. Apply Empirical Results Reporting standards (precision, citations, transparency).

CURRENT METHODS:
[PASTE YOUR METHODS HERE]

METHODS CHECKLIST (mark True/False for each):
- Data source clearly named and cited (NFHS-5, 2019–21)
- Sample size reported (n = 636,699)
- Sampling strategy explained (two-stage stratified sampling, probability proportional to size)
- Outcome variable clearly defined: ✓ or ✗
- All 13 predictor variables listed and defined: ✓ or ✗
- Statistical tests specified (chi-square, Cramer's V, logistic regression): ✓ or ✗
- Model validation metrics reported (AUC, cross-validation method): ✓ or ✗
- Any missing data or exclusion criteria explained: ✓ or ✗
- Software and versions cited (Python 3.12, libraries used): ✓ or ✗

REVISION GOALS:
1. Fix any "False" items above
2. Replace passive voice where it obscures who did what
   - Before: "Data were collected via computer-assisted interviewing."
   - After: "Ministry of Health and Family Welfare collected data via computer-assisted personal interviewing from all women aged 15–49 years and men aged 15–54 years."
3. Add rationale for statistical choices (Why logistic regression over other methods? Why this model structure?)
4. Explain how weights (hv005) were applied and why
5. State reproducibility: Are code, data, and analyses documented?

Return:
1. Revised Methods section
2. A 1-sentence summary of any major changes
3. Confirmation that all items in the checklist are addressed
```

---

### Step 6: RESULTS

When ready for Results:

```
Use the research-paper-writing skill.

I'm revising my RESULTS section. Apply Empirical Results Reporting standards (point estimates, 95% CI, concrete interpretation, table/figure linking).

CURRENT RESULTS:
[PASTE YOUR RESULTS HERE]

RESULTS INVENTORY:
- Overall HWT prevalence: [Your number, %]
- Range across states: [Low to high, %, states]
- Most common methods: [List + %]
- Key predictors (top 5 AOR with 95% CI): [List]
- Model performance (AUC): [Your number]

REVISION GOALS:
1. Every effect size must include 95% CI and p-value
   - Before: "Higher education (p < 0.001)"
   - After: "Higher education (AOR=1.830, 95% CI: 1.765–1.885, p < 0.001)"

2. Interpret effect sizes concretely
   - Before: "AOR=4.399 for richest quintile"
   - After: "Households in the richest wealth quintile had 4.4× higher odds of treating water (AOR=4.399, 95% CI: 4.271–4.59)"

3. Link every key finding to Figure or Table
   - Before: "Water treatment varies by state."
   - After: "Water treatment ranged from 95% in Nagaland and Kerala to <10% in Bihar (Figure 1; detailed results in Table S4)."

4. Table captions must be self-contained
   - Before: "Socio-demographic characteristics."
   - After: "Table 3. Weighted socio-demographic characteristics and overall water treatment prevalence (41.7%) by predictor variable, NFHS-5 (n = 636,699). Percentages are weighted survey estimates; sample counts are unweighted."

5. Organize logically: Overall findings → By geography → By demographic factor → Model results

Return:
1. Revised Results section
2. Updated table captions (all must stand alone without reference to text)
3. A list of Figure/Table references for each major claim (proof that it's linked)
```

---

### Step 7: DISCUSSION

When ready for Discussion:

```
Use the research-paper-writing skill.

I'm revising my DISCUSSION section. Apply Discussion-to-Results Alignment and Limitations Framing.

CURRENT DISCUSSION:
[PASTE YOUR DISCUSSION HERE]

DISCUSSION STRUCTURE CHECK (you should have):
□ Opening: Restate your key findings (numbers, effect sizes) before interpreting
□ Paragraph 2+: Each finding compared to cited literature (How do YOUR effect sizes match/differ from prior work?)
□ Mechanisms: Why might these predictors drive HWT adoption? (socio-economic pathway? infrastructure access? cultural?)
□ Limitations: Honest statement of what your data cannot show
□ Implications: What do results mean for policy, practice, or future research?
□ Conclusion: Bring it together with one final statement

REVISION GOALS:
1. Open each paragraph with a restatement of results
   - Before: "Our findings align with existing literature."
   - After: "We found 41.7% of households treat water; 28.3% use effective methods (boiling, filtration). Wealth (AOR=4.4) and education (AOR=1.83) were the strongest predictors. This aligns with Daniel et al. (2019) who found similar education effects in Nepal, but our effect size is comparable to their rural-only findings despite including urban areas."

2. Compare effect sizes numerically
   - Before: "Similar to prior studies."
   - After: "We found education (AOR=1.83); Daniel et al. (2019) reported 1.67 in rural Nepal; our effect is slightly larger, possibly due to including urban populations with wider educational variation."

3. Address limitations with implications
   - Before: "A limitation is cross-sectional data."
   - After: "NFHS-5 is cross-sectional, preventing causal inference about whether wealth drives HWT adoption or vice versa. Longitudinal data would clarify whether improved household economics lead to increased treatment adoption, informing the timing of subsidy-based interventions."

4. End with policy/practice implications
   - Link findings to SDG targets (SDG 6)
   - Recommend specific interventions for vulnerable groups (Scheduled Castes, Bihar/UP)
   - Explain how findings should shape programs (e.g., target education campaigns in rural areas with low literacy)

Return:
1. Revised Discussion section
2. A paragraph-by-paragraph outline showing: Result stated | Literature comparison | Mechanism | Implication
3. Revised Limitations subsection with implication statements (not just problem lists)
```

---

### Step 8: CONCLUSION

When ready for Conclusion:

```
Use the research-paper-writing skill.

I'm revising my CONCLUSION. Apply Academic Writing Clarity Rules.

CURRENT CONCLUSION:
[PASTE YOUR CONCLUSION HERE]

CONCLUSION REQUIREMENTS:
- Restate the research question
- Summarize your key findings (1-2 sentences with numbers)
- State the broader implication (Why does this matter?)
- Recommend actions (Policy? Programs? Future research?)
- End with a forward-looking statement (not "more research is needed")

REVISION GOALS:
1. First sentence: Remind reader of the problem and research question
2. Second sentence: State your main finding with one key number
3. Third+ sentences: What this means for water safety, health equity, policy, or SDGs
4. Concrete recommendation: Name a specific action (e.g., "Targeted boiling promotion in rural Bihar could reach 8 million households if integrated with the Jal Jeevan Mission")
5. Avoid: "Future studies," "further research," "limitations constrain conclusions"

Return:
1. Revised Conclusion
2. A 1-sentence check: Can someone read only the Abstract + Conclusion and understand your full contribution?
```

---

## PHASE 3: FINAL POLISH & PUBLICATION READINESS

### Step 9: FULL PAPER SELF-REVIEW

When you've revised all sections, copy and use:

```
Use the research-paper-writing skill.

I've completed my full paper draft. Perform a final adversarial review before I submit to [JOURNAL NAME].

PAPER TITLE: [Your title]
JOURNAL: [Target journal]
SUBMISSION DEADLINE: [If applicable]

SELF-REVIEW CHECKLIST:

Return an assessment on these 5 dimensions:

1. CONTRIBUTION:
   - Is my research question clear and important?
   - Do my results answer that question?
   - Do I state what's NEW compared to prior work?

2. WRITING CLARITY:
   - Can a reader understand the paper without reading it twice?
   - Are all technical terms defined?
   - Is terminology consistent throughout?

3. EMPIRICAL STRENGTH:
   - Are all statistics properly reported (point estimate + 95% CI + p-value)?
   - Are effect sizes interpreted concretely?
   - Is sample size adequate? (Yes: n = 636,699)
   - Are limitations acknowledged?

4. RESULTS COMPLETENESS:
   - Does every key finding have a Table or Figure reference?
   - Are all tables and figures self-contained (can be understood without text)?
   - Do I interpret unexpected findings?

5. METHOD SOUNDNESS:
   - Are all design choices justified?
   - Are potential biases acknowledged?
   - Is the analysis reproducible?

For each dimension, provide:
- Red flags (must fix before submission)
- Yellow flags (should fix but not blocking)
- Green checks (strength of the paper)

Return a prioritized action list for final revisions (Top 3 things to fix now).
```

---

### Step 10: FINAL CHECKLIST BEFORE SUBMISSION

Run through this checklist manually:

```
BEFORE YOU HIT SUBMIT:

☐ Title: Does it accurately reflect the paper? (Is it <15 words?)
☐ Author list: Correct names, affiliations, email addresses
☐ Abstract: <250 words (or journal limit), includes numbers from Results
☐ Keywords: 5-8 relevant terms, aligned with journal scope
☐ Sections in order: Abstract → Intro → Lit Review → Methods → Results → Discussion → Conclusion
☐ References: All citations in text appear in reference list (and vice versa)
☐ Figures: Are they high-resolution (300 DPI for print)? Captions self-contained?
☐ Tables: Captions self-contained? Footnotes explain abbreviations?
☐ Numbers consistent: If you say "41.7%" in Abstract, does it match Results section?
☐ Grammar/spelling: Run through spelling checker; read aloud for flow
☐ Jargon check: Any field-specific terms that readers outside your field won't understand?
☐ Tone: Formal and objective throughout (no first person, no emotional language)
☐ Word count: Within journal limits? (Check target journal's guidelines)
☐ Supplementary files: Are they well-organized and clearly labeled?
☐ Data availability statement: You have one? (Yes: "publicly available NFHS-5")
☐ Funding/conflict of interest: Declared correctly
☐ Reviewer response prep: Do you have 3-5 anticipated criticisms and responses ready?
```

---

## TEMPLATE QUICK REFERENCE

When starting each section, always use this template structure:

```
Use the research-paper-writing skill.

Section: [SECTION NAME]
Current text:
[PASTE YOUR TEXT]

Context for this section:
- Main point I want to make: [1 sentence]
- Evidence I have: [Tables/Figures/Citations]
- Audience: [Peer reviewers at X journal]

Revision focus: [Pick 1-3 of: clarity, jargon, flow, citations, evidence-support, tone]

Return:
1. Revised section
2. A 3-bullet summary of changes
3. Any claim-evidence map or reverse outline as needed
```

---

## HOW TO USE THIS FILE

**Two modes:**

1. **Interactive Mode (Easiest):**
   - Copy the "Quick Start" prompt at the top
   - Send it to chat
   - AI asks you all the questions
   - AI does the work automatically

2. **Self-Guided Mode (if you prefer to fill forms):**
   - Open the step you want (Step 1, Step 2, etc.)
   - Copy the prompt
   - Fill in all the [Bracketed] fields yourself
   - Paste into chat

### Step-by-step instructions:

1. **Start with Quick Start** — Let AI ask you questions (recommended)
   - OR manually fill Step 1 and get assessment
2. **After assessment**, move to Steps 2–8 — Do one section at a time
   - For each section, either:
     - Copy Step prompt + fill fields manually, OR
     - Send: "Use the research-paper-writing skill. Revise my ABSTRACT. Here it is: [paste text]. Main finding: [state it]. Target journal: [name it]"
   - AI will ask for any missing context
3. **Step 9** (Full Paper Review) — Run AFTER all sections revised
4. **Step 10** — Final checklist before submit

**Time estimate per section: 2–4 hours (depending on draft quality)**
**Total workflow: 1–2 weeks for a full revision cycle**

---

## REMEMBER

- **One prompt per session** — don't try to revise everything at once
- **Use the skill actively** — copy prompts exactly; provide all requested inputs
- **Iterate 2–3 times per section** if the first revision isn't perfect
- **Trust the process** — this workflow has been tested on published papers
- **Check the skill file** for examples and standard formats: `/research-paper-writing/SKILL.md`
- **You can mix modes** — use Quick Start for assessment, then manually fill prompts for specific sections if you prefer

Good luck! Your paper is publishable. Let's make it shine.
