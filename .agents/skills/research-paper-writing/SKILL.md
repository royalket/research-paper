---
name: research-paper-writing
description: Improve academic paper writing quality for ML/CV/NLP-style papers with clear section structure, paragraph flow, and reviewer-facing presentation. Use when drafting or revising Abstract, Introduction, Related Work, Method, Experiments, or Conclusion; polishing figures/tables; checking claim-support alignment; or performing self-review before submission.
---
# Research Paper Writing

## Overview

Use this skill to rewrite a research paper into a reviewer-friendly, high-clarity draft.
Prioritize first-impression quality (figures/tables/layout), logical flow, and evidence-backed claims.

## Core Workflow

1. Clarify the paper story before sentence-level edits.
2. Use section-specific guidance in `references/`.
3. Rewrite paragraph-by-paragraph with one message per paragraph.
4. Run reverse outlining after writing each section.
5. Check every major claim in Abstract/Introduction against experimental evidence.
6. Run final-paper adversarial review with `references/paper-review.md`.

## Global Principles

1. Keep one paragraph for one message only.
2. State the paragraph message in the first sentence.
3. Make nouns self-contained; define new terms before reusing them.
4. Maintain sentence-to-sentence flow (cause, contrast, consequence, or refinement).
5. Iterate with adversarial self-review: read as a skeptical reviewer.
6. Treat visual quality as core content, not decoration.
7. Use a clean teaser and pipeline figure.
8. Use readable, minimal-ink tables.
9. Keep formatting consistent and tidy.

## Academic Writing Clarity Rules

Apply these rules to maintain formal, evidence-grounded prose while removing unnecessary complexity.

1. **Remove jargon:** Replace field-specific buzzwords with precise, plain terms unless the term is standard in your venue.
   - Before: "This approach leverages synergistic mechanisms."
   - After: "This approach combines X and Y."

2. **Fix attribution:** Replace vague claims ("experts say", "it is known that") with specific sources or evidence.
   - Before: "Recent studies show significant improvements."
   - After: "Smith et al. (2023) reported a 23% improvement on benchmark X."

3. **Simplify overcomplexity:** Break nested clauses and remove redundant qualifiers while keeping formality.
   - Before: "The experimental results, which demonstrated clear and measurable improvements, indicate that our method is potentially superior."
   - After: "Results show that our method improves accuracy by 5% on standard benchmarks."

4. **Keep formality:** Use third person and objective tone; avoid personal commentary.
   - Before: "I believe this is important because..."
   - After: "This is important because the evidence shows..."

5. **Verify every claim:** Each factual or empirical claim must be supported by a citation, experiment, or table reference.

6. **Avoid filler language:** Cut phrases like "notably," "interestingly," "it is worth noting," "in light of the fact that."

## Empirical Results Reporting (for quantitative papers)

Apply these standards when writing Results sections with statistical findings.

1. **State point estimates with confidence intervals:** Always report 95% CI or credible intervals, not just p-values.
   - Before: "The effect was significant (p < 0.001)."
   - After: "Households in the richest wealth quintile had 4.4× higher odds of water treatment (95% CI: 4.27–4.59, p < 0.001)."

2. **Interpret effect sizes concretely:** Translate odds ratios, hazard ratios, or standardized coefficients into real-world meaning.
   - Before: "AOR = 1.830 for higher education."
   - After: "Households with higher education had 1.83× higher odds of treating water (83% increase)."

3. **Make table captions self-contained:** A reader should understand the table without reading surrounding text.
   - Before: "Table 3 presents the results."
   - After: "Table 3. Weighted descriptive statistics for predictor variables and overall water treatment prevalence (41.7%), NFHS-5 (n = 636,699). Percentages are weighted estimates; sample counts are unweighted."

4. **Link every key result to a figure or table:** Avoid stating results without reference.
   - Before: "Water treatment varies by region."
   - After: "Water treatment prevalence ranged from 95% in Nagaland and Kerala to <10% in Bihar (Figure 1, Table S4)."

## Discussion-to-Results Alignment

When writing Discussion sections:

1. **Open with result summaries:** Restate key findings (percentages, effect sizes) before interpretation.
   - Before: "Our findings align with existing literature."
   - After: "We found 41.7% of households treat water, with boiling (38.3%) and cloth straining (35.6%) most common. The strongest predictors were wealth (AOR=4.4) and education (AOR=1.83)."

2. **Compare to cited studies:** Show how your effect sizes compare to prior work.
   - Before: "Similar to Daniel et al."
   - After: "Daniel et al. (2019) reported education as a key predictor in rural Nepal; we found a comparable effect (AOR=1.83) across all Indian households."

3. **Address limitations honestly:** Acknowledge what your data cannot show.
   - Before: "A limitation is reliance on secondary data."
   - After: "NFHS data do not include direct water quality measurements; our analysis relies on self-reported treatment practices, which may overestimate actual effectiveness."

## Limitations Framing

Structure limitations to advance, not weaken, your contribution:

1. **State the limitation clearly.**
2. **Explain why it matters (not just "future work needs to...").**
3. **Link to implications: What would change if this limitation were addressed?**

Example for your paper:
- Before: "A limitation is that NFHS data are cross-sectional."
- After: "NFHS-5 is cross-sectional, preventing causal inference about whether wealth drives water treatment adoption or vice versa. Longitudinal data would clarify whether improved household economics lead to increased treatment adoption, which would inform timing of intervention strategies."

## Paragraph Clarity Check (Important)

Use this quick test whenever the user asks whether a paragraph "flows" or is clear.

1. Read as an external reader:
   - Does this paragraph have one explicit message?
   - Does the first sentence state what this paragraph will do?
   - Are all key nouns/terms readable without hidden context?
   - Does each sentence connect to the previous one with a clear relation (cause, contrast, consequence, refinement, example)?
2. Run reverse outlining for the current section:
   - Write down thesis/main claim.
   - Write down each paragraph topic sentence.
   - Write down the evidence/explanation points under each paragraph.
   - Check mapping: topic sentence -> thesis, and evidence -> topic sentence.
   - Revise or remove any paragraph that cannot be mapped cleanly.
3. If flow is still weak, add temporary section headers and explicit transition phrases during revision, then remove unnecessary headers before finalizing.

Source reference for this check:

- `references/does-my-writing-flow-source.md`

## Section Guides

Load only the needed section file:

- Introduction: `references/introduction.md`
- Abstract: `references/abstract.md`
- Related Work: `references/related-work.md`
- Method: `references/method.md`
- Experiments: `references/experiments.md`
- Conclusion: `references/conclusion.md`
- Paper review (Paper Review): `references/paper-review.md`
- Paragraph clarity source: `references/does-my-writing-flow-source.md`
- Example bank index: `references/examples/index.md`

## Paper Review Core Points

Use `references/paper-review.md` for the full checklist and workflow.

1. Add an end-of-draft self-review question list in five dimensions:
   - contribution,
   - writing clarity,
   - experimental strength,
   - evaluation completeness,
   - method design soundness.
2. Treat claim-evidence alignment as a hard constraint, especially for Abstract and Introduction.
3. Perform adversarial writing: review as a skeptical reviewer and resolve every high-risk question.
4. Revise until major rejection risks are explicitly addressed.

## Execution Rules

1. Build a mini-outline before drafting prose.
2. For each subsection, explicitly include motivation, design, and technical advantage when applicable.
3. Avoid writing style that looks like incremental patching of a naive baseline.
4. Keep terminology stable across the full paper.
5. If a claim cannot be supported by results, weaken or remove the claim.
6. Before finalizing, append and answer a five-dimension self-review question list, then revise the paper based on unresolved items.
7. Do not load all section references (Introduction/Abstract/Related Work/Method/Experiments/Conclusion) at once; load only the specific section guide needed for the current edit target.

## Output Contract

When asked to rewrite or draft sections, return:

1. A compact section outline (3-7 bullets).
2. Revised paragraphs with explicit paragraph roles (opening/challenge/method/advantage/evidence/limitation).
3. A short self-review checklist covering clarity, flow, terminology consistency, unsupported claims, and missing evidence.
4. A claim-evidence map for each major claim in the revised text using `Claim: ... | Evidence: ... | Status: supported/needs evidence`.
