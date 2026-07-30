# Reviewer 1 ablation findings

Loaded **480 variants** from **35 experiment index CSV files**. Scores missing from index CSVs were read from each row's `result_filename` artifact.

## Metric coverage

| Task | Variants | Accuracy | Precision | Recall | F-Score |
| --- | --- | --- | --- | --- | --- |
| ANALOGY | 96 | 96/96 | 0/96 | 0/96 | 96/96 |
| MORPHOLOGY | 96 | 96/96 | 0/96 | 0/96 | 96/96 |
| NER | 96 | 96/96 | 96/96 | 96/96 | 96/96 |
| POS | 96 | 96/96 | 87/96 | 87/96 | 96/96 |
| SENTIMENT | 96 | 96/96 | 96/96 | 96/96 | 96/96 |

## Best observed variants

| Task | Metric | Method | Variant | Score |
| --- | --- | --- | --- | --- |
| ANALOGY | Accuracy | lm-subword | v8 | 0.0377615 |
| MORPHOLOGY | Accuracy | lm-rank | v11 | 0.509582 |
| MORPHOLOGY | F-Score | lm-skip | v1 | 0.538576 |
| NER | Accuracy | lm-lemma | v11 | 0.95349 |
| NER | Precision | lm-subword | v4 | 0.817437 |
| NER | Recall | lm-syllable | v7 | 0.624056 |
| NER | F-Score | lm-syllable | v7 | 0.682859 |
| POS | Accuracy | lm-skip | v1 | 0.884458 |
| POS | Precision | lm-lemma | v17 | 0.722016 |
| POS | Recall | lm-subword | v4 | 0.636342 |
| POS | F-Score | lm-lemma | v8 | 0.645612 |
| SENTIMENT | Accuracy | lm-subword | v7 | 0.8396 |
| SENTIMENT | Precision | lm-subword | v7 | 0.779934 |
| SENTIMENT | Recall | lm-skip | v4 | 0.677465 |
| SENTIMENT | F-Score | lm-skip | v4 | 0.678592 |

## Method ranking by mean F-Score

| Task | Rank | Method | Mean | SD | N |
| --- | --- | --- | --- | --- | --- |
| MORPHOLOGY | 1 | lm-lemma | 0.475877 | 0 | 28 |
| MORPHOLOGY | 2 | lm-rank | 0.429881 | 0.0410262 | 28 |
| MORPHOLOGY | 3 | frequent-ngram | 0.345762 | 0 | 9 |
| MORPHOLOGY | 4 | lm-word | 0.345762 | 0 | 1 |
| MORPHOLOGY | 5 | lm-skip | 0.323134 | 0.164022 | 9 |
| MORPHOLOGY | 6 | lm-syllable | 0.263187 | 0.0713022 | 9 |
| MORPHOLOGY | 7 | lm-subword | 0.252133 | 0.0649048 | 12 |
| NER | 1 | lm-lemma | 0.648402 | 0.00734131 | 28 |
| NER | 2 | lm-skip | 0.590995 | 0.0580556 | 9 |
| NER | 3 | lm-syllable | 0.589815 | 0.0677568 | 9 |
| NER | 4 | lm-subword | 0.554834 | 0.0196288 | 12 |
| NER | 5 | lm-rank | 0.498197 | 0.0140684 | 28 |
| NER | 6 | frequent-ngram | 0.400389 | 0.00695055 | 9 |
| NER | 7 | lm-word | 0.126547 | 0 | 1 |
| POS | 1 | lm-lemma | 0.625654 | 0.00714837 | 28 |
| POS | 2 | lm-subword | 0.595057 | 0.0273181 | 12 |
| POS | 3 | lm-syllable | 0.59285 | 0.0287093 | 9 |
| POS | 4 | lm-rank | 0.578525 | 0.0191883 | 28 |
| POS | 5 | frequent-ngram | 0.471157 | 0.00768694 | 9 |
| POS | 6 | lm-word | 0.0267057 | 0 | 1 |
| SENTIMENT | 1 | lm-word | 0.651535 | 0 | 1 |
| SENTIMENT | 2 | lm-skip | 0.631789 | 0.0261567 | 9 |
| SENTIMENT | 3 | lm-syllable | 0.629367 | 0.0207552 | 9 |
| SENTIMENT | 4 | lm-lemma | 0.627032 | 0.00910633 | 28 |
| SENTIMENT | 5 | lm-subword | 0.624878 | 0.0234822 | 12 |
| SENTIMENT | 6 | lm-rank | 0.610341 | 0.0103293 | 28 |
| SENTIMENT | 7 | frequent-ngram | 0.565937 | 0.0119604 | 9 |

## `lm-word` baseline

| Task | Accuracy | F-Score |
| --- | --- | --- |
| ANALOGY | 0.000143308 | Not compared |
| MORPHOLOGY | 0.27106 | 0.345762 |
| NER | 0.859397 | 0.126547 |
| POS | 0.550511 | 0.0267057 |
| SENTIMENT | 0.8282 | 0.651535 |

`lm-word` contributes one baseline variant per task. Its zero standard deviations and method means therefore describe single observations, not stability across configurations.

## Strongest descriptive parameter effects

| Task | Metric | Parameter | Relative effect |
| --- | --- | --- | --- |
| ANALOGY | Accuracy | lm_top_split | 0.828 |
| ANALOGY | Accuracy | lm_prior_weight | 0.294 |
| ANALOGY | Accuracy | lm_likelihood_weight | 0.294 |
| MORPHOLOGY | F-Score | lm_top_split | 0.834 |
| MORPHOLOGY | F-Score | lm_prior_weight | 0.232 |
| MORPHOLOGY | F-Score | lm_likelihood_weight | 0.232 |
| NER | F-Score | lm_top_split | 0.506 |
| NER | F-Score | lm_prior_weight | 0.326 |
| NER | F-Score | lm_likelihood_weight | 0.326 |
| POS | F-Score | lm_top_split | 0.527 |
| POS | F-Score | lm_prior_weight | 0.255 |
| POS | F-Score | lm_likelihood_weight | 0.255 |
| SENTIMENT | F-Score | lm_top_split | 0.557 |
| SENTIMENT | F-Score | lm_length_penalty | 0.298 |
| SENTIMENT | F-Score | lm_prior_weight | 0.220 |

## Interpretation notes

- `f_score` has 62 values outside [0, 1] (range -3536.46–2.80816e+06).
- The unbounded Analogy `F1-MEASURE` values are legacy similarity outputs, not interpretable classification F-Scores; use the Analogy Accuracy column or a separately named similarity measure.
- 9 extrinsic rows have F-Score but no Precision/Recall. These legacy scores should not be assumed to use the current macro-F1 implementation without rerunning evaluation.
- Missing precision and recall indicate legacy result XML files that predate those fields; they are not imputed.
- Parameter effects are descriptive grouped-mean ranges, not causal effects or significance tests.
- Results use the project's fixed train/test evaluations. No cross-validation or paired t-test is performed by this script.
- Analogy F-Score is retained only in the joined raw table; it is excluded from score comparisons, rankings, and figures.
