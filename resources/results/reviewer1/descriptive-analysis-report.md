# Descriptive analysis of the Reviewer 1 ablation experiments

## Scope

This report describes the ablation results recorded under
`resources/results/reviewer1`. The analysis covers 480 variants distributed
equally across five tasks: Analogy, Morphology, Named Entity Recognition (NER),
Part-of-Speech tagging (POS), and Sentiment classification. Seven
language-model methods are represented: `frequent-ngram`, `lm-lemma`,
`lm-rank`, `lm-skip`, `lm-subword`, `lm-syllable`, and the newly added
`lm-word` baseline.

The experiment-index CSV files were joined with the XML or CSV artifact named
in each row's `result_filename`. This step was necessary because many index
rows contain the experimental parameters but leave their score fields empty.
The resulting joined data are available in
[combined_results.csv](plots/combined_results.csv).

This is a descriptive analysis of fixed train/test evaluations. It does not
use paired t-tests, or other significance tests. Differences
are therefore discussed as observed patterns, not as evidence of statistical
significance.

## Overall findings

The results do not identify one method that dominates every task. Instead,
method effectiveness depends on the downstream objective:

- `lm-lemma` is the most consistently competitive method. It has the highest
  mean Analogy Accuracy, the highest mean Morphology F-Score, the strongest
  comparable POS F-Score, and competitive NER and Sentiment results.
- `lm-skip` produces the highest observed Morphology and Sentiment F-Scores,
  but its Morphology distribution is much more variable than that of
  `lm-lemma`. Its apparent POS advantage comes from legacy result files and
  should not be interpreted as a current macro-F1 result.
- `lm-subword` produces the best single Analogy Accuracy and the best
  Sentiment Accuracy, but it is not uniformly strongest across all variants.
- `lm-rank` is particularly effective for Morphology Accuracy, but it is
  weaker on the comparable NER and POS F-Score evaluations.
- `frequent-ngram` usually provides a lower baseline, especially for
  Sentiment, POS, and NER, where its current mean F-Score is 0.400389.
- `lm-syllable` is competitive for NER and Sentiment. In particular, it
  supplies the best fully measured NER F-Score variant.
- The single `lm-word` baseline is highly task-dependent. It is weak on
  Analogy, NER, and POS, matches `frequent-ngram` on Morphology, but is
  competitive on Sentiment with 0.828200 Accuracy and 0.651535 F-Score.

These findings favor task-specific model selection over a universal choice.
If a single generally reliable method is required, `lm-lemma` has the
strongest cross-task consistency. If task-specific tuning is allowed,
`lm-rank`, `lm-skip`, `lm-subword`, and `lm-syllable` each provide an optimum
for at least one task or metric.

The `lm-word` results should be treated as baseline points rather than method
distributions because only one configuration was evaluated per task. Its
reported standard deviation of zero is a consequence of this sample size and
does not demonstrate robustness.

## Reading the figures

### Metric coverage

[Metric coverage](plots/metric_coverage.pdf) shows that Accuracy is available
for all 480 variants. F-Score is also present in the source artifacts, but the
Analogy F-Score is deliberately omitted from the plot because its legacy
`F1-MEASURE` field stores an unbounded similarity value rather than a
classification F-Score.

The plot should not be interpreted as proof that all extrinsic F-Scores are
equally comparable. Nine POS rows have an F-Score but no associated Precision
or Recall. All are `lm-skip` artifacts that predate the current macro-F1
implementation. These rows remain in the traceability tables but are excluded
from comparative plots, method means, and sensitivity estimates. All 96 NER
variants now contain complete Accuracy, Precision, Recall, and F-Score fields.

### Best-method heatmaps

[Best-method heatmaps](plots/best_method_heatmaps.pdf) present the best
observed Accuracy and F-Score for every task-method combination. Color is
normalized within a task, so it communicates within-task ordering rather than
absolute comparability across tasks. The annotated number is the value that
should be used when reporting a result.

The heatmaps show several important contrasts:

- Analogy Accuracy clearly favors `lm-lemma` and `lm-subword`; the Analogy
  F-Score row is intentionally blank.
- Morphology Accuracy favors `lm-rank`, whereas the highest single Morphology
  F-Score belongs to `lm-skip`.
- NER Accuracy is tightly grouped across methods. The larger separation in
  F-Score indicates that Accuracy alone conceals meaningful differences in
  label-level performance.
- For POS, the `lm-skip` F-Score cell is blank because all nine available
  values are legacy artifacts equal to Accuracy and are not comparable with
  current macro-F1 values.
- Sentiment has a relatively compact leading group. `lm-subword` gives the
  highest Accuracy, while `lm-skip` gives the highest F-Score.
- `lm-word` appears as one additional point per task. It is lowest on Analogy,
  NER, and POS, ties the `frequent-ngram` Morphology result, and falls within
  the leading Sentiment range without exceeding the best tuned variant.

Because each heatmap cell selects the maximum from several variants, it is an
optimistic summary. Method distributions and mean scores are more useful for
assessing consistency.

### Score distributions

The task-level distribution figures show how much each method changes across
the tested ablation settings:

- [Analogy](plots/distributions_analogy.pdf)
- [Morphology](plots/distributions_morphology.pdf)
- [NER](plots/distributions_ner.pdf)
- [POS](plots/distributions_pos.pdf)
- [Sentiment](plots/distributions_sentiment.pdf)

The narrow Analogy distribution for `lm-lemma` indicates stable performance
near the top of the observed range. `lm-subword` reaches the highest individual
score but varies more across configurations. `lm-rank` and `lm-syllable` remain
close to zero in most Analogy configurations. The `lm-word` baseline is also
near zero, with Accuracy 0.000143.

Morphology provides the clearest example of the difference between peak and
typical performance. `lm-skip/v1` reaches the highest F-Score (0.538576), but
the method's mean is only 0.323134 with a standard deviation of 0.164022.
In contrast, every evaluated `lm-lemma` configuration has an F-Score of
0.475877. Thus, `lm-skip` provides the best observed point, while `lm-lemma`
provides the more robust result under the tested settings. The single
`lm-word` point has 0.271060 Accuracy and 0.345762 F-Score, exactly matching
the `frequent-ngram` baseline scores.

NER Accuracy occupies a narrow high range, from 0.920829 to 0.953490. This
compression is consistent with a token-classification task in which common
labels and padding can dominate Accuracy. Macro-F1 is more discriminating,
although legacy rows must first be separated from current rows. `lm-word`
extends the lower end to 0.859397 Accuracy and has only 0.126547 F-Score.

POS shows a similar Accuracy/F-Score distinction. Among the current,
fully measured variants, `lm-lemma` has the strongest mean F-Score and a
relatively narrow distribution. The legacy `lm-skip` values are excluded from
the distribution and should not be used to claim an F-Score advantage. The
`lm-word` baseline is substantially lower, at 0.550511 Accuracy and 0.026706
F-Score.

Sentiment has complete current metrics for all 96 variants. The leading four
methods have close mean F-Scores: `lm-skip` 0.631789, `lm-syllable` 0.629367,
`lm-lemma` 0.627032, and `lm-subword` 0.624878. These differences are small
and have not been tested for significance. `frequent-ngram` is more clearly
separated, with a mean F-Score of 0.565937. The single `lm-word` result reaches
0.651535 F-Score and 0.828200 Accuracy. Its score is competitive with the
leading methods, but its one-run “mean” cannot be interpreted as a stability
advantage over their multi-variant distributions.

### Accuracy and F-Score relationship

[Classification Accuracy versus F-Score](plots/classification_accuracy_vs_fscore.pdf)
contains Morphology, NER, POS, and Sentiment only. Analogy is excluded because
its second score is a similarity measure rather than a classification
F-Score.

For rows with current complete metrics, Accuracy and F-Score are strongly
positively associated: the descriptive Pearson correlations are 0.889 for
Morphology, 0.961 for NER, 0.958 for POS, and 0.928 for Sentiment. These
values describe the plotted sample and are not significance-test results. The
relationship is not
one-to-one: configurations with similar Accuracy can still differ in F-Score,
and the configuration that maximizes Accuracy need not maximize F-Score. This
is most visible in Sentiment, where the best Accuracy and best F-Score come
from different methods.

### Hyperparameter sensitivity

[Parameter sensitivity](plots/parameter_sensitivity.pdf) reports a
descriptive effect measure: the range of parameter-level mean scores divided
by the observed score range for that task, method, and metric. Larger values
identify parameters associated with more variation in the tested grid. The
figure uses Analogy Accuracy and F-Score for Morphology, NER, POS, and
Sentiment, matching the metrics selected for the distribution analysis.

`lm_top_split` is the strongest median descriptive effect for every selected
task metric:

- Analogy Accuracy: 0.83
- Morphology F-Score: 0.83
- NER F-Score: 0.51
- POS F-Score: 0.53
- Sentiment F-Score: 0.56

For NER, likelihood and prior weights are the next-largest descriptive effects
at 0.33 each, followed by the coupled window, slide, and skip parameters at
0.27 each. For Sentiment F-Score, length penalty is the second-largest effect
at 0.30.

These values must not be interpreted as independent causal effects. Window
length, slide length, and skip length vary together in much of the design,
which explains their repeated sensitivity patterns. Likelihood and prior
weights are also complementary and therefore cannot be interpreted as
independent predictors. The heatmap is useful for screening and prioritizing
future experiments, not for attributing causality.

### LM graph density and task performance

[LM graph density](plots/lm_graph_density_3x2_table.pdf) compares the graph
density induced by each LM-partitioned corpus with the selected task score.
Analogy uses Accuracy; Morphology, NER, POS, and Sentiment use F-Score. Density
is displayed on a logarithmic scale, and the marker shape identifies the LM
window length. The plotted NER and POS subsets exclude legacy F-Scores that do
not have accompanying Precision and Recall.

The descriptive Pearson correlations with log-density are:

- Analogy Accuracy: -0.380 (96 variants)
- Morphology F-Score: -0.043 (96 variants)
- NER F-Score: 0.224 (96 variants)
- POS F-Score: 0.295 (87 current variants)
- Sentiment F-Score: 0.083 (96 variants)

Accordingly, graph density has no consistent relationship with performance
across tasks. Morphology and Sentiment have approximately null associations,
while NER and POS have weak positive associations. Analogy differs by showing
a moderate negative association. These are descriptive correlations rather
than significance-test results, and they combine changes in LM method, window
size, and other partition parameters.

Morphology requires a qualification. Its evaluator partitions its input
directly and does not save a task-specific partition corpus. Its panel
therefore uses the mean graph density of the corresponding LM variant across
the four stored partition corpora as a reference estimate. This approximation
is clearly marked in the figure and should not be presented as a directly
measured Morphology-corpus density.

### Average distinct n-grams per sentence

The [five-task distinct-n-gram table](plots/average_distinct_ngrams_3x2_table.pdf)
relates local partition diversity to the selected task metric. For each
sampled sentence, the analysis counts its unique LM partitions and then
averages that count across at most 10,000 non-empty sentences. Intrinsic uses
Accuracy; Morphology, NER, POS, and Sentiment use F-Score.

The descriptive Pearson correlations are:

- Intrinsic Accuracy: -0.255 (96 variants)
- Morphology F-Score: -0.376 (96 variants; reference-corpus estimate)
- NER F-Score: -0.003 (96 variants)
- POS F-Score: 0.146 (87 current variants)
- Sentiment F-Score: -0.043 (96 variants)

The directions differ by task, and none of the results supports a universal
benefit from increasing the number of distinct partitions. NER and Sentiment
are approximately uncorrelated, POS has a weak positive association, and
Intrinsic and Morphology show negative descriptive associations. These
patterns suggest that partition identity and linguistic relevance matter more
than partition quantity alone. They are descriptive associations rather than
significance-test results.

As with graph density, Morphology has no saved task-specific partition
corpus. Its x-axis values are means for the corresponding LM variants across
the four stored partition corpora. This limitation is reported explicitly and
should be resolved by saving directly partitioned Morphology sentences in a
future evaluation.

## Task-specific interpretation

### Analogy

Only Accuracy is used for interpretation. The best individual result is
`lm-subword/v8` at 0.037762. However, `lm-lemma` has the highest method mean
(0.031249), compared with 0.025318 for `lm-subword`. This distinction suggests
that subword modeling can produce the strongest tuned configuration, whereas
lemma modeling is more consistently effective across the tested grid.

The absolute Accuracy values are low: no variant exceeds 0.038. The result
should therefore be described as a relative ranking among ablations rather
than strong absolute Analogy performance. The legacy unbounded similarity
values remain in the joined data for traceability but are not analyzed as
F-Scores.

`lm-word/v1` records 0.000143 Accuracy, the lowest Analogy result among the
method-level baselines. This indicates that unpartitioned word units alone do
not recover the relations measured by this analogy evaluation. Its
`F1-MEASURE` value is the same legacy unbounded similarity output and is not
compared.

### Morphology

`lm-rank/v11` provides the highest Accuracy at 0.509582. `lm-rank` also has the
highest mean Accuracy (0.473326), supporting a reasonably consistent Accuracy
advantage.

The F-Score result is more nuanced. `lm-skip/v1` gives the best observed
F-Score at 0.538576, but its large variance makes this result sensitive to
configuration. `lm-lemma` has a lower maximum of 0.475877 but achieves that
same value across all 28 configurations. Therefore:

- choose `lm-rank` when Morphology Accuracy is primary;
- choose tuned `lm-skip/v1` for the highest observed F-Score;
- choose `lm-lemma` when robustness to the tested parameter changes is more
  important than the single highest score.

`lm-word/v1` reaches 0.271060 Accuracy and 0.345762 F-Score. It neither
improves on the best morphology-aware methods nor falls below the
`frequent-ngram` baseline; the two methods have identical reported scores.

### NER

All 96 NER variants contain current complete metrics. The best Accuracy is
0.953490 from `lm-lemma/v11`. The best F-Score is 0.682859 from
`lm-syllable/v7`, with Accuracy 0.950782.

For method means, `lm-lemma` has the highest F-Score at 0.648402
across 28 variants. It is followed by `lm-skip` (0.590995),
`lm-syllable` (0.589815 across nine variants), `lm-subword`
(0.554834), `lm-rank` (0.498197), and `frequent-ngram` (0.400389).

The gap between high Accuracy and substantially lower macro-F1 indicates that
performance on less frequent entity labels remains the more difficult aspect
of the task. NER conclusions should prioritize current macro-F1 over legacy
F-Score fields.

The fully measured `lm-word/v1` result is substantially weaker: 0.859397
Accuracy, 0.118991 Precision, 0.135128 Recall, and 0.126547 F-Score. Its low
macro-F1 shows that the unpartitioned word baseline does not transfer well to
entity-label discrimination.

### POS

Eighty-seven POS variants contain current complete metrics. Excluding the nine
legacy `lm-skip` rows, `lm-lemma/v8` is the strongest observed configuration
for both Accuracy (0.874796) and F-Score (0.645612). At the method level,
`lm-lemma` also has the highest comparable mean F-Score (0.625654), followed by
`lm-subword` (0.595057), `lm-syllable` (0.592850), `lm-rank` (0.578525), and
`frequent-ngram` (0.471157).

The archived `lm-skip` F-Score is equal to Accuracy because it predates the
current F-Score computation. It is retained in the complete findings table
but excluded from comparative figures and should be presented as historical
Accuracy, not as evidence that `lm-skip` outperforms `lm-lemma` on POS
macro-F1.

`lm-word/v1` produces 0.550511 Accuracy, 0.018011 Precision, 0.051625 Recall,
and 0.026706 F-Score. This is the weakest fully measured POS result and
suggests that the word-level baseline loses information captured by the
linguistically partitioned alternatives.

### Sentiment

Sentiment is the cleanest basis for method comparison because all 96 variants
contain current complete metrics. The best Accuracy is 0.839600 from
`lm-subword/v7`; its F-Score is 0.661130. The best F-Score is 0.678592 from
`lm-skip/v4`; its Accuracy is 0.824800.

This difference demonstrates a genuine metric-dependent choice. `lm-subword`
is preferable when overall Accuracy is the primary criterion, while
`lm-skip/v4` is preferable when class-balanced F-Score is primary. At the
method-mean level, `lm-skip`, `lm-syllable`, `lm-lemma`, and `lm-subword` are
close. A confirmatory evaluation would be needed before claiming that their
small mean differences represent a reliable ordering.

Unlike its sequence-labeling results, `lm-word/v1` is competitive on
Sentiment: 0.828200 Accuracy, 0.718043 Precision, 0.656389 Recall, and
0.651535 F-Score. It exceeds the multi-variant mean F-Score of every method,
but remains below the best tuned `lm-skip` F-Score and the best
`lm-subword` Accuracy. Because `lm-word` has only one configuration, this is
evidence for a strong baseline point, not for superior mean performance or
lower variance.

## Limitations

1. No significance tests are reported. Statements such as “higher” and
   “best” refer only to observed values.
2. Nine `lm-skip` POS result files predate the current Precision, Recall, and
   macro-F1 output. Their F-Score fields are retained for traceability but are
   excluded from comparative analysis.
3. Analogy's legacy `F1-MEASURE` is an unbounded similarity output. It is not
   a classification F-Score and is excluded from figures and rankings.
4. Method means are based on unequal numbers of variants: one for `lm-word`
   and 9, 12, or 28 for the other methods. They summarize the tested grids
   rather than a balanced experimental design.
5. Selecting the maximum across many variants introduces selection optimism.
   Best configurations should be confirmed on untouched data.
6. Several hyperparameters are coupled in the design, so the sensitivity
   plot cannot isolate independent effects.

## Recommended conclusions

The strongest defensible conclusion is that linguistic preprocessing and
language-model structure interact with the downstream task. Lemma-based
modeling is the most consistently strong choice, but specialized alternatives
can improve individual objectives: subword modeling for Analogy and Sentiment
Accuracy, rank modeling for Morphology Accuracy, skip modeling for Morphology
and Sentiment F-Score, and syllable modeling for the best current NER F-Score.
The new `lm-word` baseline reinforces this task dependence: it is clearly
insufficient for Analogy, NER, and POS, neutral relative to
`frequent-ngram` on Morphology, and unexpectedly competitive on Sentiment.

For reporting, the paper should distinguish peak performance from robustness.
The Morphology results in particular show that the best individual variant
and the most stable method are not the same. It should also distinguish
Accuracy from macro-F1 in the sequence-labeling tasks and avoid using legacy
F-Score fields as if they had been produced by the current evaluator.

Before making formal comparative claims, the nine legacy POS evaluations
should be regenerated with the current metric implementation and stored under
versioned result filenames. The selected best variants should then be
confirmed on an untouched evaluation set or through repeated runs if
uncertainty estimates are required.
