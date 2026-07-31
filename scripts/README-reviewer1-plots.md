# Reviewer 1 result plots

Install the plotting dependency:

```bash
python3 -m pip install -r scripts/plot-requirements.txt
```

Run from any directory:

```bash
python3 scripts/plot_reviewer1_results.py
```

By default, the script reads `resources/results/reviewer1/comment1-*.csv` and
writes joined data, summary tables, a findings report, and PDF figures under
`resources/results/reviewer1/plots`.

Useful options:

```bash
python3 scripts/plot_reviewer1_results.py \
  --input-dir resources/results/reviewer1 \
  --output-dir resources/results/reviewer1/plots
```

The experiment index CSVs often leave their score columns empty. The script
automatically loads Accuracy, Precision, Recall, and F-Score from the XML or CSV
path in `result_filename`. Missing legacy Precision/Recall values remain missing
and are reported as such.

Analogy F-Score values remain in `combined_results.csv` for traceability, but
the script excludes them from comparisons, rankings, and plots against Accuracy.
Figures use only Accuracy and F-Score; Precision and Recall remain available in
the generated CSV analysis tables.

The script also computes LM graph density from the first 10,000 non-empty
sentences of each stored partition corpus and writes:

- `lm_graph_density_<task>.pdf` for each task;
- `lm_graph_density_vs_fscore.pdf` combining the four F-Measure tasks;
- `lm_graph_density.csv` with the plotted values and density components;
- `lm_graph_density_correlations.csv` with descriptive Pearson correlations;
- `lm_graph_density_3x2_table.tex` for the five-panel LaTeX table.

Compile the combined density table from its output directory:

```bash
cd resources/results/reviewer1/plots
pdflatex -interaction=nonstopmode -halt-on-error \
  lm_graph_density_3x2_table.tex
```

Morphology does not save a task-specific partition corpus. Its density panel
therefore uses the mean density of the same LM variant across the four stored
partition corpora and labels this as a reference-corpus estimate.

The average-distinct-n-gram analysis writes `avg-intrinsic.pdf`,
`avg-morphology.pdf`, `avg-ner.pdf`, `avg-pos.pdf`, and `avg-sentiment.pdf`,
together with
`average_distinct_ngrams.csv`, `average_distinct_ngrams_correlations.csv`, and
the LaTeX table sources `average_distinct_ngrams_2x2_table.tex` and
`average_distinct_ngrams_3x2_table.tex`. Intrinsic uses Accuracy; the other
four panels use F-Measure. Morphology uses the mean value for the same LM
variant across the four stored partition corpora because its direct evaluator
does not save a partitioned corpus.

`all_findings_selected_metrics.tex` is a multipage LaTeX `longtable` containing
all 480 variants, all nine reported LM parameters, and the selected task
result. It uses Accuracy for Analogy and F-Measure for every other task, and
flags legacy sequence-labeling scores that lack Precision and Recall.
