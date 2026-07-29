# Reviewer 1 Comment 1 Design Matrix

Call `new Ablation().experiments()` to run every evaluation strategy and LM method in parallel.

Call `new Ablation().experiments(task, method)` for a targeted run with task `POS`, `NER`, `Sentiment`, `Analogy`, or `Morphology` and method `FrequentLM`, `LemmaLM`, `RankLM`, `SkipLM`, `SyllableLM`, or `LMSubword`.

The runner trains or loads the selected LM, constructs the task corpus when needed, trains SkipGram on that corpus for POS/NER/Sentiment/Analogy, and writes evaluation XML plus the reviewer CSV/summary under `resources/results`.
Morphology ablations do not train CBOW or SkipGram; they directly partition `resources/evaluation/morphology` sentences with the selected `AbstractLM` child and report accuracy/F1.

For `RankLM` and `LemmaLM`, ablation includes Algorithm 4 likelihood/prior damping weights and length penalty formulations.

Total ablation parameter variants across all tasks: 475

Variants per method:
- `frequent-ngram`: 9 per task
- `lm-lemma`: 28 per task
- `lm-rank`: 28 per task
- `lm-skip`: 9 per task
- `lm-syllable`: 9 per task
- `lm-subword`: 12 per task
