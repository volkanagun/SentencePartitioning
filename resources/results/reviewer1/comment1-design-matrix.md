# Reviewer 1 Comment 1 Design Matrix

Call `new Reviewer1().comment1()` to run every evaluation strategy and LM method in parallel.

Call `new Reviewer1().comment1(task, method)` for a targeted run with task `POS`, `NER`, `Sentiment`, or `Analogy` and method `FrequentLM`, `LemmaLM`, `RankLM`, `SkipLM`, `SyllableLM`, or `LMSubword`.

The runner trains or loads the selected LM, constructs the task corpus when needed, trains SkipGram on that corpus, and writes evaluation XML plus the reviewer CSV/summary under `resources/results`.

For `RankLM`, ablation includes Algorithm 4 likelihood/prior damping weights and length penalty formulations.
