# Reviewer 1 Comment 1

Morphology is evaluated as a direct LM partitioning task without CBOW or SkipGram embeddings.
Evaluation strategy: `morphology`
LM method: `lm-rank`
Embedding model: none; this task directly evaluates the selected `AbstractLM` child partitioning.
Ablation variants: 28

Evaluations use `ExtrinsicNER`, `ExtrinsicPOS`, `ExtrinsicSentiment`, `IntrinsicEvaluation`, or direct `ExtrinsicMorphology` scoring depending on the selected strategy.
The CSV file beside this summary records all tuned LM parameters and direct accuracy/F1 scores for each variant.

Best accuracy: `v11` accuracy=0.509582, f1=0.493722.
Best F1: `v11` accuracy=0.509582, f1=0.493722.

Status counts:
- `found`: 28
