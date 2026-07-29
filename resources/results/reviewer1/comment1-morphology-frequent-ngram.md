# Reviewer 1 Comment 1

Morphology is evaluated as a direct LM partitioning task without CBOW or SkipGram embeddings.
Evaluation strategy: `morphology`
LM method: `frequent-ngram`
Embedding model: none; this task directly evaluates the selected `AbstractLM` child partitioning.
Ablation variants: 9

Evaluations use `ExtrinsicNER`, `ExtrinsicPOS`, `ExtrinsicSentiment`, `IntrinsicEvaluation`, or direct `ExtrinsicMorphology` scoring depending on the selected strategy.
The CSV file beside this summary records all tuned LM parameters and direct accuracy/F1 scores for each variant.

Best accuracy: `v1` accuracy=0.271060, f1=0.345762.
Best F1: `v1` accuracy=0.271060, f1=0.345762.

Status counts:
- `found`: 9
