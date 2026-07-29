# Reviewer 1 Comment 1

Morphology is evaluated as a direct LM partitioning task without CBOW or SkipGram embeddings.
Evaluation strategy: `ner`
LM method: `lm-syllable`
Embedding model: `SkipGramModel`
Ablation variants: 9

Evaluations use `ExtrinsicNER`, `ExtrinsicPOS`, `ExtrinsicSentiment`, `IntrinsicEvaluation`, or direct `ExtrinsicMorphology` scoring depending on the selected strategy.
POS, NER, and Sentiment use their original fixed training and testing datasets.
The CSV file beside this summary records all tuned LM parameters and the result XML path for each variant.

Status counts:
- `found`: 9
