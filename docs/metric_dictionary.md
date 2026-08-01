# Metric Dictionary

| Metric | Definition | Why it is reported |
|---|---|---|
| Top-1 hit rate | Share of rows where the highest-probability category is the observed next category | Measures single-choice recommendation accuracy |
| Top-3 hit rate | Share where the observed next category appears among the three highest probabilities | Primary ranked-recommendation metric |
| Mean reciprocal rank | Average of `1 / rank` for the observed next category | Rewards placing the correct category near the top |
| Multiclass Log Loss | Negative log probability assigned to the observed category | Penalizes confident wrong predictions |
| Multiclass Brier score | Mean squared distance between category probabilities and the one-hot outcome | Measures probability quality; lower is better |
| Average Precision | Area under the precision-recall curve for 30-day purchase readiness | Primary timing metric under class imbalance |
| ROC-AUC | Probability that a purchaser is ranked above a non-purchaser | Secondary ranking measure |
| Binary Brier score | Mean squared error of the 30-day probability | Readiness calibration control |
| Recall at top 20% | Share of all 30-day purchasers captured in the highest-scored 20% | Connects ranking to contact capacity |
| Lift at top 20% | Purchase rate in the highest-scored 20% divided by the portfolio rate | Shows concentration relative to untargeted selection |
| Action score | Readiness × category probability × median category margin | Capacity heuristic, not causal value |
