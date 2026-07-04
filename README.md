# Kaggle Playground Series S4E1 -- Bank Churn Prediction

This is an autonomous run I did with auto-model-trainer (an earlier version with a slightly different structure than the current plugin). I pointed it at the objective file and let it explore on its own. It worked through 2 architecture classes -- linear and gbdt -- across 3 experiments, then blended the survivors with a Caruana ensemble. The thing that actually moved the needle was target encoding on `Surname`, `CustomerId`, the `GeoGender` cross, and an `AgeBin` bucket, all done with fold-safe mean target encoding so nothing leaked.

## Competition

Binary classification: predict `Exited` (whether a bank customer churns). Metric is ROC-AUC. The data is roughly 165K training rows and 110K test rows.

https://www.kaggle.com/competitions/playground-series-s4e1

## Results

Public LB **0.89102**, Private LB **0.89493**. The final ensemble scored **0.8958** CV ROC-AUC.

| Model | Class | CV ROC-AUC |
|---|---|---|
| Ensemble (Caruana) | blend | 0.8958 |
| exp_002_xgb | gbdt | 0.8954 |
| exp_001_lgbm | gbdt | 0.8952 |
| exp_000_logreg | linear | 0.8373 |

The Caruana blend landed on XGBoost 56.7%, LightGBM 40.0%, and LogReg 3.3%. The logistic regression barely earned a seat, but it added just enough diversity to be worth keeping.

## Project Structure

```
ps-s4e1-churn/
├── objective.yaml            # the spec I handed to auto-model-trainer
├── train.csv / test.csv      # competition data
├── submission.csv            # final predictions
└── experiments/
    ├── registry.json         # every experiment with its arch class and CV score
    ├── run.py                # experiment runner
    ├── cv.py                 # cross-validation
    ├── features.py           # fold-safe target encoding and feature engineering
    ├── ensemble.py           # Caruana greedy ensemble
    ├── converge.py           # convergence detection
    ├── data_validate.py      # data quality checks
    ├── final_report.py       # report + submission generation
    └── reports/              # data validation and ensemble reports
```

## Usage

This was built with the auto-model-trainer plugin for Claude Code. With the plugin installed:

```
/auto-train objective.yaml
```

That single command drives the whole thing -- data validation, baseline, experiment exploration, ensembling, convergence, and the final submission.

https://github.com/Hook12aaa/auto-model-trainer
