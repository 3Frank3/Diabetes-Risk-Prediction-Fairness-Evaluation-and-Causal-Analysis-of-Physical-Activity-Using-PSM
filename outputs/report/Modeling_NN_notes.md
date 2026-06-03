# ANN notes

This file just explains the final ANN figures in simple terms.

## Initial ANN

The initial ANN used one hidden layer with 32 neurons. It is the basic version we compare everything against.

## Optimized ANN

The optimized ANN came from the small Keras search. The best setting was `(64, 32)` with `sigmoid` activation and learning rate `0.001`.

## Why threshold 0.15 stayed

Threshold 0.15 was kept because this project treats the ANN as a screening model. In the threshold table for the initial ANN, 0.20 gives a slightly higher F1-score, but 0.15 keeps recall noticeably higher. That means fewer diabetes-positive cases are missed, which fits the recall-first goal better.

## How to read the comparison table

The combined table is there to show the initial ANN and optimized ANN side by side. The optimized model is not dramatically different on every metric, but it keeps slightly better recall while staying close on PR-AUC and Brier score.

## How to read the error analysis

For the optimized ANN, the key part is the balance between true positives and false negatives. A false negative means the model missed a diabetes-positive case. In a screening setting that matters more than getting extra false positives.

## Main numbers

- Initial ANN recall: 0.754
- Optimized ANN recall: 0.780
- Initial ANN PR-AUC: 0.417
- Optimized ANN PR-AUC: 0.421
- Optimized ANN false negative rate: 0.220
