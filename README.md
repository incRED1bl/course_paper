# Respiratory Sound Classification

Course research project at HSE University. The model classifies respiratory sounds into three groups: healthy, chronic obstructive pulmonary disease (COPD), and respiratory infection. The acoustic features include Bandt–Pompe permutation entropy, a measure of signal complexity.

[Open the final notebook](colab/model/final_pipeline.ipynb) for the modeling pipeline, confusion matrix, and saved evaluation results.

## Evaluation

- Split the data by patient ID so that no patient appears in both the training and test partitions. The test partition is one fold of a stratified group split.
- Used a conditional variational autoencoder (CVAE) to generate additional feature vectors for underrepresented classes in the training partition.
- Chose the threshold for predicting a disease class by maximizing macro-F₂ on a validation subset of the training partition. F₂ gives more weight to recall than precision, and macro averaging gives each class equal weight.

## Held-out test results

| Metric | Score |
| --- | ---: |
| Macro-F₂ | **0.8391** |
| Weighted-F₂ | 0.8911 |
| Macro-F₁ | 0.8415 |

| Class | Recall | Test examples |
| --- | ---: | ---: |
| Healthy | 0.7429 | 35 |
| COPD | 0.9471 | 170 |
| Respiratory infection | 0.8226 | 62 |

The notebook uses the [prepared feature table](colab/data/enlarged_dataset.csv). Adjust the CSV path in the data-loading cell if you run it outside the original Colab environment.
