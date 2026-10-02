# Titanic Survival Classifier

A comparison of three classification models — Logistic Regression, Decision Tree, 
and k-Nearest Neighbors — predicting passenger survival on the Titanic.

## Dataset

[Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data) (891 passengers).

**Features used:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
**Dropped:** Name, Ticket, Cabin (unique/mostly-missing identifiers, not predictive as-is)
**Target:** Survived (0 = perished, 1 = survived)

**Preprocessing:**
- Missing Age filled with median, missing Embarked filled with mode
- Sex and Embarked encoded to numeric
- Features scaled with `StandardScaler` (fit on training data only)
- Split 70/15/15 into train / validation / test, stratified by survival

## Approach

1. Train all three models on the training set
2. Tune `max_depth` (tree) and `n_neighbors` (kNN) with `GridSearchCV`
3. Pick each model's decision threshold by maximizing F1 on the **validation** set
4. Evaluate final performance on the **held-out test set**, which none of the three 
   models or thresholds were tuned on

## Results

![Model comparison: ROC curves, precision-recall trade-offs, threshold sensitivity, and confusion matrices](results.png)

| Model | AUC | Accuracy | Precision (Survived) | Recall (Survived) |
|---|---|---|---|---|
| Logistic Regression | 0.83 | 79% | 73% | 71% |
| Decision Tree | 0.84 | 84% | 85% | 69% |
| k-Nearest Neighbors | 0.87 | 79% | 73% | 71% |

## Takeaway

kNN has the best AUC, which means it ranks survivors vs. non-survivors best across every 
threshold. The decision tree has the best accuracy and precision at its chosen cutoff with 
only 6 false positives. So if minimizing false alarms matters more than overall ranking 
quality, it's the better choice. Which model "wins" depends on what you're optimizing for.

## Usage

```bash
pip install -r requirements.txt
python titanic_classifier.py
```

## What I'd improve next

- Engineer a title feature from Name
- Try cross-validation instead of a single validation split for threshold selection
- Test an ensemble of the three models