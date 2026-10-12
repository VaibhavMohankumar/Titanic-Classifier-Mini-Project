# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, ValidationCurveDisplay, LearningCurveDisplay, train_test_split, cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, precision_recall_curve

# loading and dropping unwanted columns
df = pd.read_csv('Titanic-Dataset.csv')
df = df.drop(['Ticket', 'Cabin'], axis=1)

# title feature engineering i.e. making the 'Name' data useful
df['Title'] = df['Name'].str.extract(r',\s*([^.]*)\.')
df = df.drop('Name', axis=1)
title_map = {
    'Mr': 'Mr',
    'Miss': 'Miss', 'Mlle': 'Miss', 'Ms': 'Miss',
    'Mrs': 'Mrs', 'Mme': 'Mrs',
    'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Major': 'Rare', 'Col': 'Rare', 'Countess': 'Rare',
    'Capt': 'Rare', 'Sir': 'Rare', 'Lady': 'Rare', 'Don': 'Rare', 'Jonkheer': 'Rare'
}
df['Title'] = df['Title'].map(title_map).fillna('Rare')

# separating into feature and target frames
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']
target = 'Survived'
X = df[features]
y = df[target]

# splitting data for training
X_temp, X_test, y_temp, y_test = train_test_split(X, y, train_size=0.85, stratify=y, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, train_size=0.8235, stratify=y_temp, random_state=2)

# pipeline
num_pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
cat_pipe = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore', drop='if_binary')
)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipe, ['Age', 'SibSp', 'Parch', 'Fare']),
        ('cat', cat_pipe, ['Embarked', 'Pclass', 'Sex', 'Title'])
    ]
)

log_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('logisticregression', LogisticRegression(solver='liblinear', max_iter=1000))])
tree_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('decisiontreeclassifier', DecisionTreeClassifier(random_state=2))])
kNN_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('kneighborsclassifier', KNeighborsClassifier())])

# training logistic regression model and optimizing C
parameters = {'logisticregression__C': np.logspace(-3, 3, 13)}
grid_search = GridSearchCV(log_pipe, param_grid=parameters, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
log_model = grid_search.best_estimator_
best_C = grid_search.best_params_['logisticregression__C']

# training decision tree model and optimizing maximum depth
parameters = {'decisiontreeclassifier__max_depth': range(1, 26)}
grid_search = GridSearchCV(tree_pipe, param_grid=parameters, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
tree_model = grid_search.best_estimator_
best_depth = grid_search.best_params_['decisiontreeclassifier__max_depth']

# training kNN model and optimizing number of neighbors
parameters = {'kneighborsclassifier__n_neighbors': range(1, 21)}
grid_search = GridSearchCV(kNN_pipe, param_grid=parameters, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
kNN_model = grid_search.best_estimator_
best_n_neighbors = grid_search.best_params_['kneighborsclassifier__n_neighbors']

# print best parameters
print(best_C)
print(best_depth)
print(best_n_neighbors)

# proabilities with optimized parameters
log_val_proba = np.asarray(log_model.predict_proba(X_val))[:, 1]
tree_val_proba = np.asarray(tree_model.predict_proba(X_val))[:, 1]
kNN_val_proba = np.asarray(kNN_model.predict_proba(X_val))[:, 1]


# Best model thresholds
log_precisions, log_recalls, log_thresholds = precision_recall_curve(y_val, log_val_proba)
tree_precisions, tree_recalls, tree_thresholds = precision_recall_curve(y_val, tree_val_proba)
kNN_precisions, kNN_recalls, kNN_thresholds = precision_recall_curve(y_val, kNN_val_proba)

log_f1_scores = 2 * (log_precisions[:-1] * log_recalls[:-1]) / (log_precisions[:-1] + log_recalls[:-1])
log_best_idx = np.argmax(log_f1_scores)
log_best_threshold = log_thresholds[log_best_idx]

tree_f1_scores = 2 * (tree_precisions[:-1] * tree_recalls[:-1]) / (tree_precisions[:-1] + tree_recalls[:-1])
tree_best_idx = np.argmax(tree_f1_scores)
tree_best_threshold = tree_thresholds[tree_best_idx]

kNN_f1_scores = 2 * (kNN_precisions[:-1] * kNN_recalls[:-1]) / (kNN_precisions[:-1] + kNN_recalls[:-1])
kNN_best_idx = np.argmax(kNN_f1_scores)
kNN_best_threshold = kNN_thresholds[kNN_best_idx]

# final probabilities on test set
log_proba = np.asarray(log_model.predict_proba(X_test))[:, 1]
tree_proba = np.asarray(tree_model.predict_proba(X_test))[:, 1]
kNN_proba = np.asarray(kNN_model.predict_proba(X_test))[:, 1]

# predictions based on optimized parameters and on test set
log_pred = (log_proba >= log_best_threshold).astype(int)
tree_pred = (tree_proba >= tree_best_threshold).astype(int)
kNN_pred = (kNN_proba >= kNN_best_threshold).astype(int)


# model evaluations and metrics

# Logistic Regression
log_cm = confusion_matrix(y_test, log_pred)
log_fpr, log_tpr, _ = roc_curve(y_test, log_proba)
log_roc_auc = auc(log_fpr, log_tpr)

# Decision Tree
tree_cm = confusion_matrix(y_test, tree_pred)
tree_fpr, tree_tpr, _ = roc_curve(y_test, tree_proba)
tree_roc_auc = auc(tree_fpr, tree_tpr)

# kNN
kNN_cm = confusion_matrix(y_test, kNN_pred)
kNN_fpr, kNN_tpr, _ = roc_curve(y_test, kNN_proba)
kNN_roc_auc = auc(kNN_fpr, kNN_tpr)


# plotting model evaluations
fig, axes = plt.subplots(3, 5, figsize=(30, 15))
plt.subplots_adjust(wspace=0.7, hspace=0.4, top=0.90)

# Position for 2x2 ROC plot 
pos_top_left = axes[0, 0].get_position()
pos_bottom_right_roc = axes[2, 1].get_position()

new_pos_roc = [
    pos_top_left.x0, 
    pos_bottom_right_roc.y0, 
    pos_bottom_right_roc.x1 - pos_top_left.x0, 
    pos_top_left.y1 - pos_bottom_right_roc.y0
]

# Remove unused axes
axes[0, 1].remove()
axes[1, 0].remove()
axes[1, 1].remove()
axes[2, 0].remove()
axes[2, 1].remove()

# Apply positions to axes
axes[0, 0].set_position(new_pos_roc)

# ROC Curves on 2x2 grid
axes[0, 0].plot(log_fpr, log_tpr, color='purple', label=f"Log ROC Curve (AUC = {log_roc_auc:.2f})")
axes[0, 0].plot(tree_fpr, tree_tpr, color='blue', label=f"Tree ROC Curve (AUC = {tree_roc_auc:.2f})")
axes[0, 0].plot(kNN_fpr, kNN_tpr, color='orange', label=f"kNN ROC Curve (AUC = {kNN_roc_auc:.2f})")
axes[0, 0].set_xlim([0.0, 1.0])
axes[0, 0].set_ylim([0.0, 1.05])
axes[0, 0].set_xlabel("False Positive Rate")
axes[0, 0].set_ylabel("True Positive Rate")
axes[0, 0].legend(loc="lower right")
axes[0, 0].set_title("ROC Curves")

def plot_confusion_matrix(axes, cm, title, color):
    """
    Displays a formatted confusion matrix using Matplotlib and scikit-learn's ConfusionMatrixDisplay.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The Matplotlib subplot axes object where the confusion matrix will be rendered.
    cm : array-like of shape (n_classes, n_classes)
        The calculated confusion matrix values to display.
    title : str
        Base string used to construct the plot title.
    color : str or matplotlib.colors.Colormap
        Color map applied to the rendered confusion matrix heat map.

    Returns
    -------
    None
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Perished', 'Survived'])
    disp.plot(ax=axes, cmap=color, xticks_rotation=45, colorbar=False)
    axes.tick_params(axis='y', rotation=45)
    axes.set_title(f"{title}\nConfusion Matrix", fontsize='small')

# confusion matrix plots for all 3 models
plot_confusion_matrix(axes[0, 3], log_cm, "logisticregression", 'Purples')
plot_confusion_matrix(axes[1, 3], tree_cm, "decisiontreeclassifier", 'Blues')
plot_confusion_matrix(axes[2, 3], kNN_cm, "kneighborsclassifier", 'Oranges')

# Show realistic evaluations
log_scores = cross_val_score(log_model, X_temp, y_temp, cv=5, scoring="roc_auc")
tree_scores = cross_val_score(tree_model, X_temp, y_temp, cv=5, scoring="roc_auc")
kNN_scores = cross_val_score(kNN_model, X_temp, y_temp, cv=5, scoring="roc_auc")
print(log_scores.mean(), log_scores.std())
print(tree_scores.mean(), tree_scores.std())
print(kNN_scores.mean(), kNN_scores.std())

# Show visual representation of a GridSearchCV
models_config = [
    {
        "name": "logisticregression",
        "estimator": log_model,
        "param_name": "logisticregression__C",
        "param_range": np.logspace(-3, 3, 13),
        "scale": "log",
        "chosen_param": best_C
    },
    {
        "name": "decisiontreeclassifier",
        "estimator": tree_model,
        "param_name": "decisiontreeclassifier__max_depth",
        "param_range": np.arange(1, 26),
        "scale": "linear",
        "chosen_param": best_depth
    },
    {
        "name": "kneighborsclassifier",
        "estimator": kNN_model,
        "param_name": "kneighborsclassifier__n_neighbors",
        "param_range": np.arange(1, 21),
        "scale": "linear",
        "chosen_param": best_n_neighbors
    }
]

count = 0
for config in models_config:
    ValidationCurveDisplay.from_estimator(
        estimator=config["estimator"],
        X=X_train,
        y=y_train,
        param_name=config["param_name"],
        param_range=config["param_range"],
        cv=5,
        scoring="roc_auc",
        ax=axes[count, 2]
    )
    
    
    axes[count, 2].set_title(f"{config['name']}\n(param: {config['param_name']})")
    axes[count, 2].axvline(config['chosen_param'], color='red', linestyle=':', label=f'Best hyperparameter value ({config['chosen_param']})')
    axes[count, 2].set_xscale(config["scale"])
    axes[count, 2].grid(True, linestyle="--", alpha=0.6)
    handles, _ = axes[count, 2].get_legend_handles_labels()
    axes[count, 2].legend(handles=handles, labels=["Train", "CV", "Best hyperparameter value"], fontsize='xx-small')
    count += 1
    
# learning curve for models - shows how model accuracy changes with change in training set size
learning_curve_params = {
    "X": X_train,
    "y": y_train,
    "cv": 5,
    "scoring": "roc_auc",
    "line_kw": {"marker": "o"},
    "score_name": "roc_auc"
}

count = 0
for model, config in zip([log_model, tree_model, kNN_model], models_config):
    LearningCurveDisplay.from_estimator(model, **learning_curve_params, ax=axes[count, 4])
    axes[count, 4].set_title(f"{config['name']}")
    axes[count, 4].set_xlabel("num training samples")
    handles, _ = axes[count, 4].get_legend_handles_labels()
    axes[count, 4].legend(handles=handles, labels=["Train", "CV"])
    count += 1

plt.suptitle("Model Comparison for Predicting Titanic Survival", x=0.5, y=0.98, fontsize=16)
plt.savefig('results.png', bbox_inches='tight')

# model threshold plots
def plot_precision_recall_curve(axes, recalls, precisions, best_idx, best_threshold, title, color):
    """
    Plots the precision-recall curve along with a highlighted point for the best threshold.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The Matplotlib subplot axes object where the curve will be drawn.
    recalls : array-like
        Array of recall values corresponding to different evaluation thresholds.
    precisions : array-like
        Array of precision values corresponding to different evaluation thresholds.
    best_idx : int
        Index corresponding to the optimal threshold that maximizes the F1-score.
    best_threshold : float
        The numerical threshold value corresponding to the optimal point.
    title : str
        Base string used to construct the plot title.
    color : str
        Color used to plot the Precision-Recall line.

    Returns
    -------
    None
    """
    axes.plot(recalls, precisions, color=color, lw=2, label='PR Curve')
    axes.scatter(recalls[best_idx], precisions[best_idx], color='red', marker='o', s=100, label=f'Max F1 Threshold: {best_threshold:.2f}', zorder=5)
    axes.set_xlabel('Recall')
    axes.set_ylabel('Precision')
    axes.set_title(f'{title}' + '\nPrecision-Recall Trade-off', fontsize='medium')
    axes.grid(True, linestyle='--', alpha=0.5)
    axes.legend(loc='lower left', fontsize='xx-small')
    
def plot_metric_threshold(axes, precisions, recalls, f1_scores, thresholds, best_threshold, title, color):
    """
    Plots Precision, Recall, and F1-score metrics across a range of probability decision thresholds.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The Matplotlib subplot axes object where the curves will be drawn.
    precisions : array-like
        Array of precision values for each threshold (length must match thresholds + 1).
    recalls : array-like
        Array of recall values for each threshold (length must match thresholds + 1).
    f1_scores : array-like
        Array of calculated F1-score values corresponding to each decision threshold.
    thresholds : array-like
        Array of decision threshold probability values.
    best_threshold : float
        The chosen probability cutoff threshold to highlight with a vertical reference line.
    title : str
        Base string used to construct the plot title.
    color : str
        Color used for the vertical cutoff reference line.

    Returns
    -------
    None
    """
    axes.plot(thresholds, precisions[:-1], label='Precision', color='green', linestyle='--')
    axes.plot(thresholds, recalls[:-1], label='Recall', color='black', linestyle='--')
    axes.plot(thresholds, f1_scores, label='F1-Score', color='red', lw=2)
    axes.axvline(best_threshold, color=color, linestyle=':', lw=2, label=f'Chosen Cutoff ({best_threshold:.2f})')
    axes.set_xlabel('Probability Decision Threshold', fontsize='small')
    axes.set_ylabel('Score Metric')
    axes.set_title(f'{title}\nThreshold Metric Shift', fontsize='medium')
    axes.grid(True, linestyle='--', alpha=0.5)
    axes.legend(loc='lower left', fontsize='xx-small')
    
fig_thresh, axes_thresh = plt.subplots(3, 2, figsize=(15, 10))
plt.subplots_adjust(wspace=0.7, hspace=0.4, top=0.90)

# precisio-recall curves for all 3 models
plot_precision_recall_curve(axes_thresh[0, 0], log_recalls, log_precisions, log_best_idx, log_best_threshold, "logisticregression", 'purple')
plot_precision_recall_curve(axes_thresh[1, 0], tree_recalls, tree_precisions, tree_best_idx, tree_best_threshold, "decisiontreeclassifier", 'blue')
plot_precision_recall_curve(axes_thresh[2, 0], kNN_recalls, kNN_precisions, kNN_best_idx, kNN_best_threshold, "kneighborsclassifier", 'orange')

# metric-threshold plots for all 3 models
plot_metric_threshold(axes_thresh[0, 1], log_precisions, log_recalls, log_f1_scores, log_thresholds, log_best_threshold, "logisticregression", 'purple')
plot_metric_threshold(axes_thresh[1, 1], tree_precisions, tree_recalls, tree_f1_scores, tree_thresholds, tree_best_threshold, "decisiontreeclassifier", 'blue')
plot_metric_threshold(axes_thresh[2, 1], kNN_precisions, kNN_recalls, kNN_f1_scores, kNN_thresholds, kNN_best_threshold, "kneighborsclassifier", 'orange')

plt.suptitle("Model threshold plots", x=0.5, y=0.98, fontsize=16)
plt.savefig('thresholds.png', bbox_inches='tight')