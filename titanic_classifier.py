# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, precision_recall_curve

# loading and cleaning csv
df = pd.read_csv("Titanic-Dataset.csv")
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

# encoding strings to numbers for training
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].replace('male', 0).replace('female', 1)
df['Embarked'] = df["Embarked"].replace('C', 0).replace('Q', 1).replace('S', 2)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# separating into feature and target frames
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'
X = df[features]
y = df[target]

# splitting and scaling data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=2)
scalar = StandardScaler().fit(X)
X_train_scaled = scalar.transform(X_train)
X_test_scaled = scalar.transform(X_test)


# training logistic regression model
log_model = LogisticRegression(solver='liblinear', max_iter=1000)
log_model.fit(X_train_scaled, y_train)

# training decision tree model and optimizing maximum depth
parameters = {
    "max_depth": range(1, 26)
}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=parameters)
grid_tree = grid_search.fit(X_train_scaled, y_train)
best_depth = grid_tree.best_params_["max_depth"]
tree_model = DecisionTreeClassifier(random_state=2, max_depth=best_depth)
tree_model.fit(X_train_scaled, y_train)

# training kNN model and optimizing number of neighbors
parameters = {
    "n_neighbors": range(1, 21)
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid=parameters)
grid_kNN = grid_search.fit(X_train_scaled, y_train)
best_n_neighbors = grid_kNN.best_params_["n_neighbors"]
kNN_model = KNeighborsClassifier(n_neighbors=best_n_neighbors)
kNN_model.fit(X_train_scaled, y_train)

# proabilities and predictions with optimized parameters
log_proba = np.asarray(log_model.predict_proba(X_test_scaled))[:, 1]
tree_proba = np.asarray(tree_model.predict_proba(X_test_scaled))[:, 1]
kNN_proba = np.asarray(kNN_model.predict_proba(X_test_scaled))[:, 1]
# log_pred = log_model.predict(X_test_scaled)
# tree_pred = tree_model.predict(X_test_scaled)
# kNN_pred = kNN_model.predict(X_test_scaled)


# Best model thresholds
log_precisions, log_recalls, log_thresholds = precision_recall_curve(y_test, log_proba)
tree_precisions, tree_recalls, tree_thresholds = precision_recall_curve(y_test, tree_proba)
kNN_precisions, kNN_recalls, kNN_thresholds = precision_recall_curve(y_test, kNN_proba)

log_f1_scores = 2 * (log_precisions[:-1] * log_recalls[:-1]) / (log_precisions[:-1] + log_recalls[:-1])
log_best_idx = np.argmax(log_f1_scores)
log_best_threshold = log_thresholds[log_best_idx]

tree_f1_scores = 2 * (tree_precisions[:-1] * tree_recalls[:-1]) / (tree_precisions[:-1] + tree_recalls[:-1])
tree_best_idx = np.argmax(tree_f1_scores)
tree_best_threshold = tree_thresholds[tree_best_idx]

kNN_f1_scores = 2 * (kNN_precisions[:-1] * kNN_recalls[:-1]) / (kNN_precisions[:-1] + kNN_recalls[:-1])
kNN_best_idx = np.argmax(kNN_f1_scores)
kNN_best_threshold = kNN_thresholds[kNN_best_idx]

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


# plotting model metrics and evaluations
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
plt.subplots_adjust(wspace=0.7, hspace=0.2, top=0.90)

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

# Logistic Regression Heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=['Perished', 'Survived'])
disp.plot(ax=axes[0, 2], cmap='Purples', xticks_rotation=45, colorbar=False)
axes[0, 2].tick_params(axis='y', rotation=45)
axes[0, 2].set_title("Logistic Regression Metrics")

# Decision Tree Heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=tree_cm, display_labels=['Perished', 'Survived'])
disp.plot(ax=axes[0, 3], cmap='Blues', xticks_rotation=45, colorbar=False)
axes[0, 3].tick_params(axis='y', rotation=45)
axes[0, 3].set_title("Decision Tree Metrics")

# kNN Heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=kNN_cm, display_labels=['Perished', 'Survived'])
disp.plot(ax=axes[0, 4], cmap='Oranges', xticks_rotation=45, colorbar=False)
axes[0, 4].tick_params(axis='y', rotation=45)
axes[0, 4].set_title("K-Nearest Neighbors Metrics")

plt.suptitle("Model Comparison for Predicting Titanic Survival", x=0.5, y=0.98, fontsize=16)
plt.show()