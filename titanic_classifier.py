# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

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

# predictions with optimized parameters
log_proba = np.asarray(log_model.predict_proba(X_test_scaled))
tree_proba = np.asarray(tree_model.predict_proba(X_test_scaled))
kNN_proba = np.asarray(kNN_model.predict_proba(X_test_scaled))
log_pred = log_model.predict(X_test_scaled)
tree_pred = tree_model.predict(X_test_scaled)
kNN_pred = kNN_model.predict(X_test_scaled)


# model evaluations and metrics

# Logistic Regression
log_cm = confusion_matrix(y_test, log_pred)
log_fpr, log_tpr, _ = roc_curve(y_test, log_proba[:, 1])
log_roc_auc = auc(log_fpr, log_tpr)

# Decision Tree
tree_cm = confusion_matrix(y_test, tree_pred)
tree_fpr, tree_tpr, _ = roc_curve(y_test, tree_proba[:, 1])
tree_roc_auc = auc(tree_fpr, tree_tpr)

# kNN
kNN_cm = confusion_matrix(y_test, kNN_pred)
kNN_fpr, kNN_tpr, _ = roc_curve(y_test, kNN_proba[:, 1])
kNN_roc_auc = auc(kNN_fpr, kNN_tpr)

# predictions with optimized parameters
log_proba = np.asarray(log_model.predict_proba(X_test_scaled))
tree_proba = np.asarray(tree_model.predict_proba(X_test_scaled))
kNN_proba = np.asarray(kNN_model.predict_proba(X_test_scaled))
    
# plotting model metrics and evaluations
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
plt.subplots_adjust(wspace=1, hspace=0.2, top=0.90)

# Position for 2x2 ROC plot 
pos_top_left = axes[0, 0].get_position()
pos_bottom_right_roc = axes[1, 1].get_position()

new_pos_roc = [
    pos_top_left.x0, 
    pos_bottom_right_roc.y0, 
    pos_bottom_right_roc.x1 - pos_top_left.x0, 
    pos_top_left.y1 - pos_bottom_right_roc.y0
]

# Position for bottom stretched plot
pos_bottom_left_span = axes[1, 2].get_position()
pos_bottom_right_span = axes[1, 4].get_position()

new_pos_bottom_span = [
    pos_bottom_left_span.x0,
    pos_bottom_left_span.y0,
    pos_bottom_right_span.x1 - pos_bottom_left_span.x0,
    pos_bottom_left_span.height
]

# Remove unused axes
axes[0, 1].remove()
axes[1, 0].remove()
axes[1, 1].remove()
axes[1, 3].remove()
axes[1, 4].remove()

# Apply positions to axes
axes[0, 0].set_position(new_pos_roc)
axes[1, 2].set_position(new_pos_bottom_span)

# ROC Curves on 2x2 grid
axes[0, 0].plot(log_fpr, log_tpr, color='purple', label=f"Log ROC Curve (AUC = {log_roc_auc:.2f})")
axes[0, 0].plot(tree_fpr, tree_tpr, color='blue', label=f"Tree ROC Curve (AUC = {tree_roc_auc:.2f})")
axes[0, 0].plot(kNN_fpr, kNN_tpr, color='orange', label=f"kNN ROC Curve (AUC = {kNN_roc_auc:.2f})")
axes[0, 0].set_xlim([0.0, 1.0])
axes[0, 0].set_ylim([0.0, 1.05])
axes[0, 0].set_xlabel("False Positive Rate")
axes[0, 0].set_ylabel("True Positive Rate")
axes[0, 0].legend(loc="lower right")
axes[0, 0].set_title('ROC Curves')

# Logistic Regression Heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=['Perished', 'Survived'])
disp.plot(ax=axes[0, 2], cmap='Purples', values_format='d', xticks_rotation=45)

# Decision Tree Heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=tree_cm, display_labels=['Perished', 'Survived'])
disp.plot(ax=axes[0, 3], cmap='Blues', values_format='d', xticks_rotation=45)

# kNN Heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=kNN_cm, display_labels=['Perished', 'Survived'])
disp.plot(ax=axes[0, 4], cmap='Oranges', values_format='d', xticks_rotation=45)


axes[1, 2].plot([0, 1], [0, 1], label="Placeholder Metric / Calibration Plot")
axes[1, 2].set_title("Stretched Bottom Plot")
axes[1, 2].set_xlabel("X Label")
axes[1, 2].set_ylabel("Y Label")
axes[1, 2].legend()

plt.suptitle("Model Comparison for Predicting Titanic Survival", x=0.5, y=0.98, fontsize=16)
plt.show()