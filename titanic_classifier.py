# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc

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


# accuracy and model evaluations
n_classes = len(np.unique(y))

# Logistic Regression
cf_log = confusion_matrix(y_test, log_proba)
fpr_log = {}
tpr_log = {}
roc_auc_log = {}
for i in range(n_classes):
    fpr_log[i], tpr_log[i], _ = roc_curve(y_true=(y_test == i), y_score=log_proba[:, i])
    roc_auc_log[i] = auc(fpr_log[i], tpr_log[i])

# Decision Tree
cf_tree = confusion_matrix(y_test, tree_proba)
fpr_tree = {}
tpr_tree = {}
roc_auc_tree = {}
for i in range(n_classes):
    fpr_tree[i], tpr_tree[i], _ = roc_curve(y_true=(y_test == i), y_score=tree_proba[:, i])
    roc_auc_tree[i] = auc(fpr_tree[i], tpr_tree[i])

# kNN
cf_kNN = confusion_matrix(y_test, kNN_proba)
fpr_kNN = {}
tpr_kNN = {}
roc_auc_kNN = {}
for i in range(n_classes):
    fpr_kNN[i], tpr_kNN[i], _ = roc_curve(y_true=(y_test == i), y_score=kNN_proba[:, i])
    roc_auc_kNN[i] = auc(fpr_kNN[i], tpr_kNN[i])