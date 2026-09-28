# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# loading and cleaning csv
df = pd.read_csv("Titanic-Dataset.csv")
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

# encoding strings to numbers for training
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].replace('male', 0).replace('female', 1)
df['Embarked'] = df["Embarked"].replace('C', 0).replace('Q', 1).replace('S', 2)

# separating into feature and target frames
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = ['Survived']
X = df[features]
y = df[target]

# splitting and scaling data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=2)
scalar = StandardScaler().fit(X)
X_train_scaled = scalar.transform(X_train)
X_test_scaled = scalar.transform(X_test)


# training logistic regression model
log_model = LogisticRegression(solver='liblinear', random_state=2)

# training decision tree model and optimizing maximum depth
parameters = {
    "max_depth": range(1, 26)
}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=parameters)
best_depth = grid_search.best_params_["max_depth"]
tree_model = DecisionTreeClassifier(random_state=2, max_depth=best_depth)

# training kNN model and optimizing number of neighbors
parameters = {
    "n_neighbors": range(1, 21)
}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=parameters)
best_n_neighbors = grid_search.best_params_["n_neighbors"]
kNN_model = KNeighborsClassifier(n_neighbors=best_n_neighbors)
