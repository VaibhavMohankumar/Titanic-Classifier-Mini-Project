# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Loading and cleaning csv
df = pd.read_csv("Titanic-Dataset.csv")
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

# encoding strings to numbers for training
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Sex"] = df["Sex"].replace("male", 0).replace("female", 1)