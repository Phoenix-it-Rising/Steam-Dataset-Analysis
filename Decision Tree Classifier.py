# Title: Decision Tree Classifier Using Gini Impurity
# Author: Kiana Lang
# Course: CS379 - Machine Learning
# Date: October 04, 2025
# Description:
# This script builds and evaluates a decision tree classifier using Gini impurity to predict whether a Steam game
# receives more positive than negative reviews. It uses game metadata such as price and average playtime.
# The dataset is balanced using undersampling to improve classification performance for both positive and negative reviews.
# The final model is trained and evaluated, and the decision tree is visualized and saved.
# References:
# - The CALIPSO Version 4 Automated Aerosol Classification and Lidar Ratio Selection Algorithm
# - Structure-Texture Image Decomposition—Modeling, Algorithms, and Parameter Selection
# - ML | Gini Impurity and Entropy in Decision Tree - GeeksforGeeks

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("all_data.csv", skiprows=1)
df.columns = ['index', 'appid', 'name', 'developer', 'publisher', 'score_rank', 'positive', 'negative',
              'userscore', 'owners', 'average_forever', 'average_2weeks', 'median_forever', 'median_2weeks',
              'price', 'initialprice', 'discount', 'ccu']

# Drop rows with missing values in relevant columns
df = df.dropna(subset=['positive', 'negative', 'price', 'average_forever'])

# Create binary target variable: 1 if positive > negative, else 0
df['positive_review'] = (df['positive'] > df['negative']).astype(int)

# Select features
features = ['price', 'average_forever']
X = df[features]
y = df['positive_review']

# Balance the dataset using undersampling
df_balanced = pd.concat([
    df[df['positive_review'] == 1].sample(n=df[df['positive_review'] == 0].shape[0], random_state=42),
    df[df['positive_review'] == 0]
])

X_balanced = df_balanced[features]
y_balanced = df_balanced['positive_review']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# Train decision tree classifier using Gini impurity
clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=features, class_names=['Negative', 'Positive'], filled=True)
plt.title("Balanced Decision Tree using Gini Impurity")
plt.savefig("balanced_decision_tree_gini.png")
plt.close()

# Save evaluation results
with open("balanced_model_evaluation.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Balanced decision tree model trained and evaluated.")
print("Results saved to 'balanced_model_evaluation.txt' and tree image saved to 'balanced_decision_tree_gini.png'.")
