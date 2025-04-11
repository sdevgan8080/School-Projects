# @Author - Sahil Devgan | Assignment 13 | April 11, 2025

# ------------------ Imports ------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# ------------------ Load Data ------------------
# Main dataset for segmentation
test = pd.read_csv('Test.csv')

# Train dataset (with known segments) for validation
train = pd.read_csv('Train.csv')

# ------------------ Data Overview ------------------
print("Test data shape:", test.shape)
print(test.head())

print("\nTest Data Info:")
print(test.info())

print("\nMissing values in Test Data:")
print(test.isnull().sum())

print("\nSummary statistics for numerical columns in Test Data:")
print(test.describe())

print("\nCategory value counts in Test Data:")
print("Gender:\n", test['Gender'].value_counts())
print("\nEver_Married:\n", test['Ever_Married'].value_counts(dropna=False))
print("\nGraduated:\n", test['Graduated'].value_counts(dropna=False))
print("\nProfession (top 5):\n", test['Profession'].value_counts().head())
print("\nSpending_Score:\n", test['Spending_Score'].value_counts())
print("\nVar_1:\n", test['Var_1'].value_counts(dropna=False))

# ------------------ Visualizations ------------------
plt.figure(figsize=(6, 4))
plt.hist(test['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(5, 4))
sns.countplot(x='Spending_Score', data=test, order=['Low', 'Average', 'High'], palette='pastel')
plt.title('Spending Score Distribution')
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(x='Profession', data=test, order=test['Profession'].value_counts().index, palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title('Customer Count by Profession')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
sns.countplot(x='Ever_Married', hue='Spending_Score', data=test, hue_order=['Low', 'Average', 'High'], palette='Set2')
plt.title('Spending Score by Marital Status')
plt.show()

# ------------------ Data Cleaning ------------------
# Fill missing categorical values with 'Unknown'
for col in ['Ever_Married', 'Graduated', 'Profession', 'Var_1']:
    test[col].fillna('Unknown', inplace=True)

# Fill missing numeric values with median
test['Work_Experience'].fillna(test['Work_Experience'].median(), inplace=True)
test['Family_Size'].fillna(test['Family_Size'].median(), inplace=True)

print("\nMissing values after imputation:")
print(test.isnull().sum())

# ------------------ Encoding & Scaling ------------------
categorical_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
numeric_cols = ['Age', 'Work_Experience', 'Family_Size']

# One-hot encoding
test_dummies = pd.get_dummies(test.drop(columns=['ID']), columns=categorical_cols, drop_first=False)

# Normalize numeric features
scaler = MinMaxScaler()
test_dummies[numeric_cols] = scaler.fit_transform(test_dummies[numeric_cols])

print("\nPreprocessed feature columns:", test_dummies.columns.tolist()[:10], "...")
print("Total features after encoding:", test_dummies.shape[1])

# ------------------ Elbow Method ------------------
inertia = []
K_range = range(2, 7)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(test_dummies)
    inertia.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# ------------------ Clustering ------------------
# Based on elbow method, let's choose k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
test['Cluster'] = kmeans.fit_predict(test_dummies)

# ------------------ Cluster Profiling ------------------
cluster_profiles = test.groupby('Cluster').agg({
    'Age': 'mean',
    'Work_Experience': 'mean',
    'Family_Size': 'mean',
    'Gender': lambda x: x.value_counts().idxmax(),
    'Ever_Married': lambda x: x.value_counts().idxmax(),
    'Graduated': lambda x: x.value_counts().idxmax(),
    'Spending_Score': lambda x: x.value_counts().idxmax(),
    'Profession': lambda x: x.value_counts().idxmax()
}).round(2)

print("\nCluster Profile (mean/mode values):")
print(cluster_profiles)

# ------------------ Validation with Train Set ------------------
train_processed = train.copy()

# Fill missing
for col in ['Ever_Married', 'Graduated', 'Profession', 'Var_1']:
    train_processed[col].fillna('Unknown', inplace=True)

train_processed['Work_Experience'].fillna(train_processed['Work_Experience'].median(), inplace=True)
train_processed['Family_Size'].fillna(train_processed['Family_Size'].median(), inplace=True)

# One-hot encode same categorical columns
train_dummies = pd.get_dummies(train_processed.drop(columns=['ID', 'Segmentation']),
                               columns=categorical_cols, drop_first=False)

# Align train columns with test columns
for col in test_dummies.columns:
    if col not in train_dummies.columns:
        train_dummies[col] = 0

train_dummies = train_dummies[test_dummies.columns]

# Scale numeric features
train_dummies[numeric_cols] = scaler.transform(train_dummies[numeric_cols])

# Predict cluster for train data
train['PredCluster'] = kmeans.predict(train_dummies)

# Cross-tab for validation
print("\nCross-tab: Actual Train Segmentation vs. Predicted Cluster")
print(pd.crosstab(train['Segmentation'], train['PredCluster']))